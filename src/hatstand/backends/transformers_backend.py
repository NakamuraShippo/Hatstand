from __future__ import annotations

import gc
import importlib.util
import json
import logging
from pathlib import Path
import re
import sys
from threading import Thread
from time import perf_counter
from typing import Any
from collections.abc import Callable

from hatstand.domain.dtos import (
    CancelToken,
    GenerateProgress,
    GenerateRequest,
    GenerateResult,
    LoadOptions,
    LoadResult,
    WarmupResult,
)
from hatstand.domain.enums import DeviceMap, DType, Quantization


class BackendConfigurationError(RuntimeError):
    """Raised when the backend configuration is invalid."""


class _CancelStoppingCriteria:
    def __init__(self, cancel_token: CancelToken) -> None:
        self._cancel_token = cancel_token

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        return self._cancel_token.cancelled


class TransformersBackend:
    def __init__(
        self, cache_dir: str = "", logger: logging.Logger | None = None
    ) -> None:
        self._cache_dir = cache_dir or None
        self._logger = logger or logging.getLogger(__name__)
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.current_model_id: str = ""
        self.current_options: LoadOptions | None = None
        self.current_model_path: str = ""
        self.current_model_architecture: str = ""

    def update_cache_dir(self, cache_dir: str) -> None:
        self._cache_dir = cache_dir or None

    def load_model(self, options: LoadOptions) -> LoadResult:
        self._validate_options(options)
        started_at = perf_counter()
        try:
            if self.is_loaded():
                self.unload_model()
            torch, transformers = self._import_runtime()
            model_source = self._ensure_local_model_source(
                options.model_id, revision=options.revision or None
            )
            architecture = self._detect_model_architecture(model_source)
            quantization_config = self._build_quantization_config(
                transformers, options.quant
            )
            model_kwargs = {
                "trust_remote_code": True,
                "cache_dir": self._cache_dir,
                "local_files_only": True,
                "torch_dtype": self._resolve_torch_dtype(torch, options.dtype),
            }
            device_map = self._resolve_device_map(torch, options.device_map)
            if device_map is not None:
                model_kwargs["device_map"] = device_map
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            self._logger.info(
                "load_model | model_id=%s local_path=%s architecture=%s quant=%s dtype=%s device_map=%s",
                options.model_id,
                model_source,
                architecture,
                options.quant,
                options.dtype,
                options.device_map,
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                cache_dir=self._cache_dir,
                local_files_only=True,
            )
            if (
                self.tokenizer.pad_token is None
                and self.tokenizer.eos_token is not None
            ):
                self.tokenizer.pad_token = self.tokenizer.eos_token
            model_loader = self._resolve_model_loader(transformers, architecture)
            self.model = model_loader.from_pretrained(model_source, **model_kwargs)
            self.current_model_id = options.model_id
            self.current_options = options
            self.current_model_path = model_source
            self.current_model_architecture = architecture
            return LoadResult(
                success=True,
                model_id=options.model_id,
                load_seconds=perf_counter() - started_at,
            )
        except Exception as exc:
            self.unload_model()
            self._logger.exception("load_model_failed | model_id=%s", options.model_id)
            return LoadResult(
                success=False,
                model_id=options.model_id,
                load_seconds=perf_counter() - started_at,
                error_message=self._format_load_error(exc),
            )

    def unload_model(self) -> None:
        self._logger.info("unload_model | model_id=%s", self.current_model_id)
        model = self.model
        tokenizer = self.tokenizer
        try:
            if model is not None:
                try:
                    from accelerate.hooks import remove_hook_from_module

                    remove_hook_from_module(model, recurse=True)
                except Exception:
                    self._logger.debug(
                        "unload_model | accelerate hook removal skipped", exc_info=True
                    )
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            self._logger.debug(
                "unload_model | model cpu offload skipped", exc_info=True
            )
        self.model = None
        self.tokenizer = None
        self.current_model_id = ""
        self.current_options = None
        self.current_model_path = ""
        self.current_model_architecture = ""
        try:
            from accelerate.utils import release_memory

            model, tokenizer = release_memory(model, tokenizer)
        except Exception:
            del model
            del tokenizer
            gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    self._logger.debug(
                        "unload_model | cuda synchronize skipped", exc_info=True
                    )
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        self._logger.debug(
                            "unload_model | cuda ipc_collect skipped", exc_info=True
                        )
        except ImportError:
            pass

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def generate(
        self,
        request: GenerateRequest,
        cancel_token: CancelToken,
        stream_callback: Callable[[GenerateProgress], None] | None = None,
    ) -> GenerateResult:
        if not self.is_loaded():
            return GenerateResult(
                success=False,
                text="",
                first_token_seconds=0.0,
                total_seconds=0.0,
                output_tokens=0,
                tokens_per_second=0.0,
                error_message="No model is loaded.",
                output_mode=request.output_mode,
            )
        started_at = perf_counter()
        try:
            torch, transformers = self._import_runtime()
            encoded = self._build_model_inputs(
                request.messages, enable_thinking=request.enable_thinking
            )
            encoded = self._move_inputs_to_model_device(encoded)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": request.max_new_tokens,
                "repetition_penalty": request.repetition_penalty,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "stopping_criteria": transformers.StoppingCriteriaList(
                    [_CancelStoppingCriteria(cancel_token)]
                ),
                "use_cache": True,
            }
            if request.seed is not None:
                torch.manual_seed(request.seed)
            if request.temperature > 0:
                generation_kwargs["do_sample"] = True
                generation_kwargs["temperature"] = request.temperature
                generation_kwargs["top_p"] = request.top_p
            else:
                generation_kwargs["do_sample"] = False
            input_length = int(encoded["input_ids"].shape[-1])
            self._logger.info(
                "generate | model_id=%s architecture=%s prompt_tokens=%s max_new_tokens=%s",
                self.current_model_id,
                self.current_model_architecture or "unknown",
                input_length,
                request.max_new_tokens,
            )
            if stream_callback is not None and hasattr(
                transformers, "TextIteratorStreamer"
            ):
                return self._generate_with_streamer(
                    transformers=transformers,
                    request=request,
                    cancel_token=cancel_token,
                    encoded=encoded,
                    generation_kwargs=generation_kwargs,
                    input_length=input_length,
                    started_at=started_at,
                    stream_callback=stream_callback,
                )
            output_ids = self.model.generate(**encoded, **generation_kwargs)
            generated_ids = output_ids[0][input_length:]
            text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            total_seconds = perf_counter() - started_at
            output_tokens = (
                int(generated_ids.shape[-1]) if hasattr(generated_ids, "shape") else 0
            )
            if cancel_token.cancelled and not text:
                return GenerateResult(
                    success=False,
                    text="",
                    first_token_seconds=0.0,
                    total_seconds=total_seconds,
                    output_tokens=0,
                    tokens_per_second=0.0,
                    error_message="Generation stopped before output was produced.",
                    output_mode=request.output_mode,
                )
            result = GenerateResult(
                success=True,
                text=text,
                first_token_seconds=total_seconds if output_tokens else 0.0,
                total_seconds=total_seconds,
                output_tokens=output_tokens,
                tokens_per_second=(
                    (output_tokens / total_seconds) if total_seconds > 0 else 0.0
                ),
                error_message="Stopped by user." if cancel_token.cancelled else None,
                output_mode=request.output_mode,
            )
            self._logger.info(
                "generate_finished | model_id=%s total_seconds=%.3f output_tokens=%s",
                self.current_model_id,
                total_seconds,
                output_tokens,
            )
            return result
        except Exception as exc:
            self._logger.exception(
                "generate_failed | model_id=%s", self.current_model_id
            )
            return GenerateResult(
                success=False,
                text="",
                first_token_seconds=0.0,
                total_seconds=perf_counter() - started_at,
                output_tokens=0,
                tokens_per_second=0.0,
                error_message=str(exc),
                output_mode=request.output_mode,
            )

    def warmup(self) -> WarmupResult:
        if not self.is_loaded():
            return WarmupResult(
                success=False, total_seconds=0.0, error_message="No model is loaded."
            )
        started_at = perf_counter()
        result = self.generate(
            GenerateRequest(
                messages=[{"role": "user", "content": "Respond with the word ready."}],
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=8,
                repetition_penalty=1.0,
            ),
            CancelToken(),
        )
        return WarmupResult(
            success=result.success,
            total_seconds=perf_counter() - started_at,
            error_message=result.error_message,
        )

    def _build_model_inputs(
        self, messages: list[dict[str, str]], enable_thinking: bool = False
    ) -> Any:
        if hasattr(self.tokenizer, "apply_chat_template"):
            template_kwargs: dict[str, Any] = {
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "add_generation_prompt": True,
            }
            if self._supports_non_thinking_mode():
                template_kwargs["enable_thinking"] = enable_thinking
            try:
                return self.tokenizer.apply_chat_template(messages, **template_kwargs)
            except (TypeError, ValueError):
                self._logger.debug(
                    "apply_chat_template_tokenize_failed | falling back to plain tokenization"
                )
        prompt = self._build_prompt(messages, enable_thinking=enable_thinking)
        return self.tokenizer(prompt, return_tensors="pt")

    def _build_prompt(
        self, messages: list[dict[str, str]], enable_thinking: bool = False
    ) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self._supports_non_thinking_mode():
                template_kwargs["enable_thinking"] = enable_thinking
            return self.tokenizer.apply_chat_template(messages, **template_kwargs)
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )

    def _generate_with_streamer(
        self,
        transformers: Any,
        request: GenerateRequest,
        cancel_token: CancelToken,
        encoded: Any,
        generation_kwargs: dict[str, Any],
        input_length: int,
        started_at: float,
        stream_callback: Callable[[GenerateProgress], None],
    ) -> GenerateResult:
        streamer = transformers.TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        holder: dict[str, Any] = {}
        failure: dict[str, Exception] = {}
        streaming_kwargs = dict(generation_kwargs)
        streaming_kwargs["streamer"] = streamer

        def run_generate() -> None:
            try:
                holder["output_ids"] = self.model.generate(
                    **encoded, **streaming_kwargs
                )
            except Exception as exc:
                failure["error"] = exc

        worker = Thread(target=run_generate, daemon=True)
        worker.start()
        text = ""
        first_token_seconds = 0.0
        output_tokens = 0
        for delta in streamer:
            if not delta:
                continue
            text += delta
            if first_token_seconds == 0.0:
                first_token_seconds = perf_counter() - started_at
            output_tokens = self._count_output_tokens(text)
            try:
                stream_callback(
                    GenerateProgress(
                        text=text,
                        delta=delta,
                        first_token_seconds=first_token_seconds,
                        output_tokens=output_tokens,
                    )
                )
            except Exception:
                self._logger.exception(
                    "generate_stream_callback_failed | model_id=%s",
                    self.current_model_id,
                )
        worker.join()
        if "error" in failure:
            raise failure["error"]
        output_ids = holder.get("output_ids")
        if output_ids is not None:
            generated_ids = output_ids[0][input_length:]
            if hasattr(generated_ids, "shape"):
                output_tokens = int(generated_ids.shape[-1])
            if not text:
                text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                ).strip()
        text = text.strip()
        total_seconds = perf_counter() - started_at
        if cancel_token.cancelled and not text:
            return GenerateResult(
                success=False,
                text="",
                first_token_seconds=0.0,
                total_seconds=total_seconds,
                output_tokens=0,
                tokens_per_second=0.0,
                error_message="Generation stopped before output was produced.",
                output_mode=request.output_mode,
            )
        result = GenerateResult(
            success=True,
            text=text,
            first_token_seconds=first_token_seconds if output_tokens else 0.0,
            total_seconds=total_seconds,
            output_tokens=output_tokens,
            tokens_per_second=(
                (output_tokens / total_seconds) if total_seconds > 0 else 0.0
            ),
            error_message="Stopped by user." if cancel_token.cancelled else None,
            output_mode=request.output_mode,
        )
        self._logger.info(
            "generate_finished | model_id=%s total_seconds=%.3f output_tokens=%s streamed=yes",
            self.current_model_id,
            total_seconds,
            output_tokens,
        )
        return result

    def _build_quantization_config(self, transformers: Any, quant: str) -> Any | None:
        if quant == Quantization.NONE.value:
            return None
        if not self.is_bitsandbytes_available():
            raise BackendConfigurationError(
                self._build_missing_bitsandbytes_message(quant)
            )
        if quant == Quantization.INT8.value:
            return transformers.BitsAndBytesConfig(load_in_8bit=True)
        if quant == Quantization.INT4.value:
            return transformers.BitsAndBytesConfig(load_in_4bit=True)
        raise BackendConfigurationError(f"Unsupported quantization mode: {quant}")

    def _resolve_torch_dtype(self, torch: Any, dtype: str) -> Any:
        mapping = {
            DType.BFLOAT16.value: torch.bfloat16,
            DType.FLOAT16.value: torch.float16,
            DType.FLOAT32.value: torch.float32,
        }
        try:
            return mapping[dtype]
        except KeyError as exc:
            raise BackendConfigurationError(f"Unsupported dtype: {dtype}") from exc

    def _resolve_device_map(self, torch: Any, device_map: str) -> Any | None:
        if device_map == DeviceMap.AUTO.value:
            return "auto"
        if device_map == DeviceMap.CUDA.value:
            if not torch.cuda.is_available():
                raise BackendConfigurationError(
                    self._build_cuda_unavailable_message(torch)
                )
            return {"": "cuda:0"}
        if device_map == DeviceMap.CPU.value:
            return {"": "cpu"}
        raise BackendConfigurationError(f"Unsupported device_map: {device_map}")

    def _move_inputs_to_model_device(self, encoded: Any) -> Any:
        if self.model is None:
            return encoded
        model_device = getattr(self.model, "device", None)
        if model_device is None:
            return encoded
        try:
            return encoded.to(model_device)
        except (AttributeError, RuntimeError):
            return encoded

    def _validate_options(self, options: LoadOptions) -> None:
        if not options.model_id.strip():
            raise BackendConfigurationError("model_id is required.")
        supported_quant = {item.value for item in Quantization}
        supported_dtype = {item.value for item in DType}
        supported_device_map = {item.value for item in DeviceMap}
        if options.quant not in supported_quant:
            raise BackendConfigurationError(
                f"Unsupported quantization: {options.quant}"
            )
        if options.dtype not in supported_dtype:
            raise BackendConfigurationError(f"Unsupported dtype: {options.dtype}")
        if options.device_map not in supported_device_map:
            raise BackendConfigurationError(
                f"Unsupported device map: {options.device_map}"
            )

    def _import_runtime(self) -> tuple[Any, Any]:
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise BackendConfigurationError(
                "Missing dependency. Install torch and transformers to use the backend."
            ) from exc
        return torch, transformers

    def _ensure_local_model_source(
        self, model_id: str, revision: str | None = None
    ) -> str:
        local_path = Path(model_id).expanduser()
        if local_path.exists():
            resolved = str(local_path.resolve())
            self._logger.info("using_local_model_directory | path=%s", resolved)
            return resolved
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise BackendConfigurationError(
                "Missing dependency. Install huggingface_hub to download model snapshots."
            ) from exc
        self._logger.info(
            "download_model_snapshot_started | model_id=%s revision=%s cache_dir=%s",
            model_id,
            revision or "main",
            self._cache_dir or "",
        )
        download_kwargs: dict[str, Any] = {
            "repo_id": model_id,
            "cache_dir": self._cache_dir,
            "resume_download": True,
        }
        if revision:
            download_kwargs["revision"] = revision
        try:
            snapshot_path = snapshot_download(**download_kwargs)
        except Exception as exc:
            raise BackendConfigurationError(
                f"Failed to download model '{model_id}'. Check your internet connection, Hugging Face access, and cache path."
            ) from exc
        resolved_snapshot = str(Path(snapshot_path).resolve())
        self._logger.info(
            "download_model_snapshot_finished | model_id=%s snapshot_path=%s",
            model_id,
            resolved_snapshot,
        )
        return resolved_snapshot

    @staticmethod
    def normalize_model_id(raw_input: str) -> str:
        """Convert a HF URL or repo ID to a clean repo ID (e.g. 'org/model')."""
        text = raw_input.strip()
        match = re.match(r"https?://huggingface\.co/([^/]+/[^/?#]+)", text)
        if match:
            return match.group(1)
        return text

    def inspect_repo(self, repo_id: str) -> dict[str, Any]:
        """Inspect a HF repo and return format info and available branches.

        Returns a dict with keys:
        - repo_id: str
        - has_transformers: bool (safetensors/bin + config.json)
        - has_gguf: bool
        - branches: list[str]
        - error: str | None
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return {
                "repo_id": repo_id,
                "has_transformers": False,
                "has_gguf": False,
                "branches": [],
                "error": "huggingface_hub is not installed.",
            }
        api = HfApi()
        try:
            files = api.list_repo_files(repo_id=repo_id)
        except Exception as exc:
            return {
                "repo_id": repo_id,
                "has_transformers": False,
                "has_gguf": False,
                "branches": [],
                "error": str(exc),
            }
        has_config = any(f == "config.json" for f in files)
        has_safetensors = any(f.endswith(".safetensors") for f in files)
        has_bin = any(f.endswith(".bin") and "optimizer" not in f for f in files)
        has_gguf = any(f.endswith(".gguf") for f in files)
        branches: list[str] = []
        try:
            refs = api.list_repo_refs(repo_id=repo_id)
            branches = [b.name for b in refs.branches]
        except Exception:
            self._logger.debug("inspect_repo | failed to list branches for %s", repo_id)
        return {
            "repo_id": repo_id,
            "has_transformers": has_config and (has_safetensors or has_bin),
            "has_gguf": has_gguf,
            "branches": branches,
            "error": None,
        }

    def _detect_model_architecture(self, model_source: str) -> str:
        config_path = Path(model_source) / "config.json"
        if not config_path.exists():
            return ""
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures:
            return str(architectures[0])
        return str(config.get("model_type", ""))

    def _resolve_model_loader(self, transformers: Any, architecture: str) -> Any:
        if "ConditionalGeneration" in architecture and hasattr(
            transformers, "AutoModelForImageTextToText"
        ):
            return transformers.AutoModelForImageTextToText
        return transformers.AutoModelForCausalLM

    def _supports_non_thinking_mode(self) -> bool:
        model_id = (self.current_model_id or "").lower()
        architecture = (self.current_model_architecture or "").lower()
        return "qwen" in model_id or "qwen3_5" in architecture

    def _count_output_tokens(self, text: str) -> int:
        if not text.strip():
            return 0
        try:
            encoded = self.tokenizer(text, add_special_tokens=False)
        except Exception:
            return 0
        input_ids = encoded.get("input_ids")
        if isinstance(input_ids, list):
            return len(input_ids)
        return 0

    def describe_torch_runtime(self) -> str:
        try:
            import torch
        except ImportError:
            return "Torch runtime: torch is not installed in the current interpreter."
        return self._build_torch_runtime_summary(torch)

    def is_bitsandbytes_available(self) -> bool:
        return importlib.util.find_spec("bitsandbytes") is not None

    def _build_missing_bitsandbytes_message(self, quant: str) -> str:
        return (
            f"Quantization '{quant}' requires bitsandbytes, but bitsandbytes is not installed.\n\n"
            "If you want plain float16 or bfloat16 loading, set Quantization to 'none'.\n"
            "Use 4bit or 8bit only when bitsandbytes is installed in this .venv."
        )

    def _build_cuda_unavailable_message(self, torch: Any) -> str:
        summary = self._build_torch_runtime_summary(torch)
        return (
            "CUDA device map was requested, but CUDA is unavailable in the current PyTorch runtime.\n\n"
            f"{summary}\n"
            "Your NVIDIA driver may be installed, but this virtual environment is using a CPU-only torch build. "
            "Install a CUDA-enabled PyTorch package into this .venv to use device_map=cuda."
        )

    def _build_torch_runtime_summary(self, torch: Any) -> str:
        torch_version = getattr(torch, "__version__", "unknown")
        torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_built = bool(
            getattr(getattr(torch, "backends", None), "cuda", None)
            and torch.backends.cuda.is_built()
        )
        device_count = 0
        cuda_available = False
        try:
            cuda_available = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count())
        except Exception:
            pass
        return (
            f"Current Python: {sys.executable}\n"
            f"Current torch: {torch_version}\n"
            f"torch.version.cuda: {torch_cuda_version}\n"
            f"torch.backends.cuda.is_built(): {cuda_built}\n"
            f"torch.cuda.is_available(): {cuda_available}\n"
            f"torch.cuda.device_count(): {device_count}"
        )

    def _format_load_error(self, exc: Exception) -> str:
        message = str(exc)
        if "model type `qwen3_5`" not in message:
            return message
        try:
            import transformers

            transformers_version = transformers.__version__
        except ImportError:
            transformers_version = "unknown"
        return (
            f"{message}\n\n"
            f"Current Python: {sys.executable}\n"
            f"Current Transformers: {transformers_version}\n"
            "This usually means the app was launched outside the project .venv, or Transformers is too old in the active interpreter."
        )
