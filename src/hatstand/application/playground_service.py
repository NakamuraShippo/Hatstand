from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from hatstand.backends.transformers_backend import TransformersBackend
from hatstand.core.benchmark_service import BenchmarkService
from hatstand.core.chat_features import (
    build_effective_system_prompt,
    build_session_markdown,
    extract_code_blocks,
    validate_output_text,
)
from hatstand.core.model_registry import find_preset
from hatstand.core.paths import build_app_paths, ensure_app_paths, resolve_runtime_paths
from hatstand.core.presets import DEFAULT_SYSTEM_PROMPT, SYSTEM_PROMPT_PRESETS
from hatstand.core.session_store import SessionStore
from hatstand.domain.dtos import (
    BenchmarkRow,
    BenchmarkSpec,
    CancelToken,
    GenerateProgress,
    GenerateRequest,
    GenerateResult,
    LoadOptions,
    LoadResult,
    WarmupResult,
)
from hatstand.domain.entities import (
    AppSettings,
    ChatMessage,
    ChatSession,
    GenerationParameters,
    ModelStatus,
    SessionModelSettings,
)
from hatstand.domain.enums import GenerationState, MessageRole, ModelLifecycleState, OutputMode
from hatstand.infra.settings_repository import SettingsRepository


class PlaygroundService:
    def __init__(
        self,
        root_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger("hatstand")
        self.bootstrap_paths = ensure_app_paths(build_app_paths(root_dir))
        self.settings_repository = SettingsRepository(self.bootstrap_paths.settings_path, self.logger)
        self.settings = self.settings_repository.load()
        self.paths = ensure_app_paths(resolve_runtime_paths(self.bootstrap_paths, self.settings.app_data_path))
        self.session_store = SessionStore(self.paths.sessions_dir, self.logger)
        self.backend = TransformersBackend(self.settings.hf_cache_dir, self.logger)
        self.benchmark_service = BenchmarkService(self.backend, self.logger)
        self.status = ModelStatus()
        self.status.gpu_available = self._detect_gpu()
        self.current_session = ChatSession(system_prompt=DEFAULT_SYSTEM_PROMPT)
        self.current_cancel_token: CancelToken | None = None
        self._apply_default_model_settings()

    def list_sessions(self, query: str = "", sort_by: str = "updated_desc") -> list[ChatSession]:
        return self.session_store.list_sessions(query=query, sort_by=sort_by)

    def request_load_model(
        self,
        options: LoadOptions,
        on_started: Callable[[], None] | None = None,
    ) -> LoadResult:
        if self.status.model_state == ModelLifecycleState.LOADING.value:
            return LoadResult(
                success=False,
                model_id=options.model_id,
                load_seconds=0.0,
                error_message="Model load is already running.",
            )
        self.status.model_state = ModelLifecycleState.LOADING.value
        self.status.error_message = ""
        if on_started:
            on_started()
        result = self.backend.load_model(options)
        self._handle_load_finished(result)
        return result

    def request_unload_model(self, on_started: Callable[[], None] | None = None) -> bool:
        self.status.model_state = ModelLifecycleState.UNLOADING.value
        self.status.error_message = ""
        if on_started:
            on_started()
        try:
            self.backend.unload_model()
            self.status.current_model_id = ""
            self.status.quant = "none"
            self.status.dtype = "bfloat16"
            self.status.device_map = "auto"
            self.status.model_state = ModelLifecycleState.UNLOADED.value
            self.status.error_message = ""
            self.logger.info("request_unload_model | success")
            return True
        except Exception as exc:
            self.status.model_state = ModelLifecycleState.ERROR.value
            self.status.error_message = str(exc)
            self.logger.exception("Unload Model")
            return False

    def request_generate(
        self,
        user_text: str,
        system_prompt: str,
        parameters: GenerationParameters,
        on_started: Callable[[], None] | None = None,
        stream_callback: Callable[[GenerateProgress], None] | None = None,
        output_mode: str | None = None,
        enable_thinking: bool | None = None,
    ) -> GenerateResult:
        if self.status.model_state != ModelLifecycleState.LOADED.value:
            return self._build_generate_error("Load a model before generating.")
        if self.status.generation_state == GenerationState.GENERATING.value:
            return self._build_generate_error("Generation is already running.")
        cleaned_input = user_text.strip()
        if not cleaned_input:
            return self._build_generate_error("User input is empty.")
        self._prepare_generation_settings(
            system_prompt=system_prompt,
            parameters=parameters,
            output_mode=output_mode,
            enable_thinking=enable_thinking,
        )
        self.current_session.messages.append(ChatMessage(role=MessageRole.USER.value, content=cleaned_input))
        if self.current_session.title == "Untitled Session":
            self.current_session.title = cleaned_input[:40]
        return self._generate_reply_for_current_session(
            parameters=parameters,
            on_started=on_started,
            stream_callback=stream_callback,
        )

    def request_regenerate_last_response(
        self,
        parameters: GenerationParameters | None = None,
        on_started: Callable[[], None] | None = None,
        stream_callback: Callable[[GenerateProgress], None] | None = None,
    ) -> GenerateResult:
        if self.status.model_state != ModelLifecycleState.LOADED.value:
            return self._build_generate_error("Load a model before generating.")
        last_assistant_index = self._last_message_index_by_role(MessageRole.ASSISTANT.value)
        if last_assistant_index < 0:
            return self._build_generate_error("There is no assistant reply to regenerate.")
        self.current_session.messages = self.current_session.messages[:last_assistant_index]
        params = parameters or deepcopy(self.current_session.generation)
        return self._generate_reply_for_current_session(
            parameters=params,
            on_started=on_started,
            stream_callback=stream_callback,
        )

    def request_resend_last_user_message(
        self,
        parameters: GenerationParameters | None = None,
        on_started: Callable[[], None] | None = None,
        stream_callback: Callable[[GenerateProgress], None] | None = None,
    ) -> GenerateResult:
        last_user_index = self._last_message_index_by_role(MessageRole.USER.value)
        if last_user_index < 0:
            return self._build_generate_error("There is no user message to resend.")
        message = self.current_session.messages[last_user_index]
        params = parameters or deepcopy(self.current_session.generation)
        return self.request_generate(
            user_text=message.content,
            system_prompt=self.current_session.system_prompt,
            parameters=params,
            on_started=on_started,
            stream_callback=stream_callback,
            output_mode=self.current_session.output_mode,
            enable_thinking=self.current_session.thinking_enabled,
        )

    def request_edit_message(
        self,
        message_id: str,
        new_text: str,
        parameters: GenerationParameters | None = None,
        on_started: Callable[[], None] | None = None,
        stream_callback: Callable[[GenerateProgress], None] | None = None,
    ) -> GenerateResult:
        message_index = self.current_session.find_message_index(message_id)
        if message_index < 0:
            return self._build_generate_error("Message not found.")
        cleaned_text = new_text.strip()
        if not cleaned_text:
            return self._build_generate_error("Edited message is empty.")
        target_message = self.current_session.messages[message_index]
        target_message.content = cleaned_text
        target_message.touch()
        self.current_session.messages = self.current_session.messages[: message_index + 1]
        self.current_session.touch()
        if target_message.role != MessageRole.USER.value:
            self.status.error_message = ""
            self.status.generation_state = GenerationState.IDLE.value
            self._autosave_current_session()
            return GenerateResult(
                success=True,
                text=target_message.content,
                first_token_seconds=0.0,
                total_seconds=0.0,
                output_tokens=0,
                tokens_per_second=0.0,
                output_mode=self.current_session.output_mode,
            )
        params = parameters or deepcopy(self.current_session.generation)
        if message_index == 0 or self.current_session.title == "Untitled Session":
            self.current_session.title = cleaned_text[:40]
        return self._generate_reply_for_current_session(
            parameters=params,
            on_started=on_started,
            stream_callback=stream_callback,
        )

    def request_branch_session_from_message(self, message_id: str) -> ChatSession:
        cloned_messages = self.current_session.clone_messages_through(message_id)
        branch = ChatSession(
            title=f"{self.current_session.title} (Branch)",
            system_prompt=self.current_session.system_prompt,
            model=deepcopy(self.current_session.model),
            generation=deepcopy(self.current_session.generation),
            messages=cloned_messages,
            pinned=False,
            output_mode=self.current_session.output_mode,
            thinking_enabled=self.current_session.thinking_enabled,
            branch_from_session_id=self.current_session.session_id,
            branch_from_message_id=message_id,
        )
        self.current_session = branch
        self._autosave_current_session()
        return self.current_session

    def request_duplicate_session(self) -> ChatSession:
        duplicate = self.request_duplicate_session_by_id(self.current_session.session_id)
        if duplicate is None:
            raise ValueError(self.status.error_message or "Failed to duplicate session.")
        return duplicate

    def request_duplicate_session_by_id(self, session_id: str) -> ChatSession | None:
        try:
            source = self._load_session_for_action(session_id)
        except Exception as exc:
            self.status.error_message = str(exc)
            self.logger.exception("Duplicate Session")
            return None
        duplicate = ChatSession(
            title=f"{source.title} (Copy)",
            system_prompt=source.system_prompt,
            model=deepcopy(source.model),
            generation=deepcopy(source.generation),
            messages=[ChatMessage.from_dict(message.to_dict()) for message in source.messages],
            pinned=source.pinned,
            output_mode=source.output_mode,
            thinking_enabled=source.thinking_enabled,
        )
        self.current_session = duplicate
        self.status.error_message = ""
        self._autosave_current_session()
        return self.current_session

    def request_delete_session(self, session_id: str) -> bool:
        is_current = session_id == self.current_session.session_id
        try:
            deleted = self.session_store.delete_session(session_id)
        except OSError as exc:
            self.status.error_message = str(exc)
            self.logger.exception("Delete Session")
            return False
        if not deleted and not is_current:
            self.status.error_message = f"Session not found: {session_id}"
            return False
        if is_current:
            remaining_sessions = self.list_sessions(sort_by="updated_desc")
            self.current_session = remaining_sessions[0] if remaining_sessions else self._build_new_session()
        self.status.error_message = ""
        return True

    def request_stop_generation(self) -> None:
        if self.current_cancel_token is None:
            return
        self.current_cancel_token.cancel()
        self.status.generation_state = GenerationState.STOPPING.value
        self.logger.info("request_stop_generation | requested")

    def request_warmup(self, on_started: Callable[[], None] | None = None) -> WarmupResult:
        if self.status.model_state != ModelLifecycleState.LOADED.value:
            return self._build_warmup_error("Load a model before running warmup.")
        if self.status.generation_state == GenerationState.GENERATING.value:
            return self._build_warmup_error("Warmup is not available while generation is running.")
        self.status.error_message = ""
        self.status.generation_state = GenerationState.GENERATING.value
        if on_started:
            on_started()
        result = self.backend.warmup()
        self._handle_warmup_finished(result)
        return result

    def request_save_session(self, title: str) -> Path:
        if title.strip():
            self.current_session.title = title.strip()
        self.current_session.touch()
        path = self.session_store.save_session(self.current_session)
        self.logger.info("request_save_session | path=%s", path)
        return path

    def request_new_session(self) -> ChatSession:
        self.current_session = self._build_new_session()
        return self.current_session

    def request_load_session(self, session_id: str) -> bool:
        try:
            self.current_session = self.session_store.load_session(session_id)
        except Exception as exc:
            self.status.error_message = str(exc)
            self.logger.exception("Load Session")
            return False
        self.status.error_message = ""
        self.logger.info("request_load_session | session_id=%s", session_id)
        return True

    def set_current_session_title(self, title: str) -> None:
        self.current_session.title = title.strip() or "Untitled Session"
        self.current_session.touch()

    def set_current_output_mode(self, output_mode: str) -> None:
        self.current_session.output_mode = output_mode if output_mode in {item.value for item in OutputMode} else OutputMode.NORMAL.value
        self.current_session.touch()

    def set_current_thinking_enabled(self, enabled: bool) -> None:
        self.current_session.thinking_enabled = enabled
        self.current_session.touch()

    def set_current_session_pinned(self, pinned: bool) -> None:
        self.set_session_pinned(self.current_session.session_id, pinned)

    def set_session_pinned(self, session_id: str, pinned: bool) -> bool:
        try:
            session = self._load_session_for_action(session_id)
        except Exception as exc:
            self.status.error_message = str(exc)
            self.logger.exception("Pin Session")
            return False
        session.pinned = pinned
        session.touch()
        self.status.error_message = ""
        self._save_session_after_action(session)
        return True

    def export_current_session_markdown(self, output_path: Path) -> Path:
        return self.export_session_markdown(self.current_session.session_id, output_path)

    def export_session_markdown(self, session_id: str, output_path: Path) -> Path:
        session = self._load_session_for_action(session_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(build_session_markdown(session), encoding="utf-8")
        session.export_meta["last_markdown_export_path"] = str(output_path)
        self._save_session_after_action(session)
        return output_path

    def export_current_session_json(self, output_path: Path) -> Path:
        return self.export_session_json(self.current_session.session_id, output_path)

    def export_session_json(self, session_id: str, output_path: Path) -> Path:
        session = self._load_session_for_action(session_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self._session_to_json_text(session),
            encoding="utf-8",
        )
        session.export_meta["last_json_export_path"] = str(output_path)
        self._save_session_after_action(session)
        return output_path

    def get_latest_assistant_text(self) -> str:
        for message in reversed(self.current_session.messages):
            if message.role == MessageRole.ASSISTANT.value and message.content.strip():
                return message.content
        return ""

    def get_latest_code_blocks(self) -> str:
        latest_text = self.get_latest_assistant_text()
        extracted = extract_code_blocks(latest_text)
        if extracted.strip():
            return extracted
        if self.current_session.output_mode == OutputMode.CODE_ONLY.value:
            return latest_text.strip()
        return ""

    def set_current_system_prompt(self, prompt: str) -> None:
        self.current_session.system_prompt = prompt
        self.current_session.touch()
        self.logger.debug("set_current_system_prompt | length=%s", len(prompt))

    def save_system_prompt_preset(self, name: str, prompt: str) -> bool:
        preset_name = name.strip()
        prompt_text = prompt.strip()
        if not preset_name:
            self.status.error_message = "Preset name is empty."
            return False
        if not prompt_text:
            self.status.error_message = "System prompt is empty."
            return False
        if preset_name in SYSTEM_PROMPT_PRESETS:
            self.status.error_message = f'"{preset_name}" is a built-in preset and cannot be overwritten.'
            return False
        action = "updated" if preset_name in self.settings.system_prompt_presets else "created"
        self.settings.system_prompt_presets[preset_name] = prompt_text
        self.settings_repository.save(self.settings)
        self.status.error_message = ""
        self.logger.info("save_system_prompt_preset | name=%s action=%s", preset_name, action)
        return True

    def create_system_prompt_preset(self, name: str, prompt: str) -> bool:
        return self.save_system_prompt_preset(name, prompt)

    def delete_system_prompt_preset(self, name: str) -> bool:
        preset_name = name.strip()
        if not preset_name:
            self.status.error_message = "Select a preset to delete."
            return False
        if preset_name in SYSTEM_PROMPT_PRESETS:
            self.status.error_message = "Built-in presets cannot be deleted."
            return False
        if preset_name not in self.settings.system_prompt_presets:
            self.status.error_message = f'"{preset_name}" is not a custom preset.'
            return False
        del self.settings.system_prompt_presets[preset_name]
        self.settings_repository.save(self.settings)
        self.status.error_message = ""
        self.logger.info("delete_system_prompt_preset | name=%s", preset_name)
        return True

    def request_run_benchmark(
        self,
        spec: BenchmarkSpec,
        on_started: Callable[[], None] | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> list[BenchmarkRow]:
        if self.status.model_state in {ModelLifecycleState.LOADING.value, ModelLifecycleState.UNLOADING.value}:
            self.status.error_message = "A model operation is already running."
            return []
        if self.status.generation_state == GenerationState.GENERATING.value:
            self.status.error_message = "Stop generation before running benchmark."
            return []
        self.status.model_state = ModelLifecycleState.LOADING.value
        self.status.error_message = ""
        if on_started:
            on_started()
        try:
            rows = self.benchmark_service.run(spec, progress_callback)
        except Exception as exc:
            self.logger.exception("Benchmark")
            self.status.model_state = ModelLifecycleState.ERROR.value
            self.status.error_message = str(exc)
            return []
        self._refresh_status_from_backend()
        return rows

    def export_benchmark_csv(self, rows: list[dict[str, object]], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "model_name",
                    "model_id",
                    "load_seconds",
                    "first_token_seconds",
                    "total_seconds",
                    "output_tokens",
                    "tokens_per_second",
                    "success",
                    "error_summary",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        self.logger.info("export_benchmark_csv | path=%s", output_path)
        return output_path

    def save_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.paths = ensure_app_paths(resolve_runtime_paths(self.bootstrap_paths, settings.app_data_path))
        self.session_store = SessionStore(self.paths.sessions_dir, self.logger)
        self.backend.update_cache_dir(settings.hf_cache_dir)
        self.settings_repository.save(settings)
        self.status.error_message = ""

    def build_load_options_from_preset(self, preset_name: str) -> LoadOptions | None:
        preset = find_preset(preset_name)
        if preset is None:
            return None
        return LoadOptions(
            model_id=preset.model_id,
            quant=preset.quant,
            dtype=preset.dtype,
            device_map=preset.device_map,
        )

    def _handle_load_finished(self, result: LoadResult) -> None:
        if result.success:
            self.status.model_state = ModelLifecycleState.LOADED.value
            self.status.generation_state = GenerationState.IDLE.value
            self.status.error_message = ""
            self.status.current_model_id = result.model_id
            if self.backend.current_options is not None:
                self.status.quant = self.backend.current_options.quant
                self.status.dtype = self.backend.current_options.dtype
                self.status.device_map = self.backend.current_options.device_map
                self.current_session.model = SessionModelSettings(
                    model_id=self.backend.current_options.model_id,
                    quant=self.backend.current_options.quant,
                    dtype=self.backend.current_options.dtype,
                    device_map=self.backend.current_options.device_map,
                )
            return
        self.status.model_state = ModelLifecycleState.ERROR.value
        self.status.error_message = result.error_message or "Model load failed."

    def _handle_generate_finished(self, result: GenerateResult, assistant_message: ChatMessage | None = None) -> None:
        if result.success and result.text:
            target_message = assistant_message or ChatMessage(role=MessageRole.ASSISTANT.value, content="")
            target_message.content = result.text
            target_message.thinking_text = result.thinking_text
            target_message.touch()
            meta = dict(target_message.generation_meta)
            meta.update(
                {
                    "status": "completed" if not result.error_message else "completed_with_notice",
                    "output_mode": result.output_mode,
                    "first_token_seconds": result.first_token_seconds,
                    "total_seconds": result.total_seconds,
                    "output_tokens": result.output_tokens,
                    "tokens_per_second": result.tokens_per_second,
                    "json_valid": result.json_valid,
                    "retry_count": result.retry_count,
                }
            )
            if result.error_message:
                meta["notice"] = result.error_message
            target_message.generation_meta = meta
            if assistant_message is None:
                self.current_session.messages.append(target_message)
            self.status.generation_state = GenerationState.COMPLETED.value
            self.status.error_message = result.error_message or ""
            self.current_session.touch()
            self._autosave_current_session()
        else:
            if assistant_message is not None and not assistant_message.content.strip():
                self.current_session.messages = [
                    message
                    for message in self.current_session.messages
                    if message.message_id != assistant_message.message_id
                ]
            self.status.generation_state = GenerationState.FAILED.value
            self.status.error_message = result.error_message or "Generation failed."
        self.current_cancel_token = None

    def _handle_warmup_finished(self, result: WarmupResult) -> None:
        self.status.generation_state = GenerationState.IDLE.value
        self.status.error_message = result.error_message or ""

    def _build_generate_error(self, message: str) -> GenerateResult:
        self.status.error_message = message
        return GenerateResult(
            success=False,
            text="",
            first_token_seconds=0.0,
            total_seconds=0.0,
            output_tokens=0,
            tokens_per_second=0.0,
            error_message=message,
            output_mode=self.current_session.output_mode,
        )

    def _build_warmup_error(self, message: str) -> WarmupResult:
        self.status.error_message = message
        return WarmupResult(success=False, total_seconds=0.0, error_message=message)

    def _prepare_generation_settings(
        self,
        system_prompt: str,
        parameters: GenerationParameters,
        output_mode: str | None,
        enable_thinking: bool | None,
    ) -> None:
        self.current_session.system_prompt = system_prompt.strip() or DEFAULT_SYSTEM_PROMPT
        self.current_session.generation = parameters
        if output_mode is not None:
            self.set_current_output_mode(output_mode)
        if enable_thinking is not None:
            self.set_current_thinking_enabled(enable_thinking)
        self.current_session.touch()

    def _generate_reply_for_current_session(
        self,
        parameters: GenerationParameters,
        on_started: Callable[[], None] | None = None,
        stream_callback: Callable[[GenerateProgress], None] | None = None,
    ) -> GenerateResult:
        if self.status.generation_state == GenerationState.GENERATING.value:
            return self._build_generate_error("Generation is already running.")
        request = self._build_generate_request(parameters)
        assistant_message = self._build_assistant_placeholder()
        self.current_session.messages.append(assistant_message)
        self.current_cancel_token = CancelToken()
        self.status.error_message = ""
        self.status.generation_state = GenerationState.GENERATING.value
        if on_started:
            on_started()
        result = self.backend.generate(
            request,
            self.current_cancel_token,
            stream_callback=lambda progress: self._handle_stream_progress(
                assistant_message,
                progress,
                external_callback=stream_callback,
            ),
        )
        result = self._post_process_generate_result(
            request=request,
            result=result,
            assistant_message=assistant_message,
            stream_callback=stream_callback,
        )
        self._handle_generate_finished(result, assistant_message=assistant_message)
        return result

    def _build_generate_request(self, parameters: GenerationParameters) -> GenerateRequest:
        return GenerateRequest.from_parameters(
            messages=self._build_request_messages(),
            parameters=parameters,
            output_mode=self.current_session.output_mode,
            enable_thinking=self.current_session.thinking_enabled,
        )

    def _build_request_messages(self) -> list[dict[str, str]]:
        payload: list[dict[str, str]] = []
        effective_prompt = build_effective_system_prompt(
            self.current_session.system_prompt,
            self.current_session.output_mode,
        )
        if effective_prompt.strip():
            payload.append({"role": MessageRole.SYSTEM.value, "content": effective_prompt.strip()})
        payload.extend(
            {
                "role": message.role,
                "content": message.content,
            }
            for message in self.current_session.messages
            if not (message.role == MessageRole.ASSISTANT.value and not message.content.strip())
        )
        return payload

    def _build_assistant_placeholder(self) -> ChatMessage:
        placeholder = ChatMessage(role=MessageRole.ASSISTANT.value, content="")
        placeholder.generation_meta = {
            "status": "pending",
            "output_mode": self.current_session.output_mode,
            "started_at": placeholder.created_at,
        }
        return placeholder

    def _handle_stream_progress(
        self,
        assistant_message: ChatMessage,
        progress: GenerateProgress,
        external_callback: Callable[[GenerateProgress], None] | None = None,
    ) -> None:
        assistant_message.content = progress.text
        assistant_message.thinking_text = progress.thinking_text
        assistant_message.generation_meta["status"] = "streaming"
        assistant_message.generation_meta["first_token_seconds"] = progress.first_token_seconds
        assistant_message.generation_meta["output_tokens"] = progress.output_tokens
        assistant_message.touch()
        self.current_session.touch()
        if external_callback:
            external_callback(progress)

    def _post_process_generate_result(
        self,
        request: GenerateRequest,
        result: GenerateResult,
        assistant_message: ChatMessage,
        stream_callback: Callable[[GenerateProgress], None] | None = None,
    ) -> GenerateResult:
        if not result.success:
            return result
        validation = validate_output_text(request.output_mode, result.text)
        if validation.valid is True:
            result.text = validation.text
            result.json_valid = True
            return result
        if validation.valid is None:
            return result
        result.json_valid = False
        if self.current_cancel_token is None or self.current_cancel_token.cancelled:
            result.error_message = validation.error_message
            return result
        assistant_message.content = ""
        assistant_message.generation_meta["status"] = "retrying_json"
        retry_messages = self._build_request_messages()
        retry_messages.append({"role": MessageRole.ASSISTANT.value, "content": result.text})
        retry_messages.append(
            {
                "role": MessageRole.USER.value,
                "content": "The previous answer was not valid JSON. Return only one valid JSON object with no markdown fences.",
            }
        )
        retry_request = GenerateRequest.from_parameters(
            messages=retry_messages,
            parameters=self.current_session.generation,
            output_mode=request.output_mode,
            enable_thinking=request.enable_thinking,
        )
        retry_result = self.backend.generate(
            retry_request,
            self.current_cancel_token,
            stream_callback=lambda progress: self._handle_stream_progress(
                assistant_message,
                progress,
                external_callback=stream_callback,
            ),
        )
        retry_result.retry_count = 1
        retry_validation = validate_output_text(request.output_mode, retry_result.text)
        if retry_result.success and retry_validation.valid is True:
            retry_result.text = retry_validation.text
            retry_result.json_valid = True
            return retry_result
        result.retry_count = 1
        result.error_message = retry_validation.error_message or validation.error_message
        return result

    def _last_message_index_by_role(self, role: str) -> int:
        for index in range(len(self.current_session.messages) - 1, -1, -1):
            if self.current_session.messages[index].role == role:
                return index
        return -1

    def _autosave_current_session(self) -> None:
        try:
            self.session_store.save_session(self.current_session)
        except Exception:
            self.logger.exception("autosave_failed | session_id=%s", self.current_session.session_id)

    def _load_session_for_action(self, session_id: str) -> ChatSession:
        if session_id == self.current_session.session_id:
            return self.current_session
        return self.session_store.load_session(session_id)

    def _save_session_after_action(self, session: ChatSession) -> None:
        if session.session_id == self.current_session.session_id:
            self.current_session = session
            self._autosave_current_session()
            return
        self.session_store.save_session(session)

    def _session_to_json_text(self, session: ChatSession) -> str:
        import json

        return json.dumps(session.to_dict(), ensure_ascii=False, indent=2)

    def _detect_gpu(self) -> bool:
        try:
            import torch
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def _apply_default_model_settings(self) -> None:
        default_options = self.build_load_options_from_preset(self.settings.default_model_preset)
        if default_options is None:
            return
        self.current_session.model = SessionModelSettings(
            model_id=default_options.model_id,
            quant=default_options.quant,
            dtype=default_options.dtype,
            device_map=default_options.device_map,
        )
        self.current_session.generation = self.settings.default_generation

    def _build_new_session(self) -> ChatSession:
        model_settings = self._effective_session_model_settings()
        return ChatSession(
            system_prompt=self.current_session.system_prompt or DEFAULT_SYSTEM_PROMPT,
            model=model_settings,
            generation=deepcopy(self.current_session.generation),
            output_mode=self.current_session.output_mode,
            thinking_enabled=self.current_session.thinking_enabled,
        )

    def _effective_session_model_settings(self) -> SessionModelSettings:
        if self.backend.current_options is not None:
            return SessionModelSettings(
                model_id=self.backend.current_options.model_id,
                quant=self.backend.current_options.quant,
                dtype=self.backend.current_options.dtype,
                device_map=self.backend.current_options.device_map,
            )
        if self.current_session.model.model_id:
            return SessionModelSettings(
                model_id=self.current_session.model.model_id,
                quant=self.current_session.model.quant,
                dtype=self.current_session.model.dtype,
                device_map=self.current_session.model.device_map,
            )
        default_options = self.build_load_options_from_preset(self.settings.default_model_preset)
        if default_options is not None:
            return SessionModelSettings(
                model_id=default_options.model_id,
                quant=default_options.quant,
                dtype=default_options.dtype,
                device_map=default_options.device_map,
            )
        return SessionModelSettings()

    def _refresh_status_from_backend(self) -> None:
        self.status.generation_state = GenerationState.IDLE.value
        if self.backend.current_options is None or not self.backend.is_loaded():
            self.status.model_state = ModelLifecycleState.UNLOADED.value
            self.status.current_model_id = ""
            self.status.quant = "none"
            self.status.dtype = "bfloat16"
            self.status.device_map = "auto"
            return
        self.status.model_state = ModelLifecycleState.LOADED.value
        self.status.current_model_id = self.backend.current_options.model_id
        self.status.quant = self.backend.current_options.quant
        self.status.dtype = self.backend.current_options.dtype
        self.status.device_map = self.backend.current_options.device_map
        self.current_session.model = SessionModelSettings(
            model_id=self.backend.current_options.model_id,
            quant=self.backend.current_options.quant,
            dtype=self.backend.current_options.dtype,
            device_map=self.backend.current_options.device_map,
        )
