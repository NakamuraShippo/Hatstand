from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from hatstand.domain.enums import (
    BackendKind,
    DeviceMap,
    DType,
    GenerationState,
    MessageRole,
    ModelLifecycleState,
    OutputMode,
    Quantization,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(slots=True)
class GenerationParameters:
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256
    repetition_penalty: float = 1.05
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenerationParameters":
        source = data or {}
        return cls(
            temperature=float(source.get("temperature", 0.7)),
            top_p=float(source.get("top_p", 0.95)),
            max_new_tokens=int(source.get("max_new_tokens", 256)),
            repetition_penalty=float(source.get("repetition_penalty", 1.05)),
            seed=int(source["seed"]) if source.get("seed") is not None else None,
        )


@dataclass(slots=True)
class SessionModelSettings:
    model_id: str = ""
    quant: str = Quantization.NONE.value
    dtype: str = DType.BFLOAT16.value
    device_map: str = DeviceMap.AUTO.value

    def to_dict(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "quant": self.quant,
            "dtype": self.dtype,
            "device_map": self.device_map,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SessionModelSettings":
        source = data or {}
        return cls(
            model_id=str(source.get("model_id", "")),
            quant=str(source.get("quant", Quantization.NONE.value)),
            dtype=str(source.get("dtype", DType.BFLOAT16.value)),
            device_map=str(source.get("device_map", DeviceMap.AUTO.value)),
        )


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str
    message_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    generation_meta: dict[str, Any] = field(default_factory=dict)
    thinking_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "message_id": self.message_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.generation_meta:
            payload["generation_meta"] = dict(self.generation_meta)
        if self.thinking_text:
            payload["thinking_text"] = self.thinking_text
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatMessage":
        return cls(
            role=str(data["role"]),
            content=str(data["content"]),
            message_id=str(data.get("message_id") or uuid4()),
            created_at=str(data.get("created_at") or _utc_now_iso()),
            updated_at=str(data.get("updated_at") or data.get("created_at") or _utc_now_iso()),
            generation_meta=dict(data.get("generation_meta") or {}),
            thinking_text=str(data.get("thinking_text", "")),
        )

    def touch(self) -> None:
        self.updated_at = _utc_now_iso()


@dataclass(slots=True)
class ChatSession:
    session_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    title: str = "Untitled Session"
    system_prompt: str = "You are a helpful assistant."
    model: SessionModelSettings = field(default_factory=SessionModelSettings)
    generation: GenerationParameters = field(default_factory=GenerationParameters)
    messages: list[ChatMessage] = field(default_factory=list)
    pinned: bool = False
    output_mode: str = OutputMode.NORMAL.value
    thinking_enabled: bool = False
    export_meta: dict[str, Any] = field(default_factory=dict)
    branch_from_session_id: str = ""
    branch_from_message_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "title": self.title,
            "system_prompt": self.system_prompt,
            "model": self.model.to_dict(),
            "generation": self.generation.to_dict(),
            "messages": [message.to_dict() for message in self.messages],
            "pinned": self.pinned,
            "output_mode": self.output_mode,
            "thinking_enabled": self.thinking_enabled,
        }
        if self.export_meta:
            payload["export_meta"] = dict(self.export_meta)
        if self.branch_from_session_id:
            payload["branch_from_session_id"] = self.branch_from_session_id
        if self.branch_from_message_id:
            payload["branch_from_message_id"] = self.branch_from_message_id
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatSession":
        return cls(
            session_id=str(data.get("session_id") or uuid4()),
            created_at=str(data.get("created_at") or _utc_now_iso()),
            updated_at=str(data.get("updated_at") or data.get("created_at") or _utc_now_iso()),
            title=str(data.get("title", "Untitled Session")),
            system_prompt=str(data.get("system_prompt", "You are a helpful assistant.")),
            model=SessionModelSettings.from_dict(data.get("model")),
            generation=GenerationParameters.from_dict(data.get("generation")),
            messages=[ChatMessage.from_dict(item) for item in data.get("messages", [])],
            pinned=bool(data.get("pinned", False)),
            output_mode=str(data.get("output_mode", OutputMode.NORMAL.value)),
            thinking_enabled=bool(data.get("thinking_enabled", False)),
            export_meta=dict(data.get("export_meta") or {}),
            branch_from_session_id=str(data.get("branch_from_session_id", "")),
            branch_from_message_id=str(data.get("branch_from_message_id", "")),
        )

    def build_messages_payload(self) -> list[dict[str, str]]:
        payload = []
        if self.system_prompt.strip():
            payload.append(
                {
                    "role": MessageRole.SYSTEM.value,
                    "content": self.system_prompt.strip(),
                }
            )
        payload.extend(
            {
                "role": message.role,
                "content": message.content,
            }
            for message in self.messages
        )
        return payload

    def touch(self) -> None:
        self.updated_at = _utc_now_iso()

    def find_message_index(self, message_id: str) -> int:
        for index, message in enumerate(self.messages):
            if message.message_id == message_id:
                return index
        return -1

    def clone_messages_through(self, message_id: str) -> list[ChatMessage]:
        index = self.find_message_index(message_id)
        if index < 0:
            raise ValueError(f"Message not found: {message_id}")
        return [
            ChatMessage.from_dict(message.to_dict())
            for message in self.messages[: index + 1]
        ]


@dataclass(slots=True)
class ModelPreset:
    name: str
    model_id: str
    quant: str
    dtype: str
    device_map: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "quant": self.quant,
            "dtype": self.dtype,
            "device_map": self.device_map,
        }


@dataclass(slots=True)
class AppSettings:
    hf_cache_dir: str = ""
    app_data_path: str = ""
    theme: str = "dark"
    log_level: str = "INFO"
    default_model_preset: str = "Qwen3.5 2B BF16"
    default_generation: GenerationParameters = field(default_factory=GenerationParameters)
    system_prompt_presets: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hf_cache_dir": self.hf_cache_dir,
            "app_data_path": self.app_data_path,
            "theme": self.theme,
            "log_level": self.log_level,
            "default_model_preset": self.default_model_preset,
            "default_generation": self.default_generation.to_dict(),
            "system_prompt_presets": dict(self.system_prompt_presets),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AppSettings":
        source = data or {}
        return cls(
            hf_cache_dir=str(source.get("hf_cache_dir", "")),
            app_data_path=str(source.get("app_data_path", "")),
            theme=str(source.get("theme", "dark")),
            log_level=str(source.get("log_level", "INFO")).upper(),
            default_model_preset=str(source.get("default_model_preset", "Qwen3.5 2B BF16")),
            default_generation=GenerationParameters.from_dict(source.get("default_generation")),
            system_prompt_presets={
                str(name): str(prompt)
                for name, prompt in dict(source.get("system_prompt_presets", {})).items()
                if str(name).strip()
            },
        )


@dataclass(slots=True)
class AppPaths:
    root_dir: Path
    data_dir: Path
    sessions_dir: Path
    log_dir: Path
    settings_path: Path
    log_path: Path


@dataclass(slots=True)
class ModelStatus:
    backend: str = BackendKind.TRANSFORMERS.value
    model_state: str = ModelLifecycleState.UNLOADED.value
    generation_state: str = GenerationState.IDLE.value
    current_model_id: str = ""
    quant: str = Quantization.NONE.value
    dtype: str = DType.BFLOAT16.value
    device_map: str = DeviceMap.AUTO.value
    gpu_available: bool = False
    error_message: str = ""
