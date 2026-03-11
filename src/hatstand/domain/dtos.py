from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Any

from hatstand.domain.entities import GenerationParameters
from hatstand.domain.enums import OutputMode


@dataclass(slots=True)
class LoadOptions:
    model_id: str
    quant: str
    dtype: str
    device_map: str
    revision: str = ""


@dataclass(slots=True)
class LoadResult:
    success: bool
    model_id: str
    load_seconds: float
    error_message: str | None = None


@dataclass(slots=True)
class GenerateRequest:
    messages: list[dict[str, str]]
    temperature: float
    top_p: float
    max_new_tokens: int
    repetition_penalty: float
    seed: int | None = None
    output_mode: str = OutputMode.NORMAL.value
    enable_thinking: bool = False

    @classmethod
    def from_parameters(
        cls,
        messages: list[dict[str, str]],
        parameters: GenerationParameters,
        output_mode: str = OutputMode.NORMAL.value,
        enable_thinking: bool = False,
    ) -> "GenerateRequest":
        return cls(
            messages=messages,
            temperature=parameters.temperature,
            top_p=parameters.top_p,
            max_new_tokens=parameters.max_new_tokens,
            repetition_penalty=parameters.repetition_penalty,
            seed=parameters.seed,
            output_mode=output_mode,
            enable_thinking=enable_thinking,
        )


@dataclass(slots=True)
class GenerateProgress:
    text: str
    delta: str
    first_token_seconds: float = 0.0
    output_tokens: int = 0
    thinking_text: str = ""


@dataclass(slots=True)
class GenerateResult:
    success: bool
    text: str
    first_token_seconds: float
    total_seconds: float
    output_tokens: int
    tokens_per_second: float
    error_message: str | None = None
    output_mode: str = OutputMode.NORMAL.value
    thinking_text: str = ""
    json_valid: bool | None = None
    retry_count: int = 0


@dataclass(slots=True)
class WarmupResult:
    success: bool
    total_seconds: float
    error_message: str | None = None


@dataclass(slots=True)
class BenchmarkRow:
    model_name: str
    model_id: str
    load_seconds: float
    first_token_seconds: float
    total_seconds: float
    output_tokens: int
    tokens_per_second: float
    success: bool
    error_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_id": self.model_id,
            "load_seconds": self.load_seconds,
            "first_token_seconds": self.first_token_seconds,
            "total_seconds": self.total_seconds,
            "output_tokens": self.output_tokens,
            "tokens_per_second": self.tokens_per_second,
            "success": self.success,
            "error_summary": self.error_summary,
        }


@dataclass(slots=True)
class BenchmarkSpec:
    prompt_name: str
    prompt_text: str
    load_options: list[tuple[str, LoadOptions]]


@dataclass(slots=True)
class CancelToken:
    _event: Event = field(default_factory=Event, init=False, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()
