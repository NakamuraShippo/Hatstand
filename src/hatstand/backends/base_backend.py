from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from hatstand.domain.dtos import (
    CancelToken,
    GenerateProgress,
    GenerateRequest,
    GenerateResult,
    LoadOptions,
    LoadResult,
    WarmupResult,
)


class BaseBackend(Protocol):
    def load_model(self, options: LoadOptions) -> LoadResult: ...

    def unload_model(self) -> None: ...

    def is_loaded(self) -> bool: ...

    def generate(
        self,
        request: GenerateRequest,
        cancel_token: CancelToken,
        stream_callback: Callable[[GenerateProgress], None] | None = None,
    ) -> GenerateResult: ...

    def warmup(self) -> WarmupResult: ...
