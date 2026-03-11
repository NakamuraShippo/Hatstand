from __future__ import annotations

import logging
from collections.abc import Callable

from hatstand.backends.base_backend import BaseBackend
from hatstand.domain.dtos import BenchmarkRow, BenchmarkSpec, CancelToken, GenerateRequest
from hatstand.domain.entities import GenerationParameters


class BenchmarkService:
    def __init__(self, backend: BaseBackend, logger: logging.Logger | None = None) -> None:
        self._backend = backend
        self._logger = logger or logging.getLogger(__name__)

    def run(
        self,
        spec: BenchmarkSpec,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> list[BenchmarkRow]:
        rows: list[BenchmarkRow] = []
        for index, (preset_name, options) in enumerate(spec.load_options, start=1):
            if progress_callback:
                progress_callback(
                    {
                        "step": index,
                        "total": len(spec.load_options),
                        "phase": "load",
                        "model_name": preset_name,
                    }
                )
            load_result = self._backend.load_model(options)
            if not load_result.success:
                rows.append(
                    BenchmarkRow(
                        model_name=preset_name,
                        model_id=options.model_id,
                        load_seconds=load_result.load_seconds,
                        first_token_seconds=0.0,
                        total_seconds=0.0,
                        output_tokens=0,
                        tokens_per_second=0.0,
                        success=False,
                        error_summary=load_result.error_message or "Load failed",
                    )
                )
                continue
            if progress_callback:
                progress_callback(
                    {
                        "step": index,
                        "total": len(spec.load_options),
                        "phase": "generate",
                        "model_name": preset_name,
                    }
                )
            generate_result = self._backend.generate(
                GenerateRequest.from_parameters(
                    messages=[{"role": "user", "content": spec.prompt_text}],
                    parameters=GenerationParameters(max_new_tokens=128),
                ),
                CancelToken(),
            )
            rows.append(
                BenchmarkRow(
                    model_name=preset_name,
                    model_id=options.model_id,
                    load_seconds=load_result.load_seconds,
                    first_token_seconds=generate_result.first_token_seconds,
                    total_seconds=generate_result.total_seconds,
                    output_tokens=generate_result.output_tokens,
                    tokens_per_second=generate_result.tokens_per_second,
                    success=generate_result.success,
                    error_summary=generate_result.error_message or "",
                )
            )
            self._logger.info(
                "benchmark_row | model=%s success=%s tokens_per_second=%.3f",
                options.model_id,
                generate_result.success,
                generate_result.tokens_per_second,
            )
        return rows
