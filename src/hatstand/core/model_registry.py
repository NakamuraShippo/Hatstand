from __future__ import annotations

from hatstand.domain.entities import ModelPreset
from hatstand.domain.enums import DeviceMap, DType, Quantization


DEFAULT_MODEL_IDS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
]


MODEL_PRESETS: list[ModelPreset] = [
    ModelPreset(
        name="Qwen3.5 2B BF16",
        model_id="Qwen/Qwen3.5-2B",
        quant=Quantization.NONE.value,
        dtype=DType.BFLOAT16.value,
        device_map=DeviceMap.AUTO.value,
    ),
    ModelPreset(
        name="Qwen3.5 4B BF16",
        model_id="Qwen/Qwen3.5-4B",
        quant=Quantization.NONE.value,
        dtype=DType.BFLOAT16.value,
        device_map=DeviceMap.AUTO.value,
    ),
    ModelPreset(
        name="Qwen3.5 9B BF16",
        model_id="Qwen/Qwen3.5-9B",
        quant=Quantization.NONE.value,
        dtype=DType.BFLOAT16.value,
        device_map=DeviceMap.AUTO.value,
    ),
    ModelPreset(
        name="Qwen3.5 27B 8bit",
        model_id="Qwen/Qwen3.5-27B",
        quant=Quantization.INT8.value,
        dtype=DType.BFLOAT16.value,
        device_map=DeviceMap.AUTO.value,
    ),
    ModelPreset(
        name="Qwen3.5 27B 4bit",
        model_id="Qwen/Qwen3.5-27B",
        quant=Quantization.INT4.value,
        dtype=DType.BFLOAT16.value,
        device_map=DeviceMap.AUTO.value,
    ),
    ModelPreset(
        name="Qwen3.5 35B-A3B 4bit",
        model_id="Qwen/Qwen3.5-35B-A3B",
        quant=Quantization.INT4.value,
        dtype=DType.BFLOAT16.value,
        device_map=DeviceMap.AUTO.value,
    ),
]


def find_preset(name: str) -> ModelPreset | None:
    return next((preset for preset in MODEL_PRESETS if preset.name == name), None)


def find_preset_by_options(
    model_id: str,
    quant: str,
    dtype: str,
    device_map: str,
) -> ModelPreset | None:
    return next(
        (
            preset
            for preset in MODEL_PRESETS
            if preset.model_id == model_id
            and preset.quant == quant
            and preset.dtype == dtype
            and preset.device_map == device_map
        ),
        None,
    )
