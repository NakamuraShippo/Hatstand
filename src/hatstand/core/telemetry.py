from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class MemorySnapshot:
    ram_used_gb: float
    ram_total_gb: float
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None


def collect_memory_snapshot() -> MemorySnapshot:
    import psutil

    ram = psutil.virtual_memory()
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None

    try:
        import torch

        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            vram_total_gb = total_bytes / (1024**3)
            vram_used_gb = (total_bytes - free_bytes) / (1024**3)
    except (ImportError, RuntimeError):
        pass

    return MemorySnapshot(
        ram_used_gb=ram.used / (1024**3),
        ram_total_gb=ram.total / (1024**3),
        vram_used_gb=vram_used_gb,
        vram_total_gb=vram_total_gb,
    )
