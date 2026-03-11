from __future__ import annotations

from enum import Enum


class Quantization(str, Enum):
    NONE = "none"
    INT8 = "8bit"
    INT4 = "4bit"


class DType(str, Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class DeviceMap(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class BackendKind(str, Enum):
    TRANSFORMERS = "transformers"


class ModelLifecycleState(str, Enum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    LOADED = "LOADED"
    UNLOADING = "UNLOADING"
    ERROR = "ERROR"


class GenerationState(str, Enum):
    IDLE = "IDLE"
    GENERATING = "GENERATING"
    STOPPING = "STOPPING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class OutputMode(str, Enum):
    NORMAL = "normal"
    MARKDOWN = "markdown"
    JSON = "json"
    CODE_ONLY = "code_only"
    BULLET_LIST = "bullet_list"
