from __future__ import annotations

from hatstand.domain.entities import GenerationParameters


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

SYSTEM_PROMPT_PRESETS: dict[str, str] = {
    "Default Assistant": DEFAULT_SYSTEM_PROMPT,
    "Concise Assistant": "You are a helpful assistant. Respond clearly and briefly.",
    "Coding Assistant": "You are a careful coding assistant. Explain tradeoffs briefly and provide practical code-first help.",
    "Japanese Assistant": "You are a helpful assistant. Respond in natural Japanese unless the user asks for another language.",
    "Technical Reviewer": "You are a precise technical reviewer. Point out risks, likely root causes, and concrete next steps.",
}

PROMPT_PRESETS: dict[str, str] = {
    "Greeting": "Explain in two short sentences what Qwen3.5 is good at.",
    "Reasoning": "List three careful steps to debug a Python import error.",
    "Summary": "Summarize the benefits of quantized local inference in one paragraph.",
}

DEFAULT_GENERATION_PARAMETERS = GenerationParameters()
