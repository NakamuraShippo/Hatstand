from __future__ import annotations

import json
import re
from dataclasses import dataclass

from hatstand.domain.entities import ChatMessage, ChatSession
from hatstand.domain.enums import OutputMode


OUTPUT_MODE_INSTRUCTIONS: dict[str, str] = {
    OutputMode.NORMAL.value: "",
    OutputMode.MARKDOWN.value: "Format the final answer in clean Markdown. Use headings and lists only when helpful.",
    OutputMode.JSON.value: "Return one valid JSON object only. Do not wrap it in Markdown fences and do not add commentary.",
    OutputMode.CODE_ONLY.value: "Return only the code that answers the request. Do not include explanations before or after the code.",
    OutputMode.BULLET_LIST.value: "Return the answer as a concise bullet list using '-' prefixes.",
}


@dataclass(slots=True)
class OutputValidationResult:
    text: str
    valid: bool | None
    error_message: str = ""


def available_output_modes() -> tuple[str, ...]:
    return tuple(item.value for item in OutputMode)


def build_effective_system_prompt(base_prompt: str, output_mode: str) -> str:
    base = base_prompt.strip()
    mode_instruction = OUTPUT_MODE_INSTRUCTIONS.get(output_mode, "")
    if not mode_instruction:
        return base
    if not base:
        return mode_instruction
    return f"{base}\n\nOutput requirement:\n{mode_instruction}"


def validate_output_text(output_mode: str, text: str) -> OutputValidationResult:
    if output_mode != OutputMode.JSON.value:
        return OutputValidationResult(text=text, valid=None)
    candidate = _extract_fenced_code_content(text.strip()) or text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return OutputValidationResult(
            text=text,
            valid=False,
            error_message=f"JSON validation failed at line {exc.lineno}, column {exc.colno}.",
        )
    pretty = json.dumps(payload, ensure_ascii=False, indent=2)
    return OutputValidationResult(text=pretty, valid=True)


def build_json_retry_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    retry_messages = list(messages)
    retry_messages.append(
        {
            "role": "user",
            "content": "The previous answer was not valid JSON. Return only one valid JSON object with no markdown fences.",
        }
    )
    return retry_messages


def extract_code_blocks(text: str) -> str:
    matches = re.findall(r"```(?:[\w.+-]+)?\n(.*?)```", text, flags=re.DOTALL)
    cleaned = [match.strip("\n") for match in matches if match.strip()]
    return "\n\n".join(cleaned)


def build_session_markdown(session: ChatSession) -> str:
    lines = [
        f"# {session.title}",
        "",
        f"- Session ID: {session.session_id}",
        f"- Updated At: {session.updated_at}",
        f"- Output Mode: {session.output_mode}",
        f"- Pinned: {'yes' if session.pinned else 'no'}",
    ]
    if session.branch_from_session_id:
        lines.append(f"- Branched From Session: {session.branch_from_session_id}")
    if session.branch_from_message_id:
        lines.append(f"- Branched From Message: {session.branch_from_message_id}")
    lines.extend(
        [
            "",
            "## System Prompt",
            "",
            session.system_prompt or "(empty)",
            "",
            "## Conversation",
            "",
        ]
    )
    for message in session.messages:
        lines.extend(_build_message_markdown(message))
    return "\n".join(lines).strip() + "\n"


def session_matches_query(session: ChatSession, query: str) -> bool:
    needle = query.strip().lower()
    if not needle:
        return True
    haystack = "\n".join(
        [
            session.title,
            session.system_prompt,
            *(message.content for message in session.messages),
        ]
    ).lower()
    return needle in haystack


def _build_message_markdown(message: ChatMessage) -> list[str]:
    label = message.role.capitalize()
    lines = [f"### {label}", "", message.content or "(empty)", ""]
    return lines


def _extract_fenced_code_content(text: str) -> str:
    match = re.search(r"```(?:json)?\n(.*?)```", text, flags=re.DOTALL)
    if match is None:
        return ""
    return match.group(1).strip()
