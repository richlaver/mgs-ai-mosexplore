from __future__ import annotations

import re
from typing import Iterable


def strip_to_json_payload(text: str, markers: Iterable[str]) -> str:
    if not text:
        return text
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) > 1 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()
        else:
            cleaned = "\n".join(lines[1:]).strip()

    start_idx = -1
    for marker in markers:
        if not marker:
            continue
        pattern = r"[\{\[]\s*[\s\S]*?" + re.escape(marker)
        match = re.search(pattern, cleaned)
        if match and (start_idx == -1 or match.start() < start_idx):
            start_idx = match.start()

    if start_idx == -1:
        bracket_match = re.search(r"[\{\[]", cleaned)
        if bracket_match:
            start_idx = bracket_match.start()

    if start_idx != -1:
        cleaned = cleaned[start_idx:]

    if cleaned.startswith("{"):
        end_idx = cleaned.rfind("}")
        if end_idx != -1:
            cleaned = cleaned[:end_idx + 1]
    elif cleaned.startswith("["):
        end_idx = cleaned.rfind("]")
        if end_idx != -1:
            cleaned = cleaned[:end_idx + 1]
    return cleaned
