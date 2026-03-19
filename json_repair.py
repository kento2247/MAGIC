from __future__ import annotations

import ast
import json
import re
from typing import Any


def loads(text: str, *args: Any, **kwargs: Any) -> Any:
    """
    Small compatibility shim for the external `json_repair` package.

    It first tries strict JSON parsing, then a few lightweight repairs for
    common LLM-style outputs such as fenced code blocks, trailing commas, and
    Python-literal dictionaries.
    """

    cleaned = _strip_code_fences(text).strip()
    if not cleaned:
        raise json.JSONDecodeError("Empty JSON input", cleaned, 0)

    try:
        return json.loads(cleaned, *args, **kwargs)
    except Exception:
        pass

    candidate = _extract_json_like_fragment(cleaned)
    if candidate is None:
        candidate = cleaned

    for repaired in (
        candidate,
        _remove_trailing_commas(candidate),
        _normalize_python_literals(_remove_trailing_commas(candidate)),
    ):
        try:
            return json.loads(repaired, *args, **kwargs)
        except Exception:
            continue

    try:
        return ast.literal_eval(candidate)
    except Exception as exc:
        raise json.JSONDecodeError("Unable to repair JSON input", candidate, 0) from exc


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped


def _extract_json_like_fragment(text: str) -> str | None:
    for left, right in (("{", "}"), ("[", "]")):
        start = text.find(left)
        end = text.rfind(right)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
    return None


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _normalize_python_literals(text: str) -> str:
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text
