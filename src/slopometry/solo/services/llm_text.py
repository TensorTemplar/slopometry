"""Shared helpers for cleaning LLM response text before JSON parsing."""

import json


def strip_llm_wrappers(text: str) -> str:
    """Remove ``<antThinking>`` blocks, markdown code fences, and leading whitespace.

    LLMs frequently wrap JSON responses in ```` ```json ```` fences or emit
    reasoning inside ``<antThinking>...</antThinking>`` tags before the
    actual payload. This normalizer strips both so the caller can feed the
    result directly to ``json.loads``.

    Raises:
        TypeError: If ``text`` is not a string (e.g. a raw MagicMock from
            an unconfigured mock).
    """
    if not isinstance(text, str):
        raise TypeError(f"strip_llm_wrappers expected str, got {type(text).__name__}")

    result = text.strip()

    if result.startswith("<antThinking>"):
        end_marker = "</antThinking>"
        end_idx = result.find(end_marker)
        if end_idx != -1:
            result = result[end_idx + len(end_marker) :]
            while result.startswith("\n"):
                result = result[1:]

    if result.startswith("```json"):
        result = result[7:]
    elif result.startswith("```"):
        result = result[3:]
    if result.endswith("```"):
        result = result[:-3]

    return result.strip()


def parse_llm_json(text: str) -> object:
    """Strip wrappers and parse the LLM response as JSON.

    Raises:
        json.JSONDecodeError: If the cleaned text is not valid JSON.
    """
    return json.loads(strip_llm_wrappers(text))
