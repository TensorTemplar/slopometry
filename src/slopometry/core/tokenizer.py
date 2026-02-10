"""Token counting utilities using tiktoken."""

import logging
from pathlib import Path
from typing import Any

from slopometry.core.models import TokenCountError

logger = logging.getLogger(__name__)

_encoder: Any = None


def get_encoder() -> Any:
    """Get tiktoken encoder, caching for reuse.

    Returns:
        tiktoken Encoder for token counting
    """
    global _encoder
    if _encoder is not None:
        return _encoder

    import tiktoken

    try:
        _encoder = tiktoken.get_encoding("o200k_base")
    except Exception as e:
        logger.debug(f"Falling back to cl100k_base encoding: {e}")
        _encoder = tiktoken.get_encoding("cl100k_base")

    return _encoder


def count_tokens(content: str) -> int:
    """Count tokens in text content.

    Args:
        content: Text content to tokenize.

    Returns:
        Number of tokens.
    """
    encoder = get_encoder()
    return len(encoder.encode(content, disallowed_special=()))


def count_file_tokens(file_path: Path) -> int | TokenCountError:
    """Count tokens in a file.

    Args:
        file_path: Path to the file to tokenize.

    Returns:
        Number of tokens, or TokenCountError if file cannot be read.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        return count_tokens(content)
    except Exception as e:
        logger.warning("Failed to read file for token counting %s: %s", file_path, e)
        return TokenCountError(message=str(e), path=str(file_path))
