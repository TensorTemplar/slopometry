"""Pre-flight health checks for memory extraction CLI endpoints.

Both the chat LLM (used for memory extraction) and the embedding endpoint
(used for freshness + uniqueness scoring) must be reachable before any
session is processed. If either is down, the whole batch aborts with an
explicit error rather than silently processing 12 sessions that all fail.
"""

import logging

import click

logger = logging.getLogger(__name__)


def _check_endpoint(
    label: str,
    endpoint: str,
    api_key: str,
) -> str | None:
    """Return None if the endpoint is reachable, or an error string describing why not.

    Uses ``GET /v1/models`` (cheap list call, no token cost) to validate
    reachability + auth + model availability in one shot.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return f"{label}: openai package not installed"

    try:
        client = OpenAI(base_url=endpoint, api_key=api_key)
        client.models.list()
        return None
    except Exception as e:
        return f"{label} ({endpoint}): {type(e).__name__}: {e}"


def preflight_endpoints(
    chat_endpoint: str,
    embedding_endpoint: str,
    chat_api_key: str,
    embedding_api_key: str,
) -> None:
    """Validate that both endpoints are reachable before processing any session.

    Aborts with ``click.ClickException`` listing every failed endpoint so the
    user can fix them all in one pass. Individual per-session failures are
    still surfaced in the per-session error block — this pre-flight only
    catches infrastructure-level unavailability.
    """
    errors: list[str] = []
    chat_err = _check_endpoint("chat LLM", chat_endpoint, chat_api_key)
    if chat_err:
        errors.append(chat_err)
    embed_err = _check_endpoint("embedding", embedding_endpoint, embedding_api_key)
    if embed_err:
        errors.append(embed_err)
    if errors:
        message = "Pre-flight endpoint check failed:\n  - " + "\n  - ".join(errors)
        message += (
            "\nNo sessions were processed. Fix the endpoint(s) and re-run, "
            "or pass --dry-run to skip the check (parsing + discovery only)."
        )
        raise click.ClickException(message)
