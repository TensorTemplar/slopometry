"""Freshness validation and staleness auditing for memory candidates.

After LLM extraction, each new candidate is paired with semantically similar
existing memories in the same project. Pairing is gated by a per-project
similarity threshold derived from the existing memories' own pairwise
similarity distribution (mean + quantiles), so the threshold is data-driven
rather than hand-tuned.

Each above-threshold pair is then sent to an LLM judge that decides how to
reconcile the two. The judge has four actions and full authority over the
decision — there are no hardcoded length heuristics or contradiction rules,
only the statistical gate to control LLM call volume.

Actions:
- keep_both: the two memories are different enough that both belong
- merge: synthesize a single updated version that supersedes both
- supersede: the new candidate wins, mark the old as outdated
- dedupe: they say the same thing; skip the new and confirm the old

A separate **staleness audit** runs after extraction + reconciliation. It
sends the full transcript alongside the existing (active) memories and asks
the LLM which memories are now stale — describing fixed bugs, completed
work, or outdated state. Stale memories are retired via ``retired_reason``
without a direct replacement.
"""

import json
import logging
import statistics
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from slopometry.core.models.memory import (
    FreshnessAction,
    FreshnessVerdict,
    MemoryCandidate,
    MemoryEntry,
    StalenessVerdict,
)
from slopometry.solo.services.llm_text import parse_llm_json

logger = logging.getLogger(__name__)

DEFAULT_FLOOR_THRESHOLD = 0.45
DEFAULT_CEILING_THRESHOLD = 0.95
DEFAULT_RECONCILIATION_MAX_TOKENS = 200
DEFAULT_STALENESS_AUDIT_MAX_TOKENS = 1000
DEFAULT_TRANSCRIPT_TRUNCATION_CHARS = 15000

RECONCILIATION_PROMPT = """You are reconciling two memory candidates about the same subject.

Two memories reconcile in exactly one of four ways:

1. **keep_both** — they cover genuinely different aspects of the subject.
   Example: "Project uses rust-code-analysis" + "Project supports Python 3.13".
   Different tools/topics — both stay.

2. **merge** — they cover the same aspect but the new one updates, supersedes,
   or extends the old. Synthesize a single merged version that combines both
   pieces of information. Example: "Project uses radon" + "Switched to
   rust-code-analysis in 2026 because radon was abandoned". Merged:
   "Project uses rust-code-analysis (switched from radon in 2026)".

3. **supersede** — the new candidate is clearly the current truth and the
   old is outdated. The new wins; the old should be flagged as outdated.
   Example: old "Python 3.10" → new "Python 3.13".

4. **dedupe** — they say the same thing in different words. Skip the new;
   the existing memory already covers it. Example: old "user prefers pyright"
   + new "user uses pyright type checker".

NEW:
{new_content}

EXISTING:
{existing_content}

Reply with JSON only:
{{"action": "keep_both" | "merge" | "supersede" | "dedupe", "reason": "<one sentence>", "merged_content": "<only when action=merge; omit otherwise>"}}"""


@dataclass(frozen=True)
class ProjectSimilarityDistribution:
    """Per-project pairwise similarity statistics for existing memories."""

    n_pairs: int
    mean: float
    p50: float
    p75: float
    p90: float
    p95: float
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD
    ceiling_threshold: float = DEFAULT_CEILING_THRESHOLD

    @property
    def derived_threshold(self) -> float:
        """Data-driven dedupe threshold from the project's own distribution.

        Uses p75 of pairwise similarity as the candidate-relevance threshold.
        Falls back to ``floor_threshold`` when the project has too few
        memories to estimate a distribution. Capped at ``ceiling_threshold``
        so that even in projects with very similar memories, only genuinely
        redundant pairs are sent to the LLM.
        """
        if self.n_pairs == 0:
            return self.floor_threshold
        return max(min(self.p75, self.ceiling_threshold), self.floor_threshold)


@dataclass(frozen=True)
class FreshnessDecision:
    """The reconciliation outcome for one (new candidate, existing memory) pair."""

    new_candidate: MemoryCandidate
    existing_memory: MemoryEntry
    similarity: float
    action: FreshnessAction
    reason: str
    merged_content: str | None = None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _project_pairwise_similarities(existing: list[MemoryEntry]) -> list[float]:
    """Compute pairwise cosine similarities among existing memories' embeddings."""
    sims: list[float] = []
    for i, m1 in enumerate(existing):
        if not m1.embedding:
            continue
        for m2 in existing[i + 1 :]:
            if not m2.embedding:
                continue
            sims.append(_cosine_similarity(m1.embedding, m2.embedding))
    return sims


def compute_project_distribution(
    existing: list[MemoryEntry],
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD,
    ceiling_threshold: float = DEFAULT_CEILING_THRESHOLD,
) -> ProjectSimilarityDistribution:
    """Compute similarity distribution statistics for the project's memory bank.

    Used to derive a data-informed threshold for which new candidate / existing
    memory pairs are worth sending to the LLM judge.
    """
    sims = _project_pairwise_similarities(existing)
    if not sims:
        return ProjectSimilarityDistribution(
            0, 0.0, 0.0, 0.0, 0.0, 0.0,
            floor_threshold=floor_threshold,
            ceiling_threshold=ceiling_threshold,
        )
    sims_sorted = sorted(sims)
    n = len(sims_sorted)

    def quantile(q: float) -> float:
        idx = max(0, min(n - 1, int(n * q)))
        return sims_sorted[idx]

    return ProjectSimilarityDistribution(
        n_pairs=n,
        mean=statistics.fmean(sims_sorted),
        p50=quantile(0.50),
        p75=quantile(0.75),
        p90=quantile(0.90),
        p95=quantile(0.95),
        floor_threshold=floor_threshold,
        ceiling_threshold=ceiling_threshold,
    )


def _find_above_threshold(
    candidate: MemoryCandidate,
    existing: list[MemoryEntry],
    threshold: float,
) -> list[tuple[MemoryEntry, float]]:
    """Return existing memories whose embedding similarity to candidate >= threshold."""
    if not candidate.embedding:
        return []
    matches: list[tuple[MemoryEntry, float]] = []
    for memory in existing:
        if not memory.embedding:
            continue
        sim = _cosine_similarity(candidate.embedding, memory.embedding)
        if sim >= threshold:
            matches.append((memory, sim))
    matches.sort(key=lambda pair: pair[1], reverse=True)
    return matches


def _call_llm_json(
    llm_endpoint: str,
    llm_model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> Any:
    """Call an OpenAI-compatible LLM and return the parsed JSON response.

    Returns the raw parsed JSON (dict or list) on success, or raises
    ``json.JSONDecodeError`` / ``ValueError`` / ``TypeError`` on parse failure.
    Network errors propagate to the caller.
    """
    client = OpenAI(base_url=llm_endpoint, api_key=api_key)
    response = client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return parse_llm_json(content)


def _judge_reconciliation(
    candidate: MemoryCandidate,
    existing: MemoryEntry,
    llm_endpoint: str,
    llm_model: str,
    api_key: str,
    similarity: float,
    max_tokens: int = DEFAULT_RECONCILIATION_MAX_TOKENS,
) -> FreshnessDecision:
    """Ask the LLM how to reconcile the pair. Always returns a decision."""
    prompt = RECONCILIATION_PROMPT.format(
        new_content=candidate.content,
        existing_content=existing.content,
    )
    try:
        data = _call_llm_json(
            llm_endpoint,
            llm_model,
            api_key,
            system_prompt="You reconcile memory pairs. Always reply with valid JSON containing action, reason, and (only when merging) merged_content.",
            user_prompt=prompt,
            max_tokens=max_tokens,
        )
        verdict = FreshnessVerdict.model_validate(data)
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.debug("Could not parse reconciliation response for candidate vs %s", existing.id)
        return FreshnessDecision(
            new_candidate=candidate,
            existing_memory=existing,
            similarity=similarity,
            action=FreshnessAction.KEEP_BOTH,
            reason="Could not parse LLM response",
        )

    merged = verdict.merged_content if verdict.action == FreshnessAction.MERGE else None
    return FreshnessDecision(
        new_candidate=candidate,
        existing_memory=existing,
        similarity=similarity,
        action=verdict.action,
        reason=verdict.reason,
        merged_content=merged,
    )


def validate_freshness(
    candidates: list[MemoryCandidate],
    existing: list[MemoryEntry],
    llm_endpoint: str,
    llm_model: str,
    api_key: str,
    floor_threshold: float = DEFAULT_FLOOR_THRESHOLD,
    ceiling_threshold: float = DEFAULT_CEILING_THRESHOLD,
) -> tuple[list[FreshnessDecision], ProjectSimilarityDistribution]:
    """Reconcile newly-extracted candidates against existing project memories.

    For each project, computes the existing memory bank's pairwise similarity
    distribution and derives a threshold from it (p75 of pairwise similarity,
    clamped between ``floor_threshold`` and ``ceiling_threshold``). Each new
    candidate is paired with existing memories above this threshold and sent
    to the LLM for a reconciliation verdict (keep_both / merge / supersede /
    dedupe).
    """
    distribution = compute_project_distribution(
        existing,
        floor_threshold=floor_threshold,
        ceiling_threshold=ceiling_threshold,
    )
    threshold = distribution.derived_threshold

    decisions: list[FreshnessDecision] = []
    for candidate in candidates:
        similar = _find_above_threshold(candidate, existing, threshold)
        for memory, similarity in similar:
            try:
                decision = _judge_reconciliation(
                    candidate, memory, llm_endpoint, llm_model, api_key, similarity
                )
            except Exception as e:
                logger.debug("Reconciliation judge failed for candidate vs %s: %s", memory.id, e)
                continue
            decisions.append(decision)
    return decisions, distribution


STALENESS_AUDIT_PROMPT = """You are auditing existing memories for staleness after analyzing a new session transcript.

A memory is STALE and should be retired if:
- It describes a bug that was fixed in this session
- It describes work that was completed in this session
- It references a state that was changed in this session
- It describes a temporary issue that was resolved

A memory is NOT stale if:
- It describes a stable preference, design decision, or user behavior pattern
- It references external resources, infrastructure, or tool locations
- The session doesn't touch the area the memory describes
- It's a general project description that remains accurate

EXISTING MEMORIES:
{memories_block}

SESSION TRANSCRIPT:
{transcript}

Return JSON only — a list of memories to retire, referencing each by its [N] number:
[{{"ref": 1, "reason": "<one sentence why this is now stale>"}}]

If no memories are stale, return an empty array: []"""


def audit_staleness(
    existing: list[MemoryEntry],
    transcript: str,
    llm_endpoint: str,
    llm_model: str,
    api_key: str,
    max_tokens: int = DEFAULT_STALENESS_AUDIT_MAX_TOKENS,
    transcript_truncation_chars: int = DEFAULT_TRANSCRIPT_TRUNCATION_CHARS,
) -> list[tuple[MemoryEntry, str]]:
    """Ask the LLM which existing memories are now stale given the session transcript.

    Returns:
        List of (memory_entry, reason) pairs for memories to retire.
    """
    if not existing or not transcript.strip():
        return []

    memories_block = "\n".join(
        f"[{i + 1}] ({m.memory_type.value}) {m.content}" for i, m in enumerate(existing)
    )
    prompt = STALENESS_AUDIT_PROMPT.format(
        memories_block=memories_block,
        transcript=transcript[:transcript_truncation_chars],
    )

    try:
        data = _call_llm_json(
            llm_endpoint,
            llm_model,
            api_key,
            system_prompt="You audit memory staleness. Always reply with valid JSON only.",
            user_prompt=prompt,
            max_tokens=max_tokens,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.debug("Could not parse staleness audit response")
        return []

    if not isinstance(data, list):
        logger.debug("Staleness audit expected JSON array, got %s", type(data).__name__)
        return []

    results: list[tuple[MemoryEntry, str]] = []
    for item in data:
        try:
            verdict = StalenessVerdict.model_validate(item)
        except (ValueError, TypeError) as e:
            logger.debug("Skipping invalid staleness verdict: %s", e)
            continue
        idx = verdict.ref - 1
        if 0 <= idx < len(existing):
            results.append((existing[idx], verdict.reason))

    return results
