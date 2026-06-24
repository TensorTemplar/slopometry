"""Tests for MemoryFreshnessValidator."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from slopometry.core.models.memory import MemoryCandidate, MemoryEntry, MemoryType
from slopometry.solo.services.memory_freshness import (
    CEILING_THRESHOLD,
    FLOOR_THRESHOLD,
    FreshnessDecision,
    MemoryFreshnessValidator,
    ProjectSimilarityDistribution,
    _cosine_similarity,
    _find_above_threshold,
    _judge_reconciliation,
    compute_project_distribution,
)


def _candidate(content: str, embedding: list[float] | None = None) -> MemoryCandidate:
    return MemoryCandidate(
        memory_type=MemoryType.PROJECT,
        content=content,
        embedding=embedding,
    )


def _memory(
    content: str,
    embedding: list[float] | None = None,
    mem_id: str = "mem-1",
) -> MemoryEntry:
    return MemoryEntry(
        id=mem_id,
        session_id="s1",
        project_dir="/proj",
        memory_type=MemoryType.PROJECT,
        content=content,
        embedding=embedding,
        created_at=datetime.now(),
    )


class TestCosineSimilarity:
    def test_identical_vectors_have_similarity_one(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_have_similarity_zero(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_empty_vectors_return_zero(self):
        assert _cosine_similarity([], [1.0]) == 0.0

    def test_mismatched_lengths_return_zero(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


class TestProjectSimilarityDistribution:
    def test_zero_pairs_falls_back_to_floor(self):
        d = ProjectSimilarityDistribution(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert d.derived_threshold == FLOOR_THRESHOLD

    def test_threshold_is_p75_clamped_to_floor(self):
        d = ProjectSimilarityDistribution(10, 0.30, 0.30, 0.20, 0.10, 0.05)
        assert d.derived_threshold == FLOOR_THRESHOLD

    def test_threshold_is_p75_when_above_floor(self):
        d = ProjectSimilarityDistribution(100, 0.70, 0.65, 0.80, 0.90, 0.95)
        assert d.derived_threshold == pytest.approx(0.80)

    def test_threshold_is_clamped_to_ceiling(self):
        d = ProjectSimilarityDistribution(100, 0.95, 0.95, 0.99, 1.0, 1.0)
        assert d.derived_threshold == CEILING_THRESHOLD


class TestComputeProjectDistribution:
    def test_no_embeddings_returns_zero_distribution(self):
        existing = [_memory("X", embedding=None), _memory("Y", embedding=None)]
        d = compute_project_distribution(existing)
        assert d.n_pairs == 0

    def test_pairs_counted_correctly(self):
        existing = [
            _memory("a", [1.0, 0.0], "m1"),
            _memory("b", [0.0, 1.0], "m2"),
            _memory("c", [1.0, 0.0], "m3"),
        ]
        d = compute_project_distribution(existing)
        assert d.n_pairs == 3
        assert 0.0 <= d.mean <= 1.0
        assert 0.0 <= d.p50 <= 1.0
        assert 0.0 <= d.p75 <= 1.0

    def test_quantiles_are_monotonic(self):
        existing = [_memory(f"m{i}", [float(i) / 10, 1.0 - float(i) / 10], f"id{i}") for i in range(5)]
        d = compute_project_distribution(existing)
        assert d.p50 <= d.p75 <= d.p90 <= d.p95


class TestFindAboveThreshold:
    def test_returns_only_memories_above_threshold(self):
        candidate = _candidate("X", [1.0, 0.0])
        existing = [
            _memory("identical", [1.0, 0.0], "m1"),
            _memory("near", [0.95, 0.31], "m2"),
            _memory("far", [0.0, 1.0], "m3"),
        ]
        matches = _find_above_threshold(candidate, existing, threshold=0.78)
        ids = [m.id for m, _ in matches]
        assert "m3" not in ids
        assert "m1" in ids
        assert "m2" in ids

    def test_candidate_without_embedding_returns_empty(self):
        candidate = _candidate("X", embedding=None)
        existing = [_memory("Y", [1.0, 0.0], "m1")]
        assert _find_above_threshold(candidate, existing, threshold=0.5) == []

    def test_returns_empty_when_no_matches_above_threshold(self):
        candidate = _candidate("X", [1.0, 0.0])
        existing = [_memory("Y", [0.0, 1.0], "m1")]
        assert _find_above_threshold(candidate, existing, threshold=0.78) == []


class TestJudgeReconciliation:
    def test_returns_keep_both_when_llm_says_so(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "keep_both", "reason": "different topics"}'))
        ]
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("uses rust-code-analysis"),
                _memory("user prefers dark mode"),
                "https://llm.example/v1",
                "model-x",
                "key",
            )
        assert decision.action == "keep_both"
        assert "topics" in decision.reason or "different" in decision.reason

    def test_returns_merge_with_merged_content(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"action": "merge", "reason": "old was outdated", "merged_content": "uses rust-code-analysis since 2026"}'
                )
            )
        ]
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("uses rust-code-analysis"),
                _memory("uses radon"),
                "https://llm.example/v1",
                "model-x",
                "key",
            )
        assert decision.action == "merge"
        assert decision.merged_content == "uses rust-code-analysis since 2026"

    def test_returns_supersede_when_llm_says_so(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "supersede", "reason": "newer version"}'))
        ]
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("Python 3.13"),
                _memory("Python 3.10"),
                "https://llm.example/v1",
                "model-x",
                "key",
            )
        assert decision.action == "supersede"

    def test_returns_dedupe_when_llm_says_so(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "dedupe", "reason": "same info"}'))
        ]
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("user uses pyright type checker"),
                _memory("user prefers pyright"),
                "https://llm.example/v1",
                "model-x",
                "key",
            )
        assert decision.action == "dedupe"

    def test_strips_markdown_fences(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content='```json\n{"action": "merge", "reason": "old outdated", "merged_content": "merged"}\n```')
            )
        ]
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("X"),
                _memory("Y"),
                "https://llm.example/v1",
                "model-x",
                "key",
            )
        assert decision.action == "merge"
        assert decision.merged_content == "merged"

    def test_falls_back_to_keep_both_on_invalid_action(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "maybe", "reason": "unsure"}'))
        ]
        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("X"),
                _memory("Y"),
                "https://llm.example/v1",
                "model-x",
                "key",
            )
        assert decision.action == "keep_both"


class TestMemoryFreshnessValidator:
    def test_no_existing_memories_returns_empty_decisions_and_floor_distribution(self):
        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        decisions, distribution = validator.validate([_candidate("X", [1.0, 0.0])], [])
        assert decisions == []
        assert distribution.n_pairs == 0
        assert distribution.derived_threshold == FLOOR_THRESHOLD

    def test_no_above_threshold_matches_skips_llm_call(self):
        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        candidates = [_candidate("X", [1.0, 0.0])]
        existing = [_memory("orthogonal", [0.0, 1.0], "m1")]

        with patch("openai.OpenAI") as mock_openai:
            decisions, _ = validator.validate(candidates, existing)
            mock_openai.assert_not_called()

        assert decisions == []

    def test_similar_match_triggers_llm_judge_with_action(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"action": "merge", "reason": "update", "merged_content": "merged"}'
                )
            )
        ]
        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        candidates = [_candidate("uses rust-code-analysis", [1.0, 0.0])]
        existing = [_memory("uses radon", [0.99, 0.14], "m1")]

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decisions, distribution = validator.validate(candidates, existing)

        assert len(decisions) == 1
        decision = decisions[0]
        assert isinstance(decision, FreshnessDecision)
        assert decision.action == "merge"
        assert decision.merged_content == "merged"
        assert decision.similarity > distribution.derived_threshold

    def test_keep_both_action_does_not_merge_or_supersede(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "keep_both", "reason": "different aspects"}'))
        ]
        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        candidates = [_candidate("uses rust-code-analysis for complexity", [1.0, 0.0])]
        existing = [_memory("user prefers dark mode", [0.99, 0.14], "m1")]

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decisions, _ = validator.validate(candidates, existing)

        assert len(decisions) == 1
        assert decisions[0].action == "keep_both"

    def test_multiple_candidates_with_different_actions(self):
        mock_response_a = MagicMock()
        mock_response_a.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"action": "merge", "reason": "update", "merged_content": "merged-a"}'
                )
            )
        ]
        mock_response_b = MagicMock()
        mock_response_b.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"action": "dedupe", "reason": "same info"}'
                )
            )
        ]
        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        candidates = [
            _candidate("uses rust-code-analysis for complexity", [1.0, 0.0]),
            _candidate("user prefers dark mode in editors", [0.0, 1.0]),
        ]
        existing = [
            _memory("uses radon for complexity", [0.99, 0.14], "m1"),
            _memory("user prefers light mode in editors", [0.14, 0.99], "m2"),
        ]

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = [
                mock_response_a,
                mock_response_b,
            ]
            decisions, _ = validator.validate(candidates, existing)

        assert len(decisions) == 2
        actions = {d.action for d in decisions}
        assert "merge" in actions
        assert "dedupe" in actions

    def test_data_driven_threshold_for_low_similarity_project_is_low(self):
        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        candidates = [_candidate("X", [1.0, 0.0])]
        existing = [
            _memory("a", [1.0, 0.0], "m1"),
            _memory("b", [0.0, 1.0], "m2"),
        ]
        with patch("openai.OpenAI"):
            _, distribution = validator.validate(candidates, existing)
        assert distribution.derived_threshold == FLOOR_THRESHOLD

    def test_failed_llm_call_skipped_silently(self):
        validator = MemoryFreshnessValidator("https://llm.example/v1", "model-x")
        candidates = [_candidate("X", [1.0, 0.0])]
        existing = [_memory("Y", [0.99, 0.14], "m1")]

        with patch("openai.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("llm down")
            decisions, _ = validator.validate(candidates, existing)

        assert decisions == []
