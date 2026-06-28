"""Tests for validate_freshness and audit_staleness."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from slopometry.core.models.memory import (
    FreshnessAction,
    MemoryCandidate,
    MemoryEntry,
    MemoryType,
    StalenessVerdict,
)
from slopometry.solo.services.memory_freshness import (
    DEFAULT_CEILING_THRESHOLD,
    DEFAULT_FLOOR_THRESHOLD,
    FreshnessDecision,
    ProjectSimilarityDistribution,
    _cosine_similarity,
    _find_above_threshold,
    _judge_reconciliation,
    audit_staleness,
    compute_project_distribution,
    validate_freshness,
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
    def test_cosine_similarity__returns_one_for_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity__returns_zero_for_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_similarity__returns_zero_for_empty_vectors(self):
        assert _cosine_similarity([], [1.0]) == 0.0

    def test_cosine_similarity__returns_zero_for_mismatched_lengths(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


class TestProjectSimilarityDistribution:
    def test_derived_threshold__falls_back_to_floor_when_zero_pairs(self):
        d = ProjectSimilarityDistribution(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert d.derived_threshold == DEFAULT_FLOOR_THRESHOLD

    def test_derived_threshold__clamped_to_floor_when_p75_below_floor(self):
        d = ProjectSimilarityDistribution(10, 0.30, 0.30, 0.20, 0.10, 0.05)
        assert d.derived_threshold == DEFAULT_FLOOR_THRESHOLD

    def test_derived_threshold__uses_p75_when_above_floor(self):
        d = ProjectSimilarityDistribution(100, 0.70, 0.65, 0.80, 0.90, 0.95)
        assert d.derived_threshold == pytest.approx(0.80)

    def test_derived_threshold__clamped_to_ceiling_when_p75_above_ceiling(self):
        d = ProjectSimilarityDistribution(100, 0.95, 0.95, 0.99, 1.0, 1.0)
        assert d.derived_threshold == DEFAULT_CEILING_THRESHOLD


class TestComputeProjectDistribution:
    def test_compute_project_distribution__returns_zero_distribution_when_no_embeddings(self):
        existing = [_memory("X", embedding=None), _memory("Y", embedding=None)]
        d = compute_project_distribution(existing)
        assert d.n_pairs == 0

    def test_compute_project_distribution__counts_pairs_correctly(self):
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

    def test_compute_project_distribution__quantiles_are_monotonic(self):
        existing = [_memory(f"m{i}", [float(i) / 10, 1.0 - float(i) / 10], f"id{i}") for i in range(5)]
        d = compute_project_distribution(existing)
        assert d.p50 <= d.p75 <= d.p90 <= d.p95


class TestFindAboveThreshold:
    def test_find_above_threshold__returns_only_memories_above_threshold(self):
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

    def test_find_above_threshold__returns_empty_when_candidate_has_no_embedding(self):
        candidate = _candidate("X", embedding=None)
        existing = [_memory("Y", [1.0, 0.0], "m1")]
        assert _find_above_threshold(candidate, existing, threshold=0.5) == []

    def test_find_above_threshold__returns_empty_when_no_matches_above_threshold(self):
        candidate = _candidate("X", [1.0, 0.0])
        existing = [_memory("Y", [0.0, 1.0], "m1")]
        assert _find_above_threshold(candidate, existing, threshold=0.78) == []


class TestJudgeReconciliation:
    def test_judge_reconciliation__returns_keep_both_when_llm_says_keep_both(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "keep_both", "reason": "different topics"}'))
        ]
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("uses rust-code-analysis"),
                _memory("user prefers dark mode"),
                "https://llm.example/v1",
                "model-x",
                "key", 0.85,
            )
        assert decision.action == FreshnessAction.KEEP_BOTH
        assert "topics" in decision.reason or "different" in decision.reason

    def test_judge_reconciliation__returns_merge_with_merged_content_when_llm_says_merge(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"action": "merge", "reason": "old was outdated", "merged_content": "uses rust-code-analysis since 2026"}'
                )
            )
        ]
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("uses rust-code-analysis"),
                _memory("uses radon"),
                "https://llm.example/v1",
                "model-x",
                "key", 0.85,
            )
        assert decision.action == FreshnessAction.MERGE
        assert decision.merged_content == "uses rust-code-analysis since 2026"

    def test_judge_reconciliation__returns_supersede_when_llm_says_supersede(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "supersede", "reason": "newer version"}'))
        ]
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("Python 3.13"),
                _memory("Python 3.10"),
                "https://llm.example/v1",
                "model-x",
                "key", 0.85,
            )
        assert decision.action == FreshnessAction.SUPERSEDE

    def test_judge_reconciliation__returns_dedupe_when_llm_says_dedupe(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "dedupe", "reason": "same info"}'))
        ]
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("user uses pyright type checker"),
                _memory("user prefers pyright"),
                "https://llm.example/v1",
                "model-x",
                "key", 0.85,
            )
        assert decision.action == FreshnessAction.DEDUPE

    def test_judge_reconciliation__strips_markdown_fences_from_llm_response(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content='```json\n{"action": "merge", "reason": "old outdated", "merged_content": "merged"}\n```')
            )
        ]
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("X"),
                _memory("Y"),
                "https://llm.example/v1",
                "model-x",
                "key", 0.85,
            )
        assert decision.action == FreshnessAction.MERGE
        assert decision.merged_content == "merged"

    def test_judge_reconciliation__falls_back_to_keep_both_on_invalid_action(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "maybe", "reason": "unsure"}'))
        ]
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decision = _judge_reconciliation(
                _candidate("X"),
                _memory("Y"),
                "https://llm.example/v1",
                "model-x",
                "key", 0.85,
            )
        assert decision.action == FreshnessAction.KEEP_BOTH


class TestValidateFreshness:
    def test_validate_freshness__returns_empty_decisions_and_floor_distribution_when_no_existing_memories(self):
        decisions, distribution = validate_freshness(
            [_candidate("X", [1.0, 0.0])], [], "https://llm.example/v1", "model-x", "test-key"
        )
        assert decisions == []
        assert distribution.n_pairs == 0
        assert distribution.derived_threshold == DEFAULT_FLOOR_THRESHOLD

    def test_validate_freshness__skips_llm_call_when_no_above_threshold_matches(self):
        candidates = [_candidate("X", [1.0, 0.0])]
        existing = [_memory("orthogonal", [0.0, 1.0], "m1")]

        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            decisions, _ = validate_freshness(candidates, existing, "https://llm.example/v1", "model-x", "test-key")
            mock_openai.assert_not_called()

        assert decisions == []

    def test_validate_freshness__triggers_llm_judge_when_similar_match_found(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"action": "merge", "reason": "update", "merged_content": "merged"}'
                )
            )
        ]
        candidates = [_candidate("uses rust-code-analysis", [1.0, 0.0])]
        existing = [_memory("uses radon", [0.99, 0.14], "m1")]

        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decisions, distribution = validate_freshness(
                candidates, existing, "https://llm.example/v1", "model-x", "test-key"
            )

        assert len(decisions) == 1
        decision = decisions[0]
        assert isinstance(decision, FreshnessDecision)
        assert decision.action == FreshnessAction.MERGE
        assert decision.merged_content == "merged"
        assert decision.similarity > distribution.derived_threshold

    def test_validate_freshness__does_not_merge_or_supersede_on_keep_both(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "keep_both", "reason": "different aspects"}'))
        ]
        candidates = [_candidate("uses rust-code-analysis for complexity", [1.0, 0.0])]
        existing = [_memory("user prefers dark mode", [0.99, 0.14], "m1")]

        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            decisions, _ = validate_freshness(
                candidates, existing, "https://llm.example/v1", "model-x", "test-key"
            )

        assert len(decisions) == 1
        assert decisions[0].action == FreshnessAction.KEEP_BOTH

    def test_validate_freshness__handles_multiple_candidates_with_different_actions(self):
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
        candidates = [
            _candidate("uses rust-code-analysis for complexity", [1.0, 0.0]),
            _candidate("user prefers dark mode in editors", [0.0, 1.0]),
        ]
        existing = [
            _memory("uses radon for complexity", [0.99, 0.14], "m1"),
            _memory("user prefers light mode in editors", [0.14, 0.99], "m2"),
        ]

        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = [
                mock_response_a,
                mock_response_b,
            ]
            decisions, _ = validate_freshness(
                candidates, existing, "https://llm.example/v1", "model-x", "test-key"
            )

        assert len(decisions) == 2
        actions = {d.action for d in decisions}
        assert FreshnessAction.MERGE in actions
        assert FreshnessAction.DEDUPE in actions

    def test_validate_freshness__uses_floor_threshold_for_low_similarity_project(self):
        candidates = [_candidate("X", [1.0, 0.0])]
        existing = [
            _memory("a", [1.0, 0.0], "m1"),
            _memory("b", [0.0, 1.0], "m2"),
        ]
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"action": "keep_both", "reason": "different"}'))
        ]
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = mock_response
            _, distribution = validate_freshness(
                candidates, existing, "https://llm.example/v1", "model-x", "test-key"
            )
        assert distribution.derived_threshold == DEFAULT_FLOOR_THRESHOLD

    def test_validate_freshness__skips_failed_llm_call_silently(self):
        candidates = [_candidate("X", [1.0, 0.0])]
        existing = [_memory("Y", [0.99, 0.14], "m1")]

        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("llm down")
            decisions, _ = validate_freshness(
                candidates, existing, "https://llm.example/v1", "model-x", "test-key"
            )

        assert decisions == []


def _staleness_response(content: str) -> MagicMock:
    mock = MagicMock()
    mock.choices = [MagicMock(message=MagicMock(content=content))]
    return mock


class TestAuditStaleness:
    def test_audit_staleness__returns_empty_list_when_no_existing_memories(self):
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            result = audit_staleness([], "some transcript", "https://llm.example/v1", "model-x", "key")
            mock_openai.assert_not_called()
        assert result == []

    def test_audit_staleness__returns_empty_list_when_transcript_is_empty(self):
        existing = [_memory("describes a bug")]
        result = audit_staleness(existing, "", "https://llm.example/v1", "model-x", "key")
        assert result == []

    def test_audit_staleness__returns_empty_list_when_transcript_is_whitespace_only(self):
        existing = [_memory("describes a bug")]
        result = audit_staleness(existing, "   \n\n  ", "https://llm.example/v1", "model-x", "key")
        assert result == []

    def test_audit_staleness__returns_memory_and_reason_pairs_when_llm_identifies_stale(self):
        existing = [
            _memory("There is a bug in the parser", mem_id="m1"),
            _memory("User prefers dark mode", mem_id="m2"),
        ]
        llm_response = _staleness_response(
            '[{"ref": 1, "reason": "parser bug was fixed in this session"}]'
        )
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            result = audit_staleness(existing, "transcript showing bug fix", "https://llm.example/v1", "model-x", "key")

        assert len(result) == 1
        memory, reason = result[0]
        assert memory.id == "m1"
        assert reason == "parser bug was fixed in this session"

    def test_audit_staleness__returns_multiple_pairs_when_llm_identifies_multiple_stale(self):
        existing = [
            _memory("Bug in parser", mem_id="m1"),
            _memory("TODO: refactor database layer", mem_id="m2"),
            _memory("User prefers dark mode", mem_id="m3"),
        ]
        llm_response = _staleness_response(
            '[{"ref": 1, "reason": "parser bug fixed"}, {"ref": 2, "reason": "database refactor completed"}]'
        )
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            result = audit_staleness(existing, "transcript", "https://llm.example/v1", "model-x", "key")

        assert len(result) == 2
        retired_ids = {m.id for m, _ in result}
        assert retired_ids == {"m1", "m2"}

    def test_audit_staleness__returns_empty_list_when_llm_says_nothing_is_stale(self):
        existing = [_memory("User prefers dark mode", mem_id="m1")]
        llm_response = _staleness_response("[]")
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            result = audit_staleness(existing, "transcript about unrelated work", "https://llm.example/v1", "model-x", "key")

        assert result == []

    def test_audit_staleness__strips_markdown_fences_from_llm_response(self):
        existing = [_memory("describes a bug", mem_id="m1")]
        llm_response = _staleness_response(
            '```json\n[{"ref": 1, "reason": "bug was fixed"}]\n```'
        )
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            result = audit_staleness(existing, "transcript", "https://llm.example/v1", "model-x", "key")

        assert len(result) == 1
        assert result[0][0].id == "m1"

    def test_audit_staleness__skips_verdicts_with_out_of_range_refs(self):
        existing = [_memory("describes a bug", mem_id="m1")]
        llm_response = _staleness_response(
            '[{"ref": 0, "reason": "invalid zero-based ref"}, {"ref": 5, "reason": "out of range"}, {"ref": 1, "reason": "valid ref"}]'
        )
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            result = audit_staleness(existing, "transcript", "https://llm.example/v1", "model-x", "key")

        assert len(result) == 1
        assert result[0][0].id == "m1"
        assert result[0][1] == "valid ref"

    def test_audit_staleness__returns_empty_list_when_llm_response_is_not_a_json_array(self):
        existing = [_memory("describes a bug", mem_id="m1")]
        llm_response = _staleness_response('{"not": "an array"}')
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            result = audit_staleness(existing, "transcript", "https://llm.example/v1", "model-x", "key")

        assert result == []

    def test_audit_staleness__returns_empty_list_on_invalid_json_response(self):
        existing = [_memory("describes a bug", mem_id="m1")]
        llm_response = _staleness_response("this is not json at all")
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            result = audit_staleness(existing, "transcript", "https://llm.example/v1", "model-x", "key")

        assert result == []

    def test_audit_staleness__skips_invalid_verdict_objects_missing_required_fields(self):
        existing = [_memory("describes a bug", mem_id="m1")]
        llm_response = _staleness_response(
            '[{"reason": "missing ref field"}, {"ref": 1, "reason": "valid"}]'
        )
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            result = audit_staleness(existing, "transcript", "https://llm.example/v1", "model-x", "key")

        assert len(result) == 1
        assert result[0][1] == "valid"

    def test_audit_staleness__truncates_transcript_to_configured_char_limit(self):
        existing = [_memory("describes a bug", mem_id="m1")]
        long_transcript = "x" * 30000
        llm_response = _staleness_response("[]")
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            audit_staleness(
                existing,
                long_transcript,
                "https://llm.example/v1",
                "model-x",
                "key",
                transcript_truncation_chars=500,
            )

            call_args = mock_openai.return_value.chat.completions.create.call_args
            user_message = call_args.kwargs["messages"][1]["content"]
            assert "x" * 600 not in user_message

    def test_audit_staleness__uses_1_based_indexing_for_memory_refs(self):
        existing = [
            _memory("first memory", mem_id="m1"),
            _memory("second memory", mem_id="m2"),
            _memory("third memory", mem_id="m3"),
        ]
        llm_response = _staleness_response('[{"ref": 3, "reason": "third is stale"}]')
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            result = audit_staleness(existing, "transcript", "https://llm.example/v1", "model-x", "key")

        assert len(result) == 1
        assert result[0][0].id == "m3"

    def test_audit_staleness__includes_memory_type_and_content_in_prompt(self):
        existing = [
            _memory("describes a bug in parser", mem_id="m1"),
        ]
        llm_response = _staleness_response("[]")
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            audit_staleness(existing, "transcript", "https://llm.example/v1", "model-x", "key")

            call_args = mock_openai.return_value.chat.completions.create.call_args
            user_message = call_args.kwargs["messages"][1]["content"]
            assert "[1]" in user_message
            assert "describes a bug in parser" in user_message
            assert "project" in user_message  # memory_type value

    def test_audit_staleness__only_receives_memories_passed_by_caller_not_external(self):
        existing = [
            _memory("describes a bug", mem_id="m1"),
            _memory("user prefers dark mode", mem_id="m2"),
        ]
        llm_response = _staleness_response('[{"ref": 1, "reason": "bug fixed"}]')
        with patch("slopometry.solo.services.memory_freshness.OpenAI") as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = llm_response
            result = audit_staleness(existing, "transcript", "https://llm.example/v1", "model-x", "key")

        assert len(result) == 1
        assert result[0][0].id == "m1"
        call_args = mock_openai.return_value.chat.completions.create.call_args
        user_message = call_args.kwargs["messages"][1]["content"]
        assert "[1]" in user_message
        assert "[2]" in user_message


class TestStalenessVerdictModel:
    def test_staleness_verdict__accepts_positive_ref_and_reason(self):
        verdict = StalenessVerdict(ref=1, reason="bug was fixed")
        assert verdict.ref == 1
        assert verdict.reason == "bug was fixed"

    def test_staleness_verdict__defaults_reason_to_empty_string(self):
        verdict = StalenessVerdict(ref=1)
        assert verdict.reason == ""
