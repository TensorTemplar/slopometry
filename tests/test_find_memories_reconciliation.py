"""Tests for the freshness reconciliation action priority logic in find-memories.

When a single candidate matches multiple existing memories, the LLM may
return different actions for each pair. The priority resolution ensures
only one action wins: SUPERSEDE > MERGE > DEDUPE > KEEP_BOTH.
"""

from datetime import datetime

from slopometry.core.models.memory import (
    FreshnessAction,
    MemoryCandidate,
    MemoryEntry,
    MemoryType,
)
from slopometry.solo.cli.commands import _ACTION_PRIORITY, _highest_priority_action
from slopometry.solo.services.memory_freshness import FreshnessDecision


def _candidate(content: str, embedding: list[float] | None = None) -> MemoryCandidate:
    return MemoryCandidate(
        memory_type=MemoryType.PROJECT,
        content=content,
        embedding=embedding,
    )


def _memory(content: str, mem_id: str = "m1") -> MemoryEntry:
    return MemoryEntry(
        id=mem_id,
        session_id="s1",
        project_dir="/proj",
        memory_type=MemoryType.PROJECT,
        content=content,
        created_at=datetime.now(),
    )


def _decision(
    candidate: MemoryCandidate,
    existing: MemoryEntry,
    action: FreshnessAction,
    similarity: float = 0.9,
    merged_content: str | None = None,
) -> FreshnessDecision:
    return FreshnessDecision(
        new_candidate=candidate,
        existing_memory=existing,
        similarity=similarity,
        action=action,
        reason=f"test-{action.value}",
        merged_content=merged_content,
    )


class TestActionPriority:
    def test_action_priority__supersede_beats_merge_dedupe_keep_both(self):
        assert _ACTION_PRIORITY[FreshnessAction.SUPERSEDE] > _ACTION_PRIORITY[FreshnessAction.MERGE]
        assert _ACTION_PRIORITY[FreshnessAction.SUPERSEDE] > _ACTION_PRIORITY[FreshnessAction.DEDUPE]
        assert _ACTION_PRIORITY[FreshnessAction.SUPERSEDE] > _ACTION_PRIORITY[FreshnessAction.KEEP_BOTH]

    def test_action_priority__merge_beats_dedupe_keep_both(self):
        assert _ACTION_PRIORITY[FreshnessAction.MERGE] > _ACTION_PRIORITY[FreshnessAction.DEDUPE]
        assert _ACTION_PRIORITY[FreshnessAction.MERGE] > _ACTION_PRIORITY[FreshnessAction.KEEP_BOTH]

    def test_action_priority__dedupe_beats_keep_both(self):
        assert _ACTION_PRIORITY[FreshnessAction.DEDUPE] > _ACTION_PRIORITY[FreshnessAction.KEEP_BOTH]

    def test_highest_priority_action__returns_supersede_when_group_has_supersede_and_merge(self):
        cand = _candidate("new version")
        group = [
            _decision(cand, _memory("old A", "m1"), FreshnessAction.MERGE, merged_content="merged"),
            _decision(cand, _memory("old B", "m2"), FreshnessAction.SUPERSEDE),
        ]
        assert _highest_priority_action(group) == FreshnessAction.SUPERSEDE

    def test_highest_priority_action__returns_merge_when_group_has_merge_and_dedupe(self):
        cand = _candidate("updated info")
        group = [
            _decision(cand, _memory("duplicate", "m1"), FreshnessAction.DEDUPE),
            _decision(cand, _memory("outdated", "m2"), FreshnessAction.MERGE, merged_content="merged"),
        ]
        assert _highest_priority_action(group) == FreshnessAction.MERGE

    def test_highest_priority_action__returns_dedupe_when_group_has_dedupe_and_keep_both(self):
        cand = _candidate("same info")
        group = [
            _decision(cand, _memory("different topic", "m1"), FreshnessAction.KEEP_BOTH),
            _decision(cand, _memory("same info", "m2"), FreshnessAction.DEDUPE),
        ]
        assert _highest_priority_action(group) == FreshnessAction.DEDUPE

    def test_highest_priority_action__returns_keep_both_when_all_are_keep_both(self):
        cand = _candidate("unique info")
        group = [
            _decision(cand, _memory("A", "m1"), FreshnessAction.KEEP_BOTH),
            _decision(cand, _memory("B", "m2"), FreshnessAction.KEEP_BOTH),
        ]
        assert _highest_priority_action(group) == FreshnessAction.KEEP_BOTH

    def test_highest_priority_action__handles_single_decision_group(self):
        cand = _candidate("X")
        group = [_decision(cand, _memory("Y", "m1"), FreshnessAction.MERGE, merged_content="merged")]
        assert _highest_priority_action(group) == FreshnessAction.MERGE

    def test_highest_priority_action__supersede_wins_over_all_four_actions(self):
        cand = _candidate("new truth")
        group = [
            _decision(cand, _memory("dup", "m1"), FreshnessAction.DEDUPE),
            _decision(cand, _memory("mergeable", "m2"), FreshnessAction.MERGE, merged_content="merged"),
            _decision(cand, _memory("different", "m3"), FreshnessAction.KEEP_BOTH),
            _decision(cand, _memory("outdated", "m4"), FreshnessAction.SUPERSEDE),
        ]
        assert _highest_priority_action(group) == FreshnessAction.SUPERSEDE
