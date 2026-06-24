"""Tests for TranscriptFinder (Claude Code + OpenCode discovery)."""

import json
from pathlib import Path

import pytest

from slopometry.core.models.protocol.events import AbstractEventSource
from slopometry.solo.services.transcript_finder import (
    DiscoveredTranscript,
    TranscriptFinder,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_opencode_storage(
    root: Path,
    project_worktree: Path,
    *,
    session_ids: list[str],
    message_layout: dict[str, list[dict]],
    project_id: str | None = None,
) -> tuple[str, list[str]]:
    """Build a minimal OpenCode storage tree under ``root``.

    Returns (project_id, session_ids).
    """
    project_id = project_id or f"proj_{abs(hash(str(project_worktree)))}"
    _write_json(
        root / "project" / f"{project_id}.json",
        {"id": project_id, "worktree": str(project_worktree), "vcs": "git", "time": {"created": 0}},
    )
    session_dir = root / "session" / project_id
    session_dir.mkdir(parents=True, exist_ok=True)
    for sid in session_ids:
        _write_json(session_dir / f"{sid}.json", {"id": sid, "projectID": project_id, "directory": str(project_worktree)})
        msg_dir = root / "message" / sid
        msg_dir.mkdir(parents=True, exist_ok=True)
        for i, msg in enumerate(message_layout.get(sid, [])):
            mid = msg["id"]
            _write_json(msg_dir / f"{mid}.json", {**msg, "sessionID": sid})
            part_dir = root / "part" / mid
            part_dir.mkdir(parents=True, exist_ok=True)
            for j, part in enumerate(msg["parts"]):
                _write_json(part_dir / f"part_{i}_{j}.json", {**part, "messageID": mid, "sessionID": sid})
    return project_id, session_ids


@pytest.fixture
def storage_finder(monkeypatch: pytest.MonkeyPatch):
    """TranscriptFinder whose find_opencode_storage_root is monkey-patchable per test."""
    return TranscriptFinder()


def test_opencode_session_emits_opencode_source(
    storage_finder: TranscriptFinder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    storage = tmp_path / "opencode_storage"
    _make_opencode_storage(
        storage,
        project_dir,
        session_ids=["ses_a", "ses_b"],
        message_layout={
            "ses_a": [
                {
                    "id": "msg_a1",
                    "role": "user",
                    "time": {"created": 1000},
                    "parts": [{"type": "text", "text": "hi"}],
                }
            ],
            "ses_b": [
                {
                    "id": "msg_b1",
                    "role": "assistant",
                    "time": {"created": 2000},
                    "parts": [{"type": "text", "text": "hello"}],
                }
            ],
        },
    )
    monkeypatch.setattr(storage_finder, "find_opencode_storage_root", lambda: storage)

    results = storage_finder.discover_transcripts(project_dir)

    opencode_results = [r for r in results if r.source == AbstractEventSource.OPENCODE]
    assert len(opencode_results) == 2
    assert {r.session_id for r in opencode_results} == {"ses_a", "ses_b"}
    assert all(isinstance(r, DiscoveredTranscript) for r in opencode_results)


def test_non_matching_worktree_excluded(
    storage_finder: TranscriptFinder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    other_project = tmp_path / "other"
    other_project.mkdir()
    storage = tmp_path / "opencode_storage"
    _make_opencode_storage(
        storage,
        other_project,
        session_ids=["ses_x"],
        message_layout={
            "ses_x": [
                {
                    "id": "msg_x1",
                    "role": "user",
                    "time": {"created": 0},
                    "parts": [{"type": "text", "text": "x"}],
                }
            ]
        },
    )
    monkeypatch.setattr(storage_finder, "find_opencode_storage_root", lambda: storage)

    results = storage_finder.discover_transcripts(project_dir)
    assert results == []


def test_missing_opencode_storage_root_returns_only_claude(
    storage_finder: TranscriptFinder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    project_dir = tmp_path / "myproject"
    project_dir.mkdir()
    monkeypatch.setattr(storage_finder, "find_opencode_storage_root", lambda: None)
    results = storage_finder.discover_transcripts(project_dir)
    assert all(r.source == AbstractEventSource.CLAUDE_CODE for r in results)


def test_slopometry_transcript_marked_as_claude_code(tmp_path: Path):
    project_dir = tmp_path / "myproject"
    slop_dir = project_dir / ".slopometry" / "ses_abc"
    slop_dir.mkdir(parents=True)
    (slop_dir / "transcript.jsonl").write_text('{"type":"user"}\n', encoding="utf-8")
    results = TranscriptFinder().discover_transcripts(project_dir)
    assert any(
        r.session_id == "ses_abc" and r.source == AbstractEventSource.CLAUDE_CODE for r in results
    )


def test_only_matching_worktree_included(
    storage_finder: TranscriptFinder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    project_dir = tmp_path / "match"
    project_dir.mkdir()
    storage = tmp_path / "opencode_storage"
    _make_opencode_storage(
        storage,
        project_dir,
        session_ids=["ses_keep"],
        message_layout={
            "ses_keep": [
                {
                    "id": "msg_keep",
                    "role": "user",
                    "time": {"created": 1},
                    "parts": [{"type": "text", "text": "keep"}],
                }
            ]
        },
    )
    _make_opencode_storage(
        storage,
        tmp_path / "nomatch",
        session_ids=["ses_drop"],
        message_layout={
            "ses_drop": [
                {
                    "id": "msg_drop",
                    "role": "user",
                    "time": {"created": 1},
                    "parts": [{"type": "text", "text": "drop"}],
                }
            ]
        },
    )
    monkeypatch.setattr(storage_finder, "find_opencode_storage_root", lambda: storage)
    results = storage_finder.find_opencode_sessions(project_dir)
    assert {r.session_id for r in results} == {"ses_keep"}


def test_find_opencode_storage_root_returns_none_when_no_xdg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    finder = TranscriptFinder()
    assert finder.find_opencode_storage_root() == tmp_path / ".local" / "share" / "opencode" / "storage"


def test_find_opencode_storage_root_uses_xdg_when_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("XDG_DATA_HOME", "/custom/xdg")
    finder = TranscriptFinder()
    assert finder.find_opencode_storage_root() == Path("/custom/xdg/opencode/storage")


def test_opencode_session_in_subdirectory_of_worktree_included(
    storage_finder: TranscriptFinder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    project_root = tmp_path / "root"
    subdir = project_root / "packages" / "core"
    subdir.mkdir(parents=True)
    storage = tmp_path / "opencode_storage"
    _make_opencode_storage(
        storage,
        project_root,
        session_ids=["ses_subdir"],
        message_layout={
            "ses_subdir": [
                {
                    "id": "msg_subdir",
                    "role": "user",
                    "time": {"created": 1},
                    "parts": [{"type": "text", "text": "from subdir"}],
                }
            ]
        },
        project_id="proj_subdir",
    )
    session_meta_path = storage / "session" / "proj_subdir" / "ses_subdir.json"
    session_meta_path.write_text(
        json.dumps(
            {
                "id": "ses_subdir",
                "projectID": "proj_subdir",
                "directory": str(subdir),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(storage_finder, "find_opencode_storage_root", lambda: storage)
    results_subdir = storage_finder.find_opencode_sessions(subdir)
    assert {r.session_id for r in results_subdir} == {"ses_subdir"}

    results_root = storage_finder.find_opencode_sessions(project_root)
    assert results_root == []


def test_opencode_session_outside_worktree_excluded(
    storage_finder: TranscriptFinder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    worktree = tmp_path / "worktree"
    other = tmp_path / "unrelated"
    worktree.mkdir()
    other.mkdir()
    storage = tmp_path / "opencode_storage"
    _make_opencode_storage(
        storage,
        worktree,
        session_ids=["ses_unrelated"],
        message_layout={
            "ses_unrelated": [
                {
                    "id": "msg_unrelated",
                    "role": "user",
                    "time": {"created": 1},
                    "parts": [{"type": "text", "text": "x"}],
                }
            ]
        },
        project_id="proj_unrelated",
    )
    session_meta_path = storage / "session" / "proj_unrelated" / "ses_unrelated.json"
    session_meta_path.write_text(
        json.dumps(
            {
                "id": "ses_unrelated",
                "projectID": "proj_unrelated",
                "directory": str(other),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(storage_finder, "find_opencode_storage_root", lambda: storage)
    results = storage_finder.find_opencode_sessions(worktree)
    assert results == []
