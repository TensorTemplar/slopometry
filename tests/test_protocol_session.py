"""Tests for SessionManager — per-source sequence numbering and legacy-file migration.

SessionManager is constructed with an explicit `source` so concurrent harnesses
don't share sequence state, and its `state_root` can be redirected to a tmp dir
for testing. On first construction for source='claude_code', it migrates any
legacy files at `~/.claude/slopometry/seq_*.txt` to
`~/.slopometry/sessions/claude_code/seq_*.txt`.

The legacy/default state dirs are module-level constants computed at import
time from `Path.home()`, so the tests patch those constants directly rather
than `Path.home()`. Each sequence-numbering test uses its own tmp_path to
avoid cross-test contamination.
"""

from slopometry.core.protocol import session as session_module
from slopometry.core.protocol.session import SessionManager


class TestSessionManagerSequenceNumbering:
    """Tests for monotonic per-session sequence numbering."""

    def test_get_next_sequence_number__first_call_returns_one(self, tmp_path):
        """First event for a fresh session_id gets sequence_number=1."""
        sm = SessionManager(source="claude_code", state_root=tmp_path)
        assert sm.get_next_sequence_number("fresh-session") == 1

    def test_get_next_sequence_number__increments_monotonically(self, tmp_path):
        """Subsequent events get sequence_number 2, 3, 4, ... for the same session."""
        sm = SessionManager(source="claude_code", state_root=tmp_path)
        assert sm.get_next_sequence_number("s") == 1
        assert sm.get_next_sequence_number("s") == 2
        assert sm.get_next_sequence_number("s") == 3

    def test_get_next_sequence_number__distinct_sessions_have_independent_counters(self, tmp_path):
        """Sequence numbers reset per session — different sessions start from 1."""
        sm = SessionManager(source="claude_code", state_root=tmp_path)
        assert sm.get_next_sequence_number("session-a") == 1
        assert sm.get_next_sequence_number("session-a") == 2
        assert sm.get_next_sequence_number("session-b") == 1
        assert sm.get_next_sequence_number("session-a") == 3

    def test_get_next_sequence_number__survives_corrupt_sequence_file(self, tmp_path):
        """A corrupt sequence file resets to 1 (logs but does not raise)."""
        sm = SessionManager(source="claude_code", state_root=tmp_path)
        seq_file = sm.state_dir / "seq_corrupt.txt"
        seq_file.write_text("not-a-number")
        assert sm.get_next_sequence_number("corrupt") == 1

    def test_session_manager__state_dir_created_on_construction(self, tmp_path):
        """The source-scoped state directory is created when SessionManager is built."""
        sm = SessionManager(source="claude_code", state_root=tmp_path)
        assert sm.state_dir.exists()
        assert sm.state_dir == tmp_path / "claude_code"

    def test_session_manager__source_isolates_state(self, tmp_path):
        """claude_code and opencode state dirs are separate under the same root."""
        sm_cc = SessionManager(source="claude_code", state_root=tmp_path)
        sm_oc = SessionManager(source="opencode", state_root=tmp_path)
        assert sm_cc.state_dir != sm_oc.state_dir
        assert sm_cc.state_dir.name == "claude_code"
        assert sm_oc.state_dir.name == "opencode"

    def test_session_manager__sequence_persists_across_instances(self, tmp_path):
        """The sequence number on disk is the source of truth — a new SessionManager reads it back."""
        sm1 = SessionManager(source="claude_code", state_root=tmp_path)
        sm1.get_next_sequence_number("persist-s")
        sm1.get_next_sequence_number("persist-s")
        sm2 = SessionManager(source="claude_code", state_root=tmp_path)
        assert sm2.get_next_sequence_number("persist-s") == 3


class TestSessionManagerLegacyMigration:
    """Tests for migration of `~/.claude/slopometry/seq_*.txt` legacy files.

    `_LEGACY_STATE_DIR` is a module-level constant resolved at import time from
    `Path.home()`. We patch it directly to point at a tmp dir so the test is
    hermetic and doesn't touch the real user's `~/.claude` directory.
    """

    def test_legacy_migration__moves_seq_files_to_new_state_dir(self, tmp_path, monkeypatch):
        """A seq_*.txt file in the legacy dir is moved to <state_root>/claude_code/."""
        legacy_dir = tmp_path / "legacy_claude_slopometry"
        legacy_dir.mkdir()
        (legacy_dir / "seq_legacy-session.txt").write_text("5")

        new_root = tmp_path / "new_root"
        monkeypatch.setattr(session_module, "_LEGACY_STATE_DIR", legacy_dir)

        SessionManager(source="claude_code", state_root=new_root)

        migrated = new_root / "claude_code" / "seq_legacy-session.txt"
        assert migrated.exists(), "legacy file should be migrated to new state dir"
        assert migrated.read_text() == "5"
        assert not (legacy_dir / "seq_legacy-session.txt").exists(), "legacy file should be moved, not copied"

    def test_legacy_migration__does_not_move_files_for_other_sources(self, tmp_path, monkeypatch):
        """opencode sessions do not migrate files from ~/.claude/slopometry/."""
        legacy_dir = tmp_path / "legacy_claude_slopometry_oc"
        legacy_dir.mkdir()
        (legacy_dir / "seq_oc-session.txt").write_text("2")

        new_root = tmp_path / "new_root_oc"
        monkeypatch.setattr(session_module, "_LEGACY_STATE_DIR", legacy_dir)

        sm = SessionManager(source="opencode", state_root=new_root)

        assert sm.state_dir == new_root / "opencode"
        assert not (new_root / "opencode" / "seq_oc-session.txt").exists()
        assert (legacy_dir / "seq_oc-session.txt").exists(), "legacy file must not be moved for non-claude_code source"

    def test_legacy_migration__skips_when_legacy_dir_missing(self, tmp_path, monkeypatch):
        """No legacy dir -> no migration, no error."""
        missing_legacy = tmp_path / "legacy_dir_does_not_exist"
        new_root = tmp_path / "new_root_no_legacy"
        monkeypatch.setattr(session_module, "_LEGACY_STATE_DIR", missing_legacy)

        sm = SessionManager(source="claude_code", state_root=new_root)

        assert sm.state_dir == new_root / "claude_code"

    def test_legacy_migration__skips_when_target_already_exists(self, tmp_path, monkeypatch):
        """A pre-existing target file is preserved — the migration is idempotent."""
        legacy_dir = tmp_path / "legacy_claude_slopometry_idem"
        legacy_dir.mkdir()
        (legacy_dir / "seq_idem.txt").write_text("legacy-value")

        new_root = tmp_path / "new_root_idem"
        target = new_root / "claude_code" / "seq_idem.txt"
        target.parent.mkdir(parents=True)
        target.write_text("newer-value")

        monkeypatch.setattr(session_module, "_LEGACY_STATE_DIR", legacy_dir)

        SessionManager(source="claude_code", state_root=new_root)

        assert target.read_text() == "newer-value", "existing target must not be overwritten"
