"""Tests for ContextCoverageAnalyzer using real transcript data."""

import json
import subprocess
from pathlib import Path

import pytest

from slopometry.core.context_coverage_analyzer import ContextCoverageAnalyzer


class TestContextCoverageAnalyzer:
    """Integration tests for context coverage analysis."""

    @pytest.fixture
    def fixture_transcript_path(self):
        """Path to the real transcript fixture."""
        path = Path(__file__).parent / "fixtures" / "transcript.jsonl"
        assert path.exists(), f"transcript.jsonl fixture missing at {path}"
        return path

    @pytest.fixture
    def test_repo_path(self, tmp_path):
        """Create a temporary clone of the repo to match transcript context."""
        source_repo = Path.cwd()
        assert (source_repo / ".git").exists(), "Test must run from within the repository"

        dest_repo_path = tmp_path / "repo"
        subprocess.run(["git", "clone", str(source_repo), str(dest_repo_path)], check=True, capture_output=True)
        return dest_repo_path

    def test_analyze_transcript__parses_real_session(self, test_repo_path, fixture_transcript_path):
        """Test analysis using a real transcript against the repo."""
        analyzer = ContextCoverageAnalyzer(test_repo_path)

        # Run analysis
        coverage = analyzer.analyze_transcript(fixture_transcript_path)

        # Verification based on the known transcript "0f112eb4..."
        # We know it saved plans, so it likely read/wrote files.

        assert coverage is not None

        # Assert structure populated
        assert isinstance(coverage.files_edited, list)
        assert isinstance(coverage.files_read, list)
        assert isinstance(coverage.blind_spots, list)

        # Check if we can find at least one coverage entry if there were edits
        if coverage.files_edited:
            assert len(coverage.file_coverage) == len(coverage.files_edited)
            first_cov = coverage.file_coverage[0]
            assert first_cov.file_path in coverage.files_edited

    def test_import_graph_building(self, test_repo_path):
        """Test that import graph is built correctly for this repo."""
        analyzer = ContextCoverageAnalyzer(test_repo_path)
        analyzer._build_import_graph()

        # Check a known relationship in this codebase
        graph = analyzer._import_graph

        assert graph is not None

        # Sanity check: verify some nodes exist
        found_nodes = [k for k in graph.keys() if k.endswith(".py")]
        assert len(found_nodes) > 0

    def test_analyze_transcript__write_to_new_file_counts_as_read(self, test_repo_path, tmp_path):
        """Files first created via Write should not be blind spots or 'edited without reading'."""
        # Create a file via Write (simulating new file creation in transcript)
        new_file = test_repo_path / "new_module.py"
        new_file.write_text("def hello(): pass")
        subprocess.run(["git", "add", "new_module.py"], cwd=test_repo_path, check=True, capture_output=True)

        transcript_file = tmp_path / "transcript.jsonl"
        events = [
            {"tool_name": "Write", "tool_input": {"file_path": str(new_file)}},
            {"tool_name": "Edit", "tool_input": {"file_path": str(new_file)}},
        ]
        with open(transcript_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        analyzer = ContextCoverageAnalyzer(test_repo_path)
        coverage = analyzer.analyze_transcript(transcript_file)

        cov = next((c for c in coverage.file_coverage if c.file_path == "new_module.py"), None)
        assert cov is not None
        assert cov.was_read_before_edit, "Write-created files should count as read"

    def test_analyze_transcript__edit_without_read_still_flagged(self, test_repo_path, tmp_path):
        """Files edited (not written) without prior read should still be flagged."""
        existing_file = test_repo_path / "existing.py"
        existing_file.write_text("x = 1")
        subprocess.run(["git", "add", "existing.py"], cwd=test_repo_path, check=True, capture_output=True)

        transcript_file = tmp_path / "transcript.jsonl"
        events = [
            {"tool_name": "Edit", "tool_input": {"file_path": str(existing_file)}},
        ]
        with open(transcript_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        analyzer = ContextCoverageAnalyzer(test_repo_path)
        coverage = analyzer.analyze_transcript(transcript_file)

        cov = next((c for c in coverage.file_coverage if c.file_path == "existing.py"), None)
        assert cov is not None
        assert not cov.was_read_before_edit, "Edit without Read should still be flagged"

    def test_blind_spot_detection(self, test_repo_path, tmp_path):
        """Test synthetic blind spot detection."""
        # Create a simple synthetic setup in the temp repo to test logic specifically

        # File A imports B
        (test_repo_path / "A.py").write_text("import B")
        (test_repo_path / "B.py").write_text("# B logic")

        analyzer = ContextCoverageAnalyzer(test_repo_path)

        # Create a synthetic transcript: Edit A without Reading B
        transcript_file = tmp_path / "synthetic_transcript.jsonl"
        events = [
            {"tool_name": "Read", "tool_input": {"file_path": str(test_repo_path / "A.py")}},  # Read A
            {"tool_name": "Edit", "tool_input": {"file_path": str(test_repo_path / "A.py")}},  # Edit A
        ]

        with open(transcript_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        # NOTE: ContextCoverageAnalyzer uses _build_import_graph -> GitTracker -> git ls-files.
        # So we MUST add files to git for them to be seen.
        subprocess.run(["git", "add", "A.py", "B.py"], cwd=test_repo_path, check=True, capture_output=True)

        # Analyze
        coverage = analyzer.analyze_transcript(transcript_file)

        # B should be a blind spot because A imports it, but B was not read

        # Check coverage for A.py
        cov_a = next((c for c in coverage.file_coverage if c.file_path == "A.py"), None)
        assert cov_a is not None
        assert "B.py" in cov_a.imports
        assert "B.py" not in cov_a.imports_read

        assert "B.py" in coverage.blind_spots
