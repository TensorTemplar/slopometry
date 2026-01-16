import logging
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.models import AnalysisSource, HistoricalMetricStats, RepoBaseline
from slopometry.summoner.services.current_impact_service import CurrentImpactService


class TestCurrentImpactService:
    """Integration tests for CurrentImpactService."""

    @pytest.fixture(scope="module")
    def real_baseline(self):
        """Compute baseline once for the current source repo."""
        source_repo = Path.cwd()
        if not (source_repo / ".git").exists():
            return None

        analyzer = ComplexityAnalyzer(working_directory=source_repo)
        try:
            metrics = analyzer.analyze_extended_complexity()
            # Create dummy stats for required fields
            dummy_stats = HistoricalMetricStats(
                metric_name="test_metric",
                mean=0.0,
                std_dev=0.0,
                median=0.0,
                min_value=0.0,
                max_value=0.0,
                sample_count=1,
            )

            return RepoBaseline(
                repository_path=str(source_repo),
                last_commit_hash="HEAD",
                analysis_timestamp=datetime.now(),
                current_metrics=metrics,
                head_commit_sha="HEAD",
                total_commits_analyzed=1,
                cc_delta_stats=dummy_stats,
                effort_delta_stats=dummy_stats,
                mi_delta_stats=dummy_stats,
            )
        except Exception:
            return None

    @pytest.fixture
    def test_repo_path(self, tmp_path):
        """Create a temporary clone of the current repository."""
        # Use the actual current repo as source
        source_repo = Path.cwd()
        assert (source_repo / ".git").exists(), "Test must run from within the repository"

        dest_repo_path = tmp_path / "repo"

        # Clone using git command to ensuring clean state
        subprocess.run(["git", "clone", str(source_repo), str(dest_repo_path)], check=True, capture_output=True)

        return dest_repo_path

    def test_analyze_uncommitted_changes__no_changes_returns_none(self, test_repo_path, real_baseline):
        """Test that analyzing a clean repo returns None."""
        assert real_baseline is not None, "Baseline computation failed - fixture returned None"

        # Setup
        service = CurrentImpactService()

        # Use valid baseline from source repo (path differs but complexity is same)
        # We need to patch the baseline repository_path to match test_repo_path
        # or mock the baseline checking if it validates path.
        # But analyze_uncommitted_changes only uses baseline.current_metrics.

        # Mock baseline service to return our pre-computed baseline

        # Analyze
        result = service.analyze_uncommitted_changes(test_repo_path, real_baseline)

        # Should be None as there are no changes
        assert result is None

    def test_analyze_uncommitted_changes__detects_changes(self, test_repo_path, real_baseline):
        """Test analyzing a repo with uncommitted changes."""
        assert real_baseline is not None, "Baseline computation failed - fixture returned None"

        service = CurrentImpactService()

        # Modify a python file
        target_file = test_repo_path / "src" / "slopometry" / "core" / "models.py"
        if not target_file.exists():
            # Fallback if structure changes, just make a new file
            target_file = test_repo_path / "test_file.py"
            target_file.write_text("def foo():\n    pass\n")
        else:
            # Append a complex function to increase complexity
            with open(target_file, "a") as f:
                f.write("\n\ndef complex_function(x):\n    if x > 0:\n        return x\n    else:\n        return -x\n")

        # Analyze
        result = service.analyze_uncommitted_changes(test_repo_path, real_baseline)

        assert result is not None
        assert result.changed_files
        assert str(target_file.relative_to(test_repo_path)) in result.changed_files

        # Should have some delta
        # Since we modified the file, current complexity should differ from baseline
        assert result.current_metrics.total_complexity != real_baseline.current_metrics.total_complexity

    def test_analyze_previous_commit__returns_analysis_with_correct_source(self, test_repo_path, real_baseline):
        """Test that analyzing previous commit sets the correct source."""
        assert real_baseline is not None, "Baseline computation failed"

        service = CurrentImpactService()

        # The test_repo_path is a clone with commits, so previous commit should exist
        result = service.analyze_previous_commit(test_repo_path, real_baseline)

        # Should return analysis (assuming commits have Python changes)
        if result is not None:
            assert result.source == AnalysisSource.PREVIOUS_COMMIT
            assert result.analyzed_commit_sha is not None
            assert result.base_commit_sha is not None
            assert len(result.analyzed_commit_sha) == 8  # Short SHA
            assert len(result.base_commit_sha) == 8  # Short SHA

    def test_analyze_previous_commit__returns_none_when_no_previous_commit(self, tmp_path, real_baseline):
        """Test that analyze_previous_commit returns None for repos with only one commit."""
        assert real_baseline is not None, "Baseline computation failed"

        # Create a repo with only one commit
        repo_path = tmp_path / "single_commit_repo"
        repo_path.mkdir()

        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "commit.gpgsign", "false"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        test_file = repo_path / "test.py"
        test_file.write_text("def foo(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        service = CurrentImpactService()
        result = service.analyze_previous_commit(repo_path, real_baseline)

        assert result is None

    def test_analyze_previous_commit__returns_none_when_no_python_changes(self, tmp_path, real_baseline):
        """Test that analyze_previous_commit returns None when last commit has no Python changes."""
        assert real_baseline is not None, "Baseline computation failed"

        # Create a repo with two commits, but the last one has no Python changes
        repo_path = tmp_path / "no_python_changes_repo"
        repo_path.mkdir()

        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "commit.gpgsign", "false"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # First commit with Python file
        test_file = repo_path / "test.py"
        test_file.write_text("def foo(): pass\n")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        # Second commit with only a text file (no Python)
        txt_file = repo_path / "readme.txt"
        txt_file.write_text("Just a text file\n")
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add readme"],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        service = CurrentImpactService()
        result = service.analyze_previous_commit(repo_path, real_baseline)

        # Should return None because no Python files were changed
        assert result is None

    def test_analyze_previous_commit__logs_debug_on_git_operation_error(
        self, test_repo_path, real_baseline, caplog, monkeypatch
    ):
        """Test that analyze_previous_commit logs debug messages on GitOperationError."""
        assert real_baseline is not None, "Baseline computation failed"

        from slopometry.core.git_tracker import GitOperationError, GitTracker

        def mock_get_changed_python_files(self, parent_sha, child_sha):
            raise GitOperationError("Simulated git diff failure")

        monkeypatch.setattr(GitTracker, "get_changed_python_files", mock_get_changed_python_files)

        service = CurrentImpactService()

        with caplog.at_level(logging.DEBUG, logger="slopometry.summoner.services.current_impact_service"):
            result = service.analyze_previous_commit(test_repo_path, real_baseline)

        assert result is None
        assert any("Failed to get changed files" in record.message for record in caplog.records)
