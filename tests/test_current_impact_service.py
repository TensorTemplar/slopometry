import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.models import HistoricalMetricStats, RepoBaseline
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
        if not (source_repo / ".git").exists():
            pytest.skip("Must run from within the repository")

        dest_repo_path = tmp_path / "repo"

        # Clone using git command to ensuring clean state
        subprocess.run(["git", "clone", str(source_repo), str(dest_repo_path)], check=True, capture_output=True)

        return dest_repo_path

    def test_analyze_uncommitted_changes__no_changes_returns_none(self, test_repo_path, real_baseline):
        """Test that analyzing a clean repo returns None."""
        if not real_baseline:
            pytest.skip("Could not compute baseline")

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
        if not real_baseline:
            pytest.skip("Could not compute baseline")

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
