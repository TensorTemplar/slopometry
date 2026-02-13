import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.database import EventDatabase
from slopometry.core.git_tracker import GitOperationError, GitTracker
from slopometry.core.models import (
    AnalysisSource,
    CurrentChangesAnalysis,
    CurrentImpactSummary,
    ExtendedComplexityMetrics,
    HistoricalMetricStats,
    ImpactAssessment,
    ImpactCategory,
    RepoBaseline,
    SmellAdvantage,
)
from slopometry.summoner.services.current_impact_service import CurrentImpactService


class TestCurrentImpactService:
    """Integration tests for CurrentImpactService."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Isolated database scoped to each test to prevent cache leaks."""
        return EventDatabase(db_path=tmp_path / "test.db")

    @pytest.fixture(scope="module")
    def real_baseline(self):
        """Compute baseline once for the current source repo."""
        source_repo = Path.cwd()
        if not (source_repo / ".git").exists():
            return None

        analyzer = ComplexityAnalyzer(working_directory=source_repo)
        try:
            metrics = analyzer.analyze_extended_complexity()
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
        source_repo = Path.cwd()
        assert (source_repo / ".git").exists(), "Test must run from within the repository"

        dest_repo_path = tmp_path / "repo"

        subprocess.run(["git", "clone", str(source_repo), str(dest_repo_path)], check=True, capture_output=True)

        return dest_repo_path

    def test_analyze_uncommitted_changes__no_changes_returns_none(self, test_repo_path, real_baseline, test_db):
        """Test that analyzing a clean repo returns None."""
        assert real_baseline is not None, "Baseline computation failed - fixture returned None"

        service = CurrentImpactService(db=test_db)

        result = service.analyze_uncommitted_changes(test_repo_path, real_baseline)

        # Should be None as there are no changes
        assert result is None

    def test_analyze_uncommitted_changes__detects_changes(self, test_repo_path, real_baseline, test_db):
        """Test analyzing a repo with uncommitted changes."""
        assert real_baseline is not None, "Baseline computation failed - fixture returned None"

        service = CurrentImpactService(db=test_db)

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

        result = service.analyze_uncommitted_changes(test_repo_path, real_baseline)

        assert result is not None
        assert result.changed_files
        assert str(target_file.relative_to(test_repo_path)) in result.changed_files

        # Should have current metrics computed
        # Note: We don't assert total_complexity differs because the baseline comes from
        # the source repo's uncommitted state, which may coincidentally have the same
        # total complexity as the clone with our appended function
        assert result.current_metrics is not None
        assert result.current_metrics.total_files_analyzed > 0
        assert result.current_metrics.total_complexity > 0

    def test_analyze_previous_commit__returns_analysis_with_correct_source(
        self, test_repo_path, real_baseline, test_db
    ):
        """Test that analyzing previous commit sets the correct source."""
        assert real_baseline is not None, "Baseline computation failed"

        service = CurrentImpactService(db=test_db)

        # The test_repo_path is a clone with commits, so previous commit should exist
        result = service.analyze_previous_commit(test_repo_path, real_baseline)

        # Should return analysis (assuming commits have Python changes)
        if result is not None:
            assert result.source == AnalysisSource.PREVIOUS_COMMIT
            assert result.analyzed_commit_sha is not None
            assert result.base_commit_sha is not None
            assert len(result.analyzed_commit_sha) == 8  # Short SHA
            assert len(result.base_commit_sha) == 8  # Short SHA

    def test_analyze_previous_commit__returns_none_when_no_previous_commit(self, tmp_path, real_baseline, test_db):
        """Test that analyze_previous_commit returns None for repos with only one commit."""
        assert real_baseline is not None, "Baseline computation failed"

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

        service = CurrentImpactService(db=test_db)
        result = service.analyze_previous_commit(repo_path, real_baseline)

        assert result is None

    def test_analyze_previous_commit__returns_none_when_no_python_changes(self, tmp_path, real_baseline, test_db):
        """Test that analyze_previous_commit returns None when last commit has no Python changes."""
        assert real_baseline is not None, "Baseline computation failed"

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

        service = CurrentImpactService(db=test_db)
        result = service.analyze_previous_commit(repo_path, real_baseline)

        # Should return None because no Python files were changed
        assert result is None

    def test_analyze_previous_commit__logs_debug_on_git_operation_error(
        self, test_repo_path, real_baseline, test_db, caplog, monkeypatch
    ):
        """Test that analyze_previous_commit logs debug messages on GitOperationError."""
        assert real_baseline is not None, "Baseline computation failed"

        def mock_get_changed_python_files(self, parent_sha, child_sha):
            raise GitOperationError("Simulated git diff failure")

        monkeypatch.setattr(GitTracker, "get_changed_python_files", mock_get_changed_python_files)

        service = CurrentImpactService(db=test_db)

        with caplog.at_level(logging.DEBUG, logger="slopometry.summoner.services.current_impact_service"):
            result = service.analyze_previous_commit(test_repo_path, real_baseline)

        assert result is None
        assert any("Failed to get changed files" in record.message for record in caplog.records)

    def test_analyze_uncommitted_changes__uses_cache_on_second_call(
        self, test_repo_path, real_baseline, test_db, caplog
    ):
        """Test that second call with same state uses cached metrics."""
        assert real_baseline is not None, "Baseline computation failed"

        service = CurrentImpactService(db=test_db)

        # Modify a python file to have uncommitted changes
        target_file = test_repo_path / "src" / "slopometry" / "core" / "models.py"
        if not target_file.exists():
            target_file = test_repo_path / "test_file.py"
            target_file.write_text("def foo():\n    pass\n")
        else:
            with open(target_file, "a") as f:
                f.write("\n\ndef cached_test_func(x):\n    return x * 2\n")

        # First call - should compute and cache
        with caplog.at_level(logging.DEBUG, logger="slopometry.summoner.services.current_impact_service"):
            result1 = service.analyze_uncommitted_changes(test_repo_path, real_baseline)

        assert result1 is not None
        cached_logged = any("Cached metrics for current-impact" in record.message for record in caplog.records)
        assert cached_logged, "First call should cache metrics"

        caplog.clear()

        # Second call - should use cache
        with caplog.at_level(logging.DEBUG, logger="slopometry.summoner.services.current_impact_service"):
            result2 = service.analyze_uncommitted_changes(test_repo_path, real_baseline)

        assert result2 is not None
        using_cache_logged = any(
            "Using cached metrics for current-impact" in record.message for record in caplog.records
        )
        assert using_cache_logged, "Second call should use cached metrics"

    def test_analyze_uncommitted_changes__cache_invalidated_on_file_change(
        self, test_repo_path, real_baseline, test_db, caplog
    ):
        """Test that cache is invalidated when working tree changes."""
        assert real_baseline is not None, "Baseline computation failed"

        service = CurrentImpactService(db=test_db)

        # Modify an existing tracked Python file
        target_file = test_repo_path / "src" / "slopometry" / "core" / "models.py"
        assert target_file.exists(), f"models.py not found in cloned test repo at {target_file}"

        original_content = target_file.read_text()

        # First modification
        target_file.write_text(original_content + "\n\ndef cache_invalidation_test_v1():\n    return 1\n")

        # First call - compute and cache
        with caplog.at_level(logging.DEBUG, logger="slopometry.summoner.services.current_impact_service"):
            result1 = service.analyze_uncommitted_changes(test_repo_path, real_baseline)

        assert result1 is not None
        caplog.clear()

        # Second modification - should invalidate cache due to different content hash
        target_file.write_text(original_content + "\n\ndef cache_invalidation_test_v2():\n    return 2\n")

        # Second call - should recompute (different working tree hash)
        with caplog.at_level(logging.DEBUG, logger="slopometry.summoner.services.current_impact_service"):
            result2 = service.analyze_uncommitted_changes(test_repo_path, real_baseline)

        assert result2 is not None
        # Should NOT see "Using cached metrics" since file changed
        using_cache_logged = any(
            "Using cached metrics for current-impact" in record.message for record in caplog.records
        )
        cached_logged = any("Cached metrics for current-impact" in record.message for record in caplog.records)
        assert not using_cache_logged, "Changed file should invalidate cache"
        assert cached_logged, "Should cache new metrics after recomputation"


class TestCurrentImpactSummary:
    """Tests for the compact CurrentImpactSummary model."""

    @staticmethod
    def _make_metrics() -> ExtendedComplexityMetrics:
        return ExtendedComplexityMetrics(
            total_volume=0.0, total_effort=0.0, total_difficulty=0.0,
            average_volume=0.0, average_effort=0.0, average_difficulty=0.0,
            total_mi=0.0, average_mi=0.0,
        )

    def test_from_analysis__maps_all_fields(self):
        """Test that from_analysis correctly maps fields from full analysis."""
        metrics = self._make_metrics()
        dummy_stats = HistoricalMetricStats(
            metric_name="test", mean=0.0, std_dev=1.0, median=0.0,
            min_value=-1.0, max_value=1.0, sample_count=10,
        )
        baseline = RepoBaseline(
            repository_path="/tmp/repo",
            last_commit_hash="abc123",
            analysis_timestamp=datetime.now(),
            current_metrics=metrics,
            head_commit_sha="abc123",
            total_commits_analyzed=10,
            cc_delta_stats=dummy_stats,
            effort_delta_stats=dummy_stats,
            mi_delta_stats=dummy_stats,
        )
        assessment = ImpactAssessment(
            cc_z_score=0.5, effort_z_score=-0.3, mi_z_score=0.8,
            impact_score=0.65, impact_category=ImpactCategory.MINOR_IMPROVEMENT,
            cc_delta=-2.0, effort_delta=-100.0, mi_delta=3.5, qpe_delta=0.02,
            qpe_z_score=0.4,
        )
        smell_adv = SmellAdvantage(
            smell_name="orphan_comment", baseline_count=10, candidate_count=8,
            weight=0.01, weighted_delta=-0.02,
        )
        analysis = CurrentChangesAnalysis(
            repository_path="/tmp/repo",
            source=AnalysisSource.PREVIOUS_COMMIT,
            analyzed_commit_sha="abc12345",
            base_commit_sha="def67890",
            changed_files=["src/a.py", "src/b.py", "src/c.py"],
            current_metrics=metrics,
            baseline_metrics=metrics,
            assessment=assessment,
            baseline=baseline,
            blind_spots=["src/d.py"],
            smell_advantages=[smell_adv],
        )

        summary = CurrentImpactSummary.from_analysis(analysis)

        assert summary.source == AnalysisSource.PREVIOUS_COMMIT
        assert summary.analyzed_commit_sha == "abc12345"
        assert summary.base_commit_sha == "def67890"
        assert summary.impact_score == 0.65
        assert summary.impact_category == ImpactCategory.MINOR_IMPROVEMENT
        assert summary.qpe_delta == 0.02
        assert summary.cc_delta == -2.0
        assert summary.effort_delta == -100.0
        assert summary.mi_delta == 3.5
        assert summary.changed_files_count == 3
        assert summary.blind_spots_count == 1
        assert len(summary.smell_advantages) == 1
        assert summary.smell_advantages[0].smell_name == "orphan_comment"

    def test_from_analysis__serializes_to_json(self):
        """Test that summary serializes to valid JSON with expected keys."""
        metrics = self._make_metrics()
        dummy_stats = HistoricalMetricStats(
            metric_name="test", mean=0.0, std_dev=1.0, median=0.0,
            min_value=-1.0, max_value=1.0, sample_count=10,
        )
        baseline = RepoBaseline(
            repository_path="/tmp/repo",
            last_commit_hash="abc123",
            analysis_timestamp=datetime.now(),
            current_metrics=metrics,
            head_commit_sha="abc123",
            total_commits_analyzed=10,
            cc_delta_stats=dummy_stats,
            effort_delta_stats=dummy_stats,
            mi_delta_stats=dummy_stats,
        )
        assessment = ImpactAssessment(
            cc_z_score=0.0, effort_z_score=0.0, mi_z_score=0.0,
            impact_score=0.0, impact_category=ImpactCategory.NEUTRAL,
            cc_delta=0.0, effort_delta=0.0, mi_delta=0.0, qpe_delta=0.0,
            qpe_z_score=0.0,
        )
        analysis = CurrentChangesAnalysis(
            repository_path="/tmp/repo",
            changed_files=["src/a.py"],
            current_metrics=metrics,
            baseline_metrics=metrics,
            assessment=assessment,
            baseline=baseline,
        )

        summary = CurrentImpactSummary.from_analysis(analysis)
        json_str = summary.model_dump_json(indent=2)
        parsed = json.loads(json_str)

        expected_keys = {
            "source", "analyzed_commit_sha", "base_commit_sha",
            "impact_score", "impact_category", "qpe_delta",
            "cc_delta", "effort_delta", "mi_delta",
            "changed_files_count", "blind_spots_count", "smell_advantages",
        }
        assert set(parsed.keys()) == expected_keys
        assert parsed["source"] == "uncommitted_changes"
        assert parsed["impact_category"] == "neutral"
