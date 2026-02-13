"""Tests for data models."""

import pytest
from pydantic import ValidationError

from slopometry.core.display_models import (
    FeatureDisplayData,
    SessionDisplayData,
    UserStoryDisplayData,
)
from slopometry.core.models import (
    BaselineStrategy,
    ContextCoverage,
    ExtendedComplexityMetrics,
    FileCoverageStatus,
    ImplementationComparison,
    QPEScore,
    ResolvedBaselineStrategy,
    SmellAdvantage,
    UserStoryStatistics,
)


class TestExtendedComplexityMetrics:
    """Test the extended complexity metrics model."""

    def test_model_creation_without_required_fields__raises_validation_error(self) -> None:
        """Test that ValidationError is raised when required Halstead fields are missing."""
        with pytest.raises(ValidationError) as exc_info:
            ExtendedComplexityMetrics()  # pyrefly: ignore[missing-argument]

        errors = exc_info.value.errors()
        missing_fields = {e["loc"][0] for e in errors}
        # All Halstead and MI fields should be required
        assert "total_volume" in missing_fields
        assert "total_effort" in missing_fields
        assert "total_difficulty" in missing_fields
        assert "average_volume" in missing_fields
        assert "average_effort" in missing_fields
        assert "average_difficulty" in missing_fields
        assert "total_mi" in missing_fields
        assert "average_mi" in missing_fields

    def test_model_creation_with_values__creates_metrics_when_values_provided(self) -> None:
        """Test creating model with specific values when values provided."""
        metrics = ExtendedComplexityMetrics(
            total_complexity=150,
            total_volume=750.0,
            total_effort=5000.0,
            total_difficulty=10.5,
            average_volume=75.0,
            average_effort=500.0,
            average_difficulty=1.05,
            total_mi=800.0,
            average_mi=65.5,
            total_files_analyzed=10,
            files_by_complexity={"file1.py": 25, "file2.py": 30},
            files_by_effort={"file1.py": 2500.0, "file2.py": 2500.0},
            files_by_mi={"file1.py": 70.0, "file2.py": 60.0},
        )

        assert metrics.total_complexity == 150
        assert metrics.total_volume == 750.0
        assert metrics.total_difficulty == 10.5
        assert metrics.average_mi == 65.5
        assert metrics.total_files_analyzed == 10
        assert len(metrics.files_by_complexity) == 2
        assert len(metrics.files_by_effort) == 2
        assert len(metrics.files_by_mi) == 2
        assert metrics.files_by_mi["file1.py"] == 70.0


class TestUserStoryStatistics:
    """Test the user story statistics model."""

    def test_model_creation_with_values__creates_statistics_when_values_provided(self) -> None:
        """Test creating statistics model with specific values when values provided."""
        stats = UserStoryStatistics(
            total_entries=25,
            avg_rating=3.5,
            unique_models=3,
            unique_repos=2,
            rating_distribution={"3": 10, "4": 8, "5": 7},
        )

        assert stats.total_entries == 25
        assert stats.avg_rating == 3.5
        assert stats.unique_models == 3
        assert stats.unique_repos == 2
        assert stats.rating_distribution == {"3": 10, "4": 8, "5": 7}


class TestUserStoryDisplayData:
    """Test the user story display data model."""

    def test_model_creation__creates_display_data_when_values_provided(self) -> None:
        """Test creating display data model when values provided."""
        display_data = UserStoryDisplayData(
            entry_id="abc12345",
            date="2025-07-18 13:06",
            commits="9258f9b6→d3fd950d",
            rating="3/5",
            model="gemini-2.5-pro",
            repository="slopometry",
        )

        assert display_data.entry_id == "abc12345"
        assert display_data.date == "2025-07-18 13:06"
        assert display_data.commits == "9258f9b6→d3fd950d"
        assert display_data.rating == "3/5"
        assert display_data.model == "gemini-2.5-pro"
        assert display_data.repository == "slopometry"


def test_context_coverage_has_gaps__returns_false_when_perfect():
    """Test that has_gaps returns False when all coverage metrics are perfect."""
    coverage = ContextCoverage(
        files_edited=["src/foo.py"],
        files_read=["src/foo.py"],
        file_coverage=[
            FileCoverageStatus(
                file_path="src/foo.py",
                was_read_before_edit=True,
                imports_coverage=100.0,
                dependents_coverage=100.0,
            )
        ],
        blind_spots=[],
    )

    assert coverage.has_gaps is False


def test_context_coverage_has_gaps__returns_true_when_read_ratio_low():
    """Test that has_gaps returns True when files weren't read before edit."""
    coverage = ContextCoverage(
        files_edited=["src/foo.py"],
        files_read=[],
        file_coverage=[
            FileCoverageStatus(
                file_path="src/foo.py",
                was_read_before_edit=False,
                imports_coverage=100.0,
                dependents_coverage=100.0,
            )
        ],
        blind_spots=[],
    )

    assert coverage.has_gaps is True


def test_context_coverage_has_gaps__returns_true_when_blind_spots():
    """Test that has_gaps returns True when there are blind spots."""
    coverage = ContextCoverage(
        files_edited=["src/foo.py"],
        files_read=["src/foo.py"],
        file_coverage=[
            FileCoverageStatus(
                file_path="src/foo.py",
                was_read_before_edit=True,
                imports_coverage=100.0,
                dependents_coverage=100.0,
            )
        ],
        blind_spots=["src/bar.py"],
    )

    assert coverage.has_gaps is True


def test_resolved_baseline_strategy__rejects_auto_as_resolved() -> None:
    """Test that resolved strategy cannot be AUTO."""
    with pytest.raises(ValidationError, match="resolved strategy cannot be AUTO"):
        ResolvedBaselineStrategy(
            requested=BaselineStrategy.AUTO,
            resolved=BaselineStrategy.AUTO,
            merge_ratio=0.2,
            total_commits_sampled=100,
        )


def test_resolved_baseline_strategy__accepts_merge_anchored() -> None:
    """Test that MERGE_ANCHORED is accepted as resolved strategy."""
    strategy = ResolvedBaselineStrategy(
        requested=BaselineStrategy.AUTO,
        resolved=BaselineStrategy.MERGE_ANCHORED,
        merge_ratio=0.25,
        total_commits_sampled=200,
    )
    assert strategy.resolved == BaselineStrategy.MERGE_ANCHORED


def test_resolved_baseline_strategy__accepts_time_sampled() -> None:
    """Test that TIME_SAMPLED is accepted as resolved strategy."""
    strategy = ResolvedBaselineStrategy(
        requested=BaselineStrategy.AUTO,
        resolved=BaselineStrategy.TIME_SAMPLED,
        merge_ratio=0.05,
        total_commits_sampled=200,
    )
    assert strategy.resolved == BaselineStrategy.TIME_SAMPLED


def test_resolved_baseline_strategy__frozen_rejects_mutation() -> None:
    """Test that frozen model rejects field mutation."""
    strategy = ResolvedBaselineStrategy(
        requested=BaselineStrategy.AUTO,
        resolved=BaselineStrategy.MERGE_ANCHORED,
        merge_ratio=0.25,
        total_commits_sampled=200,
    )
    with pytest.raises(ValidationError):
        strategy.merge_ratio = 0.5  # pyrefly: ignore[read-only]


def test_resolved_baseline_strategy__round_trips_json() -> None:
    """Test JSON serialization round-trip."""
    strategy = ResolvedBaselineStrategy(
        requested=BaselineStrategy.AUTO,
        resolved=BaselineStrategy.MERGE_ANCHORED,
        merge_ratio=0.25,
        total_commits_sampled=200,
    )
    json_str = strategy.model_dump_json()
    restored = ResolvedBaselineStrategy.model_validate_json(json_str)
    assert restored == strategy


def test_smell_advantage__frozen_rejects_mutation() -> None:
    """Test that frozen model rejects field mutation."""
    sa = SmellAdvantage(
        smell_name="swallowed_exception",
        baseline_count=3,
        candidate_count=1,
        weight=0.15,
        weighted_delta=-0.30,
    )
    with pytest.raises(ValidationError):
        sa.weight = 0.5  # pyrefly: ignore[read-only]


def test_smell_advantage__stores_all_fields() -> None:
    """Test that all fields are stored correctly."""
    sa = SmellAdvantage(
        smell_name="hasattr_getattr",
        baseline_count=5,
        candidate_count=8,
        weight=0.10,
        weighted_delta=0.30,
    )
    assert sa.smell_name == "hasattr_getattr"
    assert sa.baseline_count == 5
    assert sa.candidate_count == 8
    assert sa.weight == 0.10
    assert sa.weighted_delta == 0.30


def test_implementation_comparison__stores_all_fields() -> None:
    """Test model creation with all fields."""
    qpe_a = QPEScore(qpe=0.5, mi_normalized=0.6, smell_penalty=0.1, adjusted_quality=0.5)
    qpe_b = QPEScore(qpe=0.7, mi_normalized=0.8, smell_penalty=0.05, adjusted_quality=0.7)

    comparison = ImplementationComparison(
        prefix_a="vendor/lib-a",
        prefix_b="vendor/lib-b",
        ref="HEAD",
        qpe_a=qpe_a,
        qpe_b=qpe_b,
        aggregate_advantage=0.35,
        smell_advantages=[],
        winner="vendor/lib-b",
    )
    assert comparison.prefix_a == "vendor/lib-a"
    assert comparison.winner == "vendor/lib-b"
    assert comparison.aggregate_advantage == 0.35


def test_implementation_comparison__round_trips_json() -> None:
    """Test JSON serialization round-trip."""
    qpe_a = QPEScore(qpe=0.5, mi_normalized=0.6, smell_penalty=0.1, adjusted_quality=0.5)
    qpe_b = QPEScore(qpe=0.7, mi_normalized=0.8, smell_penalty=0.05, adjusted_quality=0.7)

    comparison = ImplementationComparison(
        prefix_a="vendor/lib-a",
        prefix_b="vendor/lib-b",
        ref="main",
        qpe_a=qpe_a,
        qpe_b=qpe_b,
        aggregate_advantage=0.35,
        smell_advantages=[
            SmellAdvantage(
                smell_name="swallowed_exception",
                baseline_count=3,
                candidate_count=1,
                weight=0.15,
                weighted_delta=-0.30,
            )
        ],
        winner="vendor/lib-b",
    )
    json_str = comparison.model_dump_json()
    restored = ImplementationComparison.model_validate_json(json_str)
    assert restored.prefix_a == comparison.prefix_a
    assert len(restored.smell_advantages) == 1


class TestSessionDisplayData:
    """Test the session display data model."""

    def test_model_creation__creates_display_data_when_values_provided(self) -> None:
        """Test creating display data model when values provided."""
        display_data = SessionDisplayData(
            session_id="abc12345",
            start_time="2025-07-18 13:06",
            total_events=50,
            tools_used=12,
            project_name="my-project",
            project_source="git",
        )

        assert display_data.session_id == "abc12345"
        assert display_data.start_time == "2025-07-18 13:06"
        assert display_data.total_events == 50
        assert display_data.tools_used == 12
        assert display_data.project_name == "my-project"
        assert display_data.project_source == "git"

    def test_model_creation__handles_none_project(self) -> None:
        """Test that project_name and project_source can be None."""
        display_data = SessionDisplayData(
            session_id="abc12345",
            start_time="2025-07-18 13:06",
            total_events=10,
            tools_used=5,
            project_name=None,
            project_source=None,
        )

        assert display_data.project_name is None
        assert display_data.project_source is None

    def test_model_round_trip__serializes_and_deserializes(self) -> None:
        """Test JSON serialization round-trip."""
        display_data = SessionDisplayData(
            session_id="session-xyz",
            start_time="2025-01-01 12:00",
            total_events=25,
            tools_used=8,
            project_name="test-project",
            project_source="pyproject",
        )

        json_str = display_data.model_dump_json()
        restored = SessionDisplayData.model_validate_json(json_str)
        assert restored == display_data


class TestFeatureDisplayData:
    """Test the feature display data model."""

    def test_model_creation__creates_display_data_when_values_provided(self) -> None:
        """Test creating display data model when values provided."""
        display_data = FeatureDisplayData(
            feature_id="feat-001",
            feature_message="Add user authentication",
            commits_display="abc123 → def456",
            best_entry_id="entry-789",
            merge_message="feat: implement login system",
        )

        assert display_data.feature_id == "feat-001"
        assert display_data.feature_message == "Add user authentication"
        assert display_data.commits_display == "abc123 → def456"
        assert display_data.best_entry_id == "entry-789"
        assert display_data.merge_message == "feat: implement login system"

    def test_model_creation__handles_na_best_entry(self) -> None:
        """Test that best_entry_id can be 'N/A' when no user story exists."""
        display_data = FeatureDisplayData(
            feature_id="feat-002",
            feature_message="Refactor core module",
            commits_display="xyz123 → abc456",
            best_entry_id="N/A",
            merge_message="refactor: clean up codebase",
        )

        assert display_data.best_entry_id == "N/A"

    def test_model_round_trip__serializes_and_deserializes(self) -> None:
        """Test JSON serialization round-trip."""
        display_data = FeatureDisplayData(
            feature_id="feat-003",
            feature_message="New feature implementation",
            commits_display="000111 → 222333",
            best_entry_id="entry-555",
            merge_message="feat: implement something great",
        )

        json_str = display_data.model_dump_json()
        restored = FeatureDisplayData.model_validate_json(json_str)
        assert restored == display_data
