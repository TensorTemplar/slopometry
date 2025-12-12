"""Tests for data models."""

import pytest
from pydantic import ValidationError

from slopometry.core.models import ExtendedComplexityMetrics, UserStoryDisplayData, UserStoryStatistics


class TestExtendedComplexityMetrics:
    """Test the extended complexity metrics model."""

    def test_model_creation_without_required_fields__raises_validation_error(self):
        """Test that ValidationError is raised when required Halstead fields are missing."""
        with pytest.raises(ValidationError) as exc_info:
            ExtendedComplexityMetrics()

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

    def test_model_creation_with_values__creates_metrics_when_values_provided(self):
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
        )

        assert metrics.total_complexity == 150
        assert metrics.total_volume == 750.0
        assert metrics.total_difficulty == 10.5
        assert metrics.average_mi == 65.5
        assert metrics.total_files_analyzed == 10
        assert len(metrics.files_by_complexity) == 2


class TestUserStoryStatistics:
    """Test the user story statistics model."""

    def test_model_creation_with_values__creates_statistics_when_values_provided(self):
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

    def test_model_creation__creates_display_data_when_values_provided(self):
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
