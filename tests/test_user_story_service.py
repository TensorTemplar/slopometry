"""Tests for UserStoryService."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from slopometry.core.models.user_story import UserStoryDisplayData, UserStoryEntry, UserStoryStatistics
from slopometry.summoner.services.user_story_service import UserStoryService


class TestUserStoryService:
    """Test the UserStoryService class."""

    def test_get_user_story_statistics__returns_stats_when_database_has_data(self):
        """Test that statistics are returned when database has data."""
        # Mock database with sample data
        mock_db = Mock()
        mock_db.get_user_story_stats.return_value = {
            "total_entries": 10,
            "avg_rating": 3.5,
            "unique_models": 2,
            "unique_repos": 1,
            "rating_distribution": {"3": 5, "4": 3, "5": 2},
        }

        service = UserStoryService(db=mock_db)
        stats = service.get_user_story_statistics()

        assert isinstance(stats, UserStoryStatistics)
        assert stats.total_entries == 10
        assert stats.avg_rating == 3.5
        assert stats.unique_models == 2
        assert stats.unique_repos == 1
        assert stats.rating_distribution == {"3": 5, "4": 3, "5": 2}

    def test_get_user_story_statistics__returns_empty_stats_when_database_fails(self):
        """Test that empty statistics are returned when database fails.

        This fallback is reasonable because:
        - Database might be corrupted, locked, or have permission issues
        - User should see empty stats rather than a crash
        - This allows the application to continue functioning
        """
        # Mock database that raises an exception
        mock_db = Mock()
        mock_db.get_user_story_stats.side_effect = Exception("Database connection failed")

        service = UserStoryService(db=mock_db)
        stats = service.get_user_story_statistics()

        assert isinstance(stats, UserStoryStatistics)
        assert stats.total_entries == 0
        assert stats.avg_rating == 0
        assert stats.unique_models == 0
        assert stats.unique_repos == 0
        assert stats.rating_distribution == {}

    def test_get_user_story_entries__returns_entries_when_database_has_data(self):
        """Test that entries are returned when database has data."""
        # Create sample entries
        sample_entries = [
            UserStoryEntry(
                id="test-id-1",
                base_commit="abc123",
                head_commit="def456",
                diff_content="sample diff",
                user_stories="sample stories",
                rating=3,
                repository_path="/test/repo",
            ),
            UserStoryEntry(
                id="test-id-2",
                base_commit="ghi789",
                head_commit="jkl012",
                diff_content="another diff",
                user_stories="more stories",
                rating=4,
                repository_path="/test/repo",
            ),
        ]

        mock_db = Mock()
        mock_db.get_user_story_entries.return_value = sample_entries

        service = UserStoryService(db=mock_db)
        entries = service.get_user_story_entries(limit=10)

        assert len(entries) == 2
        assert all(isinstance(entry, UserStoryEntry) for entry in entries)
        assert entries[0].id == "test-id-1"
        assert entries[1].id == "test-id-2"

    def test_get_user_story_entries__returns_empty_list_when_database_fails(self):
        """Test that empty list is returned when database fails.

        This fallback is reasonable because:
        - Database might be corrupted, locked, or have permission issues
        - Empty list is a sensible default that won't break downstream code
        - User can still use other features of the application
        """
        # Mock database that raises an exception
        mock_db = Mock()
        mock_db.get_user_story_entries.side_effect = Exception("Database query failed")

        service = UserStoryService(db=mock_db)
        entries = service.get_user_story_entries(limit=10)

        assert entries == []

    def test_prepare_entries_data_for_display__formats_entries_correctly_when_given_valid_entries(self):
        """Test that entries are properly formatted for display when given valid entries."""
        from datetime import datetime

        # Create sample entries with known data
        entries = [
            UserStoryEntry(
                id="abcd1234-5678-9abc-def0-123456789abc",
                created_at=datetime(2025, 7, 18, 13, 6, 39),
                base_commit="9258f9b6abc123def456",
                head_commit="d3fd950d789abc012345",
                diff_content="sample diff",
                user_stories="sample stories",
                rating=3,
                model_used="gemini-2.5-pro",
                repository_path="/home/user/projects/slopometry",
            )
        ]

        service = UserStoryService(db=Mock())
        display_data = service.prepare_entries_data_for_display(entries)

        assert len(display_data) == 1
        assert isinstance(display_data[0], UserStoryDisplayData)

        data = display_data[0]
        assert data.entry_id == "abcd1234"  # short_id
        assert data.date == "2025-07-18 13:06"
        assert data.commits == "9258f9b6→d3fd950d"
        assert data.rating == "3/5"  # default rating
        assert data.model == "gemini-2.5-pro"
        assert data.repository == "slopometry"  # just the name

    def test_export_user_stories__delegates_to_database_when_called(self):
        """Test that export delegates to database when called."""
        mock_db = Mock()
        mock_db.export_user_stories.return_value = 5

        service = UserStoryService(db=mock_db)
        output_path = Path("/tmp/test.parquet")

        count = service.export_user_stories(output_path)

        assert count == 5
        mock_db.export_user_stories.assert_called_once_with(output_path)

    def test_export_user_stories__reraises_import_error_with_helpful_message(self):
        """Test that ImportError is re-raised with helpful message.

        This behavior is reasonable because:
        - ImportError means required dependencies are missing
        - The original error might not be clear about which dependencies
        - Re-raising with explicit dependency info helps user fix the issue
        """
        mock_db = Mock()
        mock_db.export_user_stories.side_effect = ImportError("No module named 'pandas'")

        service = UserStoryService(db=mock_db)
        output_path = Path("/tmp/test.parquet")

        with pytest.raises(ImportError) as exc_info:
            service.export_user_stories(output_path)

        assert "Missing required dependencies for export" in str(exc_info.value)
        assert "No module named 'pandas'" in str(exc_info.value)

    def test_upload_to_huggingface__uses_default_repo_when_settings_provided(self):
        """Test that upload uses default repo when settings provided."""
        with patch("slopometry.summoner.services.user_story_service.settings") as mock_settings:
            mock_settings.hf_default_repo = "user/test-repo"

            # Mock the import and function
            mock_upload_func = Mock()
            with patch("slopometry.summoner.services.hf_uploader.upload_to_huggingface", mock_upload_func):
                service = UserStoryService(db=Mock())
                output_path = Path("/tmp/test.parquet")

                result = service.upload_to_huggingface(output_path)

                assert result == "user/test-repo"
                mock_upload_func.assert_called_once_with(output_path, "user/test-repo")

    def test_upload_to_huggingface__generates_repo_name_from_current_directory_when_no_settings_provided(self):
        """Test that upload generates repo name from current directory when no settings provided."""
        with patch("slopometry.summoner.services.user_story_service.settings") as mock_settings:
            mock_settings.hf_default_repo = None

            with patch("slopometry.summoner.services.user_story_service.Path") as mock_path:
                mock_path.cwd.return_value.name = "My_Project Name!"

                # Mock the import and function
                mock_upload_func = Mock()
                with patch("slopometry.summoner.services.hf_uploader.upload_to_huggingface", mock_upload_func):
                    service = UserStoryService(db=Mock())
                    output_path = Path("/tmp/test.parquet")

                    result = service.upload_to_huggingface(output_path)

                    # Should clean up the name and add prefix
                    assert result == "slopometry-my-project-name-userstories"
                    mock_upload_func.assert_called_once_with(output_path, "slopometry-my-project-name-userstories")

    def test_upload_to_huggingface__reraises_import_error_with_helpful_message(self):
        """Test that ImportError is re-raised with helpful dependency info.

        This behavior is reasonable because:
        - ImportError means HuggingFace dependencies are missing
        - The error message tells user exactly what to install
        - This is better than a generic import error
        """
        service = UserStoryService(db=Mock())
        output_path = Path("/tmp/test.parquet")

        # Mock the __import__ to fail when trying to import hf_uploader
        with patch("builtins.__import__", side_effect=ImportError("No module named 'datasets'")):
            with pytest.raises(ImportError) as exc_info:
                service.upload_to_huggingface(output_path)

            error_msg = str(exc_info.value)
            assert "Hugging Face datasets library not installed" in error_msg
            assert "pip install datasets huggingface-hub pandas pyarrow" in error_msg

    def test_filter_entries_for_rating__filters_by_model_when_model_specified(self):
        """Test that entries are filtered by model when model specified."""
        entries = [
            UserStoryEntry(
                id="1",
                base_commit="abc",
                head_commit="def",
                diff_content="diff1",
                user_stories="stories1",
                model_used="gpt-4",
                rating=3,
            ),
            UserStoryEntry(
                id="2",
                base_commit="ghi",
                head_commit="jkl",
                diff_content="diff2",
                user_stories="stories2",
                model_used="claude-3",
                rating=4,
            ),
            UserStoryEntry(
                id="3",
                base_commit="mno",
                head_commit="pqr",
                diff_content="diff3",
                user_stories="stories3",
                model_used="gpt-4",
                rating=5,
            ),
        ]

        mock_db = Mock()
        service = UserStoryService(db=mock_db)
        service.get_user_story_entries = Mock(return_value=entries)

        filtered = service.filter_entries_for_rating(limit=10, filter_model="gpt-4")

        assert len(filtered) == 2
        assert all(entry.model_used == "gpt-4" for entry in filtered)
        assert filtered[0].id == "1"
        assert filtered[1].id == "3"

    def test_filter_entries_for_rating__filters_unrated_only_when_flag_enabled(self):
        """Test that entries are filtered to unrated only when flag enabled."""
        entries = [
            UserStoryEntry(
                id="1",
                base_commit="abc",
                head_commit="def",
                diff_content="diff1",
                user_stories="stories1",
                model_used="gpt-4",
                rating=3,
            ),
            UserStoryEntry(
                id="2",
                base_commit="ghi",
                head_commit="jkl",
                diff_content="diff2",
                user_stories="stories2",
                model_used="claude-3",
                rating=4,
            ),
            UserStoryEntry(
                id="3",
                base_commit="mno",
                head_commit="pqr",
                diff_content="diff3",
                user_stories="stories3",
                model_used="gpt-4",
                rating=3,
            ),
        ]

        mock_db = Mock()
        service = UserStoryService(db=mock_db)
        service.get_user_story_entries = Mock(return_value=entries)

        filtered = service.filter_entries_for_rating(limit=10, unrated_only=True)

        assert len(filtered) == 2
        assert all(entry.rating == 3 for entry in filtered)
        assert filtered[0].id == "1"
        assert filtered[1].id == "3"

    def test_filter_entries_for_rating__applies_limit_after_filtering_when_limit_specified(self):
        """Test that limit is applied after filtering when limit specified."""
        entries = [
            UserStoryEntry(
                id="1",
                base_commit="abc",
                head_commit="def",
                diff_content="diff1",
                user_stories="stories1",
                model_used="gpt-4",
                rating=3,
            ),
            UserStoryEntry(
                id="2",
                base_commit="ghi",
                head_commit="jkl",
                diff_content="diff2",
                user_stories="stories2",
                model_used="claude-3",
                rating=3,
            ),
            UserStoryEntry(
                id="3",
                base_commit="mno",
                head_commit="pqr",
                diff_content="diff3",
                user_stories="stories3",
                model_used="gpt-4",
                rating=3,
            ),
            UserStoryEntry(
                id="4",
                base_commit="stu",
                head_commit="vwx",
                diff_content="diff4",
                user_stories="stories4",
                model_used="gpt-4",
                rating=3,
            ),
        ]

        mock_db = Mock()
        service = UserStoryService(db=mock_db)
        service.get_user_story_entries = Mock(return_value=entries)

        filtered = service.filter_entries_for_rating(limit=2, filter_model="gpt-4")

        assert len(filtered) == 2
        assert all(entry.model_used == "gpt-4" for entry in filtered)
        assert filtered[0].id == "1"
        assert filtered[1].id == "3"

    def test_collect_user_rating_and_feedback__returns_rating_and_guidelines_when_user_provides_input(self):
        """Test that rating and guidelines are returned when user provides input."""
        with patch("slopometry.summoner.services.user_story_service.click.prompt") as mock_prompt:
            mock_prompt.side_effect = [4, "Please add more detail to user stories"]

            service = UserStoryService(db=Mock())
            rating, guidelines = service.collect_user_rating_and_feedback()

            assert rating == 4
            assert guidelines == "Please add more detail to user stories"

    def test_rate_user_story_entry__updates_entry_and_saves_to_database_when_called(self):
        """Test that entry is updated and saved to database when called."""
        mock_db = Mock()
        service = UserStoryService(db=mock_db)

        entry = UserStoryEntry(
            id="test",
            base_commit="abc",
            head_commit="def",
            diff_content="diff",
            user_stories="stories",
            rating=3,
            guidelines_for_improving="",
        )

        service.rate_user_story_entry(entry, 5, "Great work!")

        assert entry.rating == 5
        assert entry.guidelines_for_improving == "Great work!"
        mock_db.save_user_story_entry.assert_called_once_with(entry)

    def test_save_user_story_entry__delegates_to_database_when_called(self):
        """Test that save delegates to database when called."""
        mock_db = Mock()
        service = UserStoryService(db=mock_db)

        entry = UserStoryEntry(
            id="test", base_commit="abc", head_commit="def", diff_content="diff", user_stories="stories", rating=3
        )
        service.save_user_story_entry(entry)

        mock_db.save_user_story_entry.assert_called_once_with(entry)
