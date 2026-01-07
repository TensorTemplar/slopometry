"""Test database functionality."""

import tempfile
from datetime import datetime
from pathlib import Path

from slopometry.core.database import EventDatabase
from slopometry.core.models import LeaderboardEntry, UserStoryEntry


def test_user_story_export_functionality():
    """Test exporting user stories with existing or minimal test data."""
    db = EventDatabase()

    stats = db.get_user_story_stats()

    if stats["total_entries"] == 0:
        test_entry = UserStoryEntry(
            base_commit="test-base",
            head_commit="test-head",
            diff_content="diff --git a/test.py b/test.py\n+def hello():\n+    print('world')",
            user_stories="## Test User Stories\n\n1. As a tester, I want test functionality...",
            rating=3,
            guidelines_for_improving="Test guidelines",
            model_used="test-model",
            prompt_template="Test prompt template",
            repository_path=str(Path.cwd()),
        )
        db.save_user_story_entry(test_entry)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        count = db.export_user_stories(output_path)

        assert count >= 1, f"Expected at least 1 entry, got {count}"
        assert output_path.exists(), "Export file was not created"
        assert output_path.stat().st_size > 0, "Export file is empty"

        try:
            import pandas as pd

            df = pd.read_parquet(output_path)

            expected_columns = [
                "id",
                "created_at",
                "base_commit",
                "head_commit",
                "diff_content",
                "user_stories",
                "rating",
                "guidelines_for_improving",
                "model_used",
                "prompt_template",
                "repository_path",
            ]
            assert all(col in df.columns for col in expected_columns)
            assert len(df) >= 1

        except ImportError:
            pass

    finally:
        if output_path.exists():
            output_path.unlink()


def test_user_story_stats():
    """Test user story statistics calculation."""
    db = EventDatabase()

    stats = db.get_user_story_stats()

    assert stats["total_entries"] >= 0
    assert "avg_rating" in stats
    assert "unique_models" in stats
    assert "unique_repos" in stats
    assert "rating_distribution" in stats


def test_user_story_generation_cli_integration():
    """Test that the CLI command for generating user story entries works.

    Note: Does not run the actual command as it requires LLM access.
    """
    from click.testing import CliRunner

    from slopometry.cli import cli

    runner = CliRunner()

    result = runner.invoke(cli, ["summoner", "userstorify", "--help"])
    assert result.exit_code == 0
    assert "Generate user stories from commits using configured AI agents" in result.output
    assert "--base-commit" in result.output
    assert "--head-commit" in result.output


def test_leaderboard_upsert__updates_existing_project_on_new_commit():
    """Test that saving a leaderboard entry with same project_path but different commit updates the entry."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db = EventDatabase(db_path=Path(tmp_dir) / "test.db")

        project_path = "/test/project"
        entry_v1 = LeaderboardEntry(
            project_name="test-project",
            project_path=project_path,
            commit_sha_short="abc1234",
            commit_sha_full="abc1234567890",
            measured_at=datetime(2023, 1, 1),
            qpe_score=0.5,
            mi_normalized=0.6,
            smell_penalty=0.1,
            adjusted_quality=0.7,
            effort_factor=1.2,
            total_effort=1000.0,
            metrics_json="{}",
        )
        db.save_leaderboard_entry(entry_v1)

        leaderboard = db.get_leaderboard()
        assert len(leaderboard) == 1
        assert leaderboard[0].commit_sha_short == "abc1234"
        assert leaderboard[0].qpe_score == 0.5

        entry_v2 = LeaderboardEntry(
            project_name="test-project",
            project_path=project_path,
            commit_sha_short="def5678",
            commit_sha_full="def5678901234",
            measured_at=datetime(2024, 6, 1),
            qpe_score=0.8,
            mi_normalized=0.7,
            smell_penalty=0.05,
            adjusted_quality=0.85,
            effort_factor=1.1,
            total_effort=1200.0,
            metrics_json='{"updated": true}',
        )
        db.save_leaderboard_entry(entry_v2)

        leaderboard = db.get_leaderboard()
        assert len(leaderboard) == 1, "Should update existing entry, not create duplicate"
        assert leaderboard[0].commit_sha_short == "def5678"
        assert leaderboard[0].commit_sha_full == "def5678901234"
        assert leaderboard[0].qpe_score == 0.8
        assert leaderboard[0].measured_at == datetime(2024, 6, 1)
