"""Test dataset export functionality."""

import tempfile
from pathlib import Path

from slopometry.core.database import EventDatabase
from slopometry.core.models import DiffUserStoryDataset


def test_dataset_export_functionality():
    """Test exporting dataset with existing or minimal test data."""
    db = EventDatabase()

    # Check if we have existing dataset entries
    stats = db.get_dataset_stats()

    if stats["total_entries"] == 0:
        # Create a minimal test entry if none exist
        test_entry = DiffUserStoryDataset(
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
        db.save_dataset_entry(test_entry)

    # Test export functionality
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        output_path = Path(tmp.name)

    try:
        # Export the dataset
        count = db.export_dataset(output_path)

        # Verify export worked
        assert count >= 1, f"Expected at least 1 entry, got {count}"
        assert output_path.exists(), "Export file was not created"
        assert output_path.stat().st_size > 0, "Export file is empty"

        # Verify we can read it back with pandas if available
        try:
            import pandas as pd

            df = pd.read_parquet(output_path)

            # Check structure
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

            # Check we have data
            assert len(df) >= 1

        except ImportError:
            # If pandas not available, just check file exists
            pass

    finally:
        # Cleanup
        if output_path.exists():
            output_path.unlink()


def test_dataset_stats():
    """Test dataset statistics calculation."""
    db = EventDatabase()

    # Get stats (should have entries from previous test or real usage)
    stats = db.get_dataset_stats()

    assert stats["total_entries"] >= 0
    assert "avg_rating" in stats
    assert "unique_models" in stats
    assert "unique_repos" in stats
    assert "rating_distribution" in stats


def test_dataset_generation_cli_integration():
    """Test that the CLI command for generating dataset entries works."""
    from click.testing import CliRunner

    from slopometry.cli import cli

    runner = CliRunner()

    # Test the userstorify command help (now under summoner subcommand)
    result = runner.invoke(cli, ["summoner", "userstorify", "--help"])
    assert result.exit_code == 0
    assert "Generate user stories from commits using configured AI agents" in result.output
    assert "--base-commit" in result.output
    assert "--head-commit" in result.output

    # Note: We don't run the actual command here as it requires LLM access
    # and would be slow/expensive. The command is tested manually.
