"""Tests for summoner/cli/commands.py."""

import os
import subprocess
from pathlib import Path

from click.testing import CliRunner

from slopometry.summoner.cli.commands import summoner


def test_analyze_commits__exits_cleanly_when_not_a_git_repo(tmp_path: Path) -> None:
    """Test that analyze-commits exits cleanly when path is not a git repo (no Python detected)."""
    runner = CliRunner()

    # Run against a plain temp directory - language guard will detect no Python files
    result = runner.invoke(summoner, ["analyze-commits", "--repo-path", str(tmp_path)])

    # Language guard exits cleanly (exit code 0) when no Python files detected
    assert result.exit_code == 0
    assert "requires Python files" in result.output


def test_analyze_commits__fails_gracefully_when_insufficient_commits(tmp_path: Path) -> None:
    """Test that analyze-commits fails when default limit (HEAD~10) is not reachable."""
    runner = CliRunner()

    # Initialize a git repo with just 1 commit
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    subprocess.run(["git", "init"], cwd=tmp_path, env=env, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, env=env, check=True, capture_output=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, env=env, check=True, capture_output=True)

    # Add a Python file so language guard passes
    (tmp_path / "main.py").write_text("print('hello')")
    subprocess.run(["git", "add", "."], cwd=tmp_path, env=env, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=tmp_path, env=env, check=True, capture_output=True)

    # Run analyze-commits which defaults to HEAD~10 -> HEAD
    result = runner.invoke(summoner, ["analyze-commits", "--repo-path", str(tmp_path)])

    assert result.exit_code == 1
    assert "Failed to analyze commits" in result.output
    # Git usually says "fatal: ambiguous argument 'HEAD~10'" or similar
    # or the service wrapper captures it.
    # We expect some failure indication.
