"""Tests for llm_wrapper utilities."""

from slopometry.summoner.services.llm_wrapper import (
    calculate_stride_size,
    get_commit_diff,
    resolve_commit_reference,
)


def test_get_commit_diff__returns_diff_between_two_commits():
    """Test that get_commit_diff returns the diff between specified commits."""
    # Using real commits from this repo:
    # 378c825 = "Auto-whitelist slopometry solo commands on install"
    # 6170d65 = parent commit
    base = "6170d651"
    head = "378c8258"

    diff = get_commit_diff(base, head)

    assert "hook_service.py" in diff
    assert "def " in diff or "+" in diff  # Should have actual diff content
    assert "Error getting diff" not in diff


def test_get_commit_diff__does_not_include_commits_outside_range():
    """Test that the diff only includes changes within the specified range."""
    base = "6170d651"
    head = "378c8258"

    diff = get_commit_diff(base, head)

    # This commit (d53a77d - "Add lint and test exec gha") is outside the range
    # and should NOT appear in the diff
    assert ".github/workflows" not in diff


def test_resolve_commit_reference__resolves_short_hash():
    """Test that short commit hashes are resolved to full hashes."""
    short_hash = "378c825"

    resolved = resolve_commit_reference(short_hash)

    assert len(resolved) == 40  # Full SHA-1 hash
    assert resolved.startswith("378c825")


def test_calculate_stride_size__returns_commit_count_between_commits():
    """Test that stride size returns the number of commits in range."""
    base = "6170d651"
    head = "378c8258"

    stride = calculate_stride_size(base, head)

    assert stride == 1  # Only one commit between these
