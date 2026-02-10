"""Subtree-prefix-aware implementation comparator for GRPO reward signals.

Compares two code subtrees (e.g., two implementations of the same feature living
side-by-side in the repo) by analyzing each independently with QPE and computing
a bounded advantage signal suitable for GRPO training.
"""

import logging
import subprocess
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

from slopometry.core.complexity_analyzer import ComplexityAnalyzer
from slopometry.core.models import ImplementationComparison
from slopometry.summoner.services.qpe_calculator import (
    calculate_qpe,
    grpo_advantage,
    smell_advantage,
)

logger = logging.getLogger(__name__)

_TIE_DEADBAND = 0.01


class SubtreeExtractionError(Exception):
    """Raised when git archive fails to extract a subtree prefix."""


def compare_subtrees(
    repo_path: Path,
    prefix_a: str,
    prefix_b: str,
    ref: str = "HEAD",
) -> ImplementationComparison | None:
    """Compare two subtree prefixes from the same git ref.

    Extracts Python files under each prefix via git archive, runs
    ComplexityAnalyzer on each, computes QPE scores, and returns the
    aggregate GRPO advantage with per-smell decomposition.

    Args:
        repo_path: Path to the git repository
        prefix_a: Subtree path prefix for implementation A
        prefix_b: Subtree path prefix for implementation B
        ref: Git ref to extract from (default: HEAD)

    Returns:
        ImplementationComparison or None if either subtree has no Python files
    """
    repo_path = repo_path.resolve()

    with (
        tempfile.TemporaryDirectory(prefix="slopometry_compare_a_") as dir_a_str,
        tempfile.TemporaryDirectory(prefix="slopometry_compare_b_") as dir_b_str,
    ):
        dir_a = Path(dir_a_str)
        dir_b = Path(dir_b_str)

        extracted_a = _extract_subtree(repo_path, ref, prefix_a, dir_a)
        if not extracted_a:
            logger.warning(f"No Python files found under prefix '{prefix_a}' at ref '{ref}'")
            return None

        extracted_b = _extract_subtree(repo_path, ref, prefix_b, dir_b)
        if not extracted_b:
            logger.warning(f"No Python files found under prefix '{prefix_b}' at ref '{ref}'")
            return None

        analyzer = ComplexityAnalyzer(working_directory=repo_path)
        metrics_a = analyzer.analyze_extended_complexity(dir_a)
        metrics_b = analyzer.analyze_extended_complexity(dir_b)

        qpe_a = calculate_qpe(metrics_a)
        qpe_b = calculate_qpe(metrics_b)

        aggregate = grpo_advantage(qpe_a, qpe_b)
        smell_advantages = smell_advantage(qpe_a, qpe_b)

        if abs(aggregate) < _TIE_DEADBAND:
            winner = "tie"
        elif aggregate > 0:
            winner = prefix_b
        else:
            winner = prefix_a

        return ImplementationComparison(
            prefix_a=prefix_a,
            prefix_b=prefix_b,
            ref=ref,
            qpe_a=qpe_a,
            qpe_b=qpe_b,
            aggregate_advantage=aggregate,
            smell_advantages=smell_advantages,
            winner=winner,
        )


def _extract_subtree(repo_path: Path, ref: str, prefix: str, dest_dir: Path) -> bool:
    """Extract Python files from a subtree prefix via git archive.

    Uses `git archive --format=tar <ref> -- <prefix>` to extract only
    files under the given prefix.

    Args:
        repo_path: Path to the git repository
        ref: Git ref to extract from
        prefix: Subtree path prefix to extract
        dest_dir: Destination directory for extracted files

    Returns:
        True if Python files were extracted, False otherwise

    Raises:
        SubtreeExtractionError: If git archive fails
    """
    result = subprocess.run(
        ["git", "archive", "--format=tar", ref, "--", prefix],
        cwd=repo_path,
        capture_output=True,
        timeout=60,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode().strip()
        raise SubtreeExtractionError(f"git archive failed for prefix '{prefix}' at ref '{ref}': {stderr}")

    tar_data = BytesIO(result.stdout)
    try:
        with tarfile.open(fileobj=tar_data, mode="r") as tar:
            python_members = [m for m in tar.getmembers() if m.name.endswith(".py")]
            if not python_members:
                return False
            tar.extractall(path=dest_dir, members=python_members, filter="data")
    except tarfile.TarError as e:
        raise SubtreeExtractionError(f"Failed to extract tar for prefix '{prefix}': {e}") from e

    return True
