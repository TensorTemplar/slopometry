"""Git worktree management for parallel experiments."""

import shutil
import subprocess
import tempfile
from pathlib import Path


class WorktreeManager:
    """Manages git worktrees for parallel experiments."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path.resolve()

    def create_experiment_worktree(self, experiment_id: str, target_commit: str) -> Path:
        """Create isolated worktree for experiment.

        Args:
            experiment_id: Unique identifier for the experiment
            target_commit: Git commit reference to checkout

        Returns:
            Path to the created worktree

        Raises:
            subprocess.CalledProcessError: If git worktree creation fails
        """
        worktree_path = Path(tempfile.mkdtemp(prefix=f"slopometry_exp_{experiment_id}_"))

        try:
            subprocess.run(
                ["git", "worktree", "add", str(worktree_path), target_commit],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return worktree_path
        except subprocess.CalledProcessError as e:
            if worktree_path.exists():
                shutil.rmtree(worktree_path, ignore_errors=True)
            raise RuntimeError(f"Failed to create worktree: {e.stderr}")

    def cleanup_worktree(self, worktree_path: Path) -> None:
        """Remove worktree after experiment.

        Args:
            worktree_path: Path to the worktree to remove
        """
        try:
            subprocess.run(
                ["git", "worktree", "remove", str(worktree_path)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on error
            )
        except subprocess.CalledProcessError:
            pass

        # Ensure directory is removed even if git command failed
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)

    def list_worktrees(self) -> list[dict[str, str]]:
        """List all worktrees for this repository.

        Returns:
            List of dicts with 'path', 'head', and 'branch' keys
        """
        try:
            result = subprocess.run(
                ["git", "worktree", "list", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )

            worktrees = []
            current_worktree: dict[str, str] = {}

            for line in result.stdout.strip().split("\n"):
                if not line:
                    if current_worktree:
                        worktrees.append(current_worktree)
                        current_worktree = {}
                elif line.startswith("worktree "):
                    current_worktree["path"] = line.split(" ", 1)[1]
                elif line.startswith("HEAD "):
                    current_worktree["head"] = line.split(" ", 1)[1]
                elif line.startswith("branch "):
                    current_worktree["branch"] = line.split(" ", 1)[1]

            if current_worktree:
                worktrees.append(current_worktree)

            return worktrees

        except subprocess.CalledProcessError:
            return []

    def cleanup_all_experiment_worktrees(self) -> None:
        """Clean up all slopometry experiment worktrees."""
        worktrees = self.list_worktrees()

        for worktree in worktrees:
            path = Path(worktree["path"])
            if path.name.startswith("slopometry_exp_"):
                self.cleanup_worktree(path)
