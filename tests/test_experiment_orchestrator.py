"""Tests for experiment_orchestrator.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_test_metrics

from slopometry.core.models import ExperimentStatus, ExtendedComplexityMetrics
from slopometry.summoner.services.experiment_orchestrator import ExperimentOrchestrator


class TestExperimentOrchestratorInit:
    """Tests for ExperimentOrchestrator initialization."""

    def test_init__creates_dependencies(self, tmp_path: Path):
        """Test that orchestrator creates required dependencies."""
        with (
            patch("slopometry.summoner.services.experiment_orchestrator.EventDatabase") as MockDB,
            patch("slopometry.summoner.services.experiment_orchestrator.WorktreeManager") as MockWorktree,
            patch("slopometry.summoner.services.experiment_orchestrator.GitTracker") as MockGit,
            patch("slopometry.summoner.services.experiment_orchestrator.CLICalculator") as MockCLI,
        ):
            orchestrator = ExperimentOrchestrator(tmp_path)

            assert orchestrator.repo_path == tmp_path.resolve()
            MockDB.assert_called_once()
            MockWorktree.assert_called_once_with(tmp_path.resolve())
            MockGit.assert_called_once_with(tmp_path.resolve())
            MockCLI.assert_called_once()


class TestRunSingleExperiment:
    """Tests for ExperimentOrchestrator._run_single_experiment."""

    def test_run_single_experiment__cleans_up_worktree_on_exception(self, tmp_path: Path):
        """Test that worktree is cleaned up even when experiment fails."""
        mock_db = MagicMock()
        mock_worktree = MagicMock()
        mock_worktree.create_experiment_worktree.return_value = tmp_path / "worktree"

        with (
            patch(
                "slopometry.summoner.services.experiment_orchestrator.EventDatabase",
                return_value=mock_db,
            ),
            patch(
                "slopometry.summoner.services.experiment_orchestrator.WorktreeManager",
                return_value=mock_worktree,
            ),
            patch("slopometry.summoner.services.experiment_orchestrator.GitTracker"),
            patch("slopometry.summoner.services.experiment_orchestrator.CLICalculator"),
            patch("slopometry.summoner.services.experiment_orchestrator.ComplexityAnalyzer") as MockAnalyzer,
        ):
            # Make analyzer.analyze_extended_complexity raise an exception
            MockAnalyzer.return_value.analyze_extended_complexity.side_effect = Exception("Analysis failed")

            orchestrator = ExperimentOrchestrator(tmp_path)

            with pytest.raises(Exception, match="Analysis failed"):
                orchestrator._run_single_experiment("exp-123", "start-sha", "target-sha")

            # Verify cleanup was called despite exception
            mock_worktree.cleanup_worktree.assert_called_once_with(tmp_path / "worktree")

    def test_run_single_experiment__creates_worktree_and_saves_progress(self, tmp_path: Path):
        """Test that experiment creates worktree and saves initial progress."""
        mock_db = MagicMock()
        mock_worktree = MagicMock()
        worktree_path = tmp_path / "worktree"
        mock_worktree.create_experiment_worktree.return_value = worktree_path

        test_metrics = ExtendedComplexityMetrics(**make_test_metrics(total_complexity=50))

        with (
            patch(
                "slopometry.summoner.services.experiment_orchestrator.EventDatabase",
                return_value=mock_db,
            ),
            patch(
                "slopometry.summoner.services.experiment_orchestrator.WorktreeManager",
                return_value=mock_worktree,
            ),
            patch("slopometry.summoner.services.experiment_orchestrator.GitTracker"),
            patch("slopometry.summoner.services.experiment_orchestrator.CLICalculator"),
            patch("slopometry.summoner.services.experiment_orchestrator.ComplexityAnalyzer") as MockAnalyzer,
        ):
            MockAnalyzer.return_value.analyze_extended_complexity.return_value = test_metrics

            orchestrator = ExperimentOrchestrator(tmp_path)
            # Mock _simulate_agent_progress to avoid actual simulation
            orchestrator._simulate_agent_progress = MagicMock()

            orchestrator._run_single_experiment("exp-123", "start-sha", "target-sha")

            # Verify worktree was created
            mock_worktree.create_experiment_worktree.assert_called_once_with("exp-123", "start-sha")

            # Verify worktree path was saved to DB
            mock_db.update_experiment_worktree.assert_called_once_with("exp-123", worktree_path)

            # Verify initial progress was saved
            mock_db.save_experiment_progress.assert_called()

            # Verify cleanup
            mock_worktree.cleanup_worktree.assert_called_once_with(worktree_path)


class TestDisplayAggregateProgress:
    """Tests for ExperimentOrchestrator.display_aggregate_progress."""

    def test_display_aggregate_progress__returns_early_when_no_running_experiments(self, tmp_path: Path, capsys):
        """Test that display returns early when no experiments are running."""
        mock_db = MagicMock()
        mock_db.get_running_experiments.return_value = []

        with (
            patch(
                "slopometry.summoner.services.experiment_orchestrator.EventDatabase",
                return_value=mock_db,
            ),
            patch("slopometry.summoner.services.experiment_orchestrator.WorktreeManager"),
            patch("slopometry.summoner.services.experiment_orchestrator.GitTracker"),
            patch("slopometry.summoner.services.experiment_orchestrator.CLICalculator"),
        ):
            orchestrator = ExperimentOrchestrator(tmp_path)
            orchestrator.display_aggregate_progress()

            captured = capsys.readouterr()
            assert "EXPERIMENT PROGRESS" not in captured.out


class TestAnalyzeCommitChain:
    """Tests for ExperimentOrchestrator.analyze_commit_chain."""

    def test_analyze_commit_chain__skips_merge_commits(self, tmp_path: Path):
        """Test that merge commits are skipped during analysis."""
        mock_db = MagicMock()
        mock_db.create_commit_chain.return_value = "chain-123"

        with (
            patch(
                "slopometry.summoner.services.experiment_orchestrator.EventDatabase",
                return_value=mock_db,
            ),
            patch("slopometry.summoner.services.experiment_orchestrator.WorktreeManager"),
            patch("slopometry.summoner.services.experiment_orchestrator.GitTracker") as MockGit,
            patch("slopometry.summoner.services.experiment_orchestrator.CLICalculator"),
            patch("slopometry.summoner.services.experiment_orchestrator.subprocess.run") as mock_subprocess,
            patch("slopometry.summoner.services.experiment_orchestrator.console") as mock_console,
        ):
            # First call: git rev-list returns commits
            # Second+ calls: git rev-list --parents returns parent info
            def subprocess_side_effect(*args, **kwargs):
                result = MagicMock()
                result.returncode = 0

                cmd = args[0]
                if "rev-list" in cmd and "--parents" not in cmd:
                    # Initial rev-list call
                    result.stdout = "commit1\ncommit2\nmerge_commit\n"
                elif "--parents" in cmd:
                    # Parent check calls
                    commit_sha = cmd[-1]
                    if commit_sha == "merge_commit":
                        # Merge commit has 2 parents
                        result.stdout = "merge_commit parent1 parent2"
                    else:
                        # Regular commit has 1 parent
                        result.stdout = f"{commit_sha} single_parent"
                return result

            mock_subprocess.side_effect = subprocess_side_effect

            # Mock git_tracker context manager for extract_files_from_commit_ctx
            mock_git_tracker = MockGit.return_value
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=None)  # Return None to skip analysis
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_git_tracker.extract_files_from_commit_ctx.return_value = mock_context

            orchestrator = ExperimentOrchestrator(tmp_path)
            orchestrator.analyze_commit_chain("base", "head")

            # Check that console.print was called with merge commit skip message
            skip_calls = [c for c in mock_console.print.call_args_list if "Skipping merge commit" in str(c)]
            assert len(skip_calls) == 1, "Expected one merge commit to be skipped"

    def test_analyze_commit_chain__handles_empty_commit_range(self, tmp_path: Path):
        """Test that empty commit range is handled.

        Note: Due to Python's split behavior, "".split("\\n") returns [""],
        not [], so the code will try to process one empty commit. This test
        verifies the code doesn't crash in this edge case.
        """
        mock_db = MagicMock()
        mock_db.create_commit_chain.return_value = "chain-123"

        with (
            patch(
                "slopometry.summoner.services.experiment_orchestrator.EventDatabase",
                return_value=mock_db,
            ),
            patch("slopometry.summoner.services.experiment_orchestrator.WorktreeManager"),
            patch("slopometry.summoner.services.experiment_orchestrator.GitTracker") as MockGit,
            patch("slopometry.summoner.services.experiment_orchestrator.CLICalculator"),
            patch("slopometry.summoner.services.experiment_orchestrator.subprocess.run") as mock_subprocess,
            patch("slopometry.summoner.services.experiment_orchestrator.console"),
            patch("slopometry.summoner.services.experiment_orchestrator.ComplexityAnalyzer"),
        ):

            def subprocess_side_effect(*args, **kwargs):
                result = MagicMock()
                result.returncode = 0
                cmd = args[0]
                if "rev-list" in cmd and "--parents" not in cmd:
                    result.stdout = ""  # Empty result leads to [""] after split
                elif "--parents" in cmd:
                    # Empty commit SHA would return just itself
                    result.stdout = ""
                return result

            mock_subprocess.side_effect = subprocess_side_effect

            # Mock git_tracker to return None (skip analysis)
            mock_git_tracker = MockGit.return_value
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(return_value=None)
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_git_tracker.extract_files_from_commit_ctx.return_value = mock_context

            orchestrator = ExperimentOrchestrator(tmp_path)
            # Should not crash
            orchestrator.analyze_commit_chain("base", "head")

            # Verify chain was created (even with empty commit list)
            mock_db.create_commit_chain.assert_called_once()

    def test_analyze_commit_chain__handles_git_operation_error(self, tmp_path: Path):
        """Test that git operation errors are handled gracefully."""
        from slopometry.core.git_tracker import GitOperationError

        mock_db = MagicMock()
        mock_db.create_commit_chain.return_value = "chain-123"

        with (
            patch(
                "slopometry.summoner.services.experiment_orchestrator.EventDatabase",
                return_value=mock_db,
            ),
            patch("slopometry.summoner.services.experiment_orchestrator.WorktreeManager"),
            patch("slopometry.summoner.services.experiment_orchestrator.GitTracker") as MockGit,
            patch("slopometry.summoner.services.experiment_orchestrator.CLICalculator"),
            patch("slopometry.summoner.services.experiment_orchestrator.subprocess.run") as mock_subprocess,
            patch("slopometry.summoner.services.experiment_orchestrator.console") as mock_console,
        ):
            # Rev-list returns one commit
            def subprocess_side_effect(*args, **kwargs):
                result = MagicMock()
                result.returncode = 0
                cmd = args[0]
                if "rev-list" in cmd and "--parents" not in cmd:
                    result.stdout = "commit1\n"
                elif "--parents" in cmd:
                    result.stdout = "commit1 parent1"
                return result

            mock_subprocess.side_effect = subprocess_side_effect

            # Make extract_files_from_commit_ctx raise GitOperationError
            mock_git_tracker = MockGit.return_value
            mock_context = MagicMock()
            mock_context.__enter__ = MagicMock(side_effect=GitOperationError("Git failed"))
            mock_context.__exit__ = MagicMock(return_value=False)
            mock_git_tracker.extract_files_from_commit_ctx.return_value = mock_context

            orchestrator = ExperimentOrchestrator(tmp_path)
            # Should not raise, should handle gracefully
            orchestrator.analyze_commit_chain("base", "head")

            # Check that skipping message was printed
            skip_calls = [c for c in mock_console.print.call_args_list if "Skipping commit" in str(c)]
            assert len(skip_calls) == 1


class TestRunParallelExperiments:
    """Tests for ExperimentOrchestrator.run_parallel_experiments."""

    def test_run_parallel_experiments__returns_empty_dict_for_empty_input(self, tmp_path: Path):
        """Test that empty commit pairs returns empty dict."""
        mock_db = MagicMock()

        with (
            patch(
                "slopometry.summoner.services.experiment_orchestrator.EventDatabase",
                return_value=mock_db,
            ),
            patch("slopometry.summoner.services.experiment_orchestrator.WorktreeManager"),
            patch("slopometry.summoner.services.experiment_orchestrator.GitTracker"),
            patch("slopometry.summoner.services.experiment_orchestrator.CLICalculator"),
        ):
            orchestrator = ExperimentOrchestrator(tmp_path)
            result = orchestrator.run_parallel_experiments([])

            assert result == {}
            mock_db.save_experiment_run.assert_not_called()

    def test_run_parallel_experiments__creates_experiments_for_each_pair(self, tmp_path: Path):
        """Test that an experiment is created for each commit pair."""
        mock_db = MagicMock()
        mock_db.get_running_experiments.return_value = []

        with (
            patch(
                "slopometry.summoner.services.experiment_orchestrator.EventDatabase",
                return_value=mock_db,
            ),
            patch("slopometry.summoner.services.experiment_orchestrator.WorktreeManager"),
            patch("slopometry.summoner.services.experiment_orchestrator.GitTracker"),
            patch("slopometry.summoner.services.experiment_orchestrator.CLICalculator"),
            patch("slopometry.summoner.services.experiment_orchestrator.ProcessPoolExecutor") as MockExecutor,
        ):
            # Mock the executor to immediately complete futures
            mock_executor_instance = MagicMock()
            mock_future = MagicMock()
            mock_future.done.return_value = True
            mock_future.result.return_value = None
            mock_executor_instance.submit.return_value = mock_future
            mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
            mock_executor_instance.__exit__ = MagicMock(return_value=False)
            MockExecutor.return_value = mock_executor_instance

            orchestrator = ExperimentOrchestrator(tmp_path)
            commit_pairs = [("start1", "target1"), ("start2", "target2")]

            result = orchestrator.run_parallel_experiments(commit_pairs, max_workers=2)

            # Should have 2 experiments
            assert len(result) == 2
            # Should have called save_experiment_run twice
            assert mock_db.save_experiment_run.call_count == 2
            # Should have submitted 2 tasks
            assert mock_executor_instance.submit.call_count == 2

    def test_run_parallel_experiments__handles_failed_experiment(self, tmp_path: Path, capsys):
        """Test that failed experiments are marked as FAILED."""
        mock_db = MagicMock()
        mock_db.get_running_experiments.return_value = []

        with (
            patch(
                "slopometry.summoner.services.experiment_orchestrator.EventDatabase",
                return_value=mock_db,
            ),
            patch("slopometry.summoner.services.experiment_orchestrator.WorktreeManager"),
            patch("slopometry.summoner.services.experiment_orchestrator.GitTracker"),
            patch("slopometry.summoner.services.experiment_orchestrator.CLICalculator"),
            patch("slopometry.summoner.services.experiment_orchestrator.ProcessPoolExecutor") as MockExecutor,
        ):
            # Mock the executor with a failing future
            mock_executor_instance = MagicMock()
            mock_future = MagicMock()
            mock_future.done.return_value = True
            mock_future.result.side_effect = Exception("Experiment failed!")
            mock_executor_instance.submit.return_value = mock_future
            mock_executor_instance.__enter__ = MagicMock(return_value=mock_executor_instance)
            mock_executor_instance.__exit__ = MagicMock(return_value=False)
            MockExecutor.return_value = mock_executor_instance

            orchestrator = ExperimentOrchestrator(tmp_path)
            result = orchestrator.run_parallel_experiments([("start", "target")])

            # Experiment should be marked as FAILED
            assert len(result) == 1
            experiment = list(result.values())[0]
            assert experiment.status == ExperimentStatus.FAILED

            # Should have called update_experiment_run with failed status
            mock_db.update_experiment_run.assert_called()

            # Should print error message
            captured = capsys.readouterr()
            assert "failed" in captured.out.lower()
