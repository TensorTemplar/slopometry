"""Tests for the ProjectTracker class."""

from unittest.mock import MagicMock, patch

import pytest

from slopometry.models import ProjectSource
from slopometry.project_tracker import ProjectTracker


@pytest.fixture
def mock_subprocess_run():
    """Fixture to mock subprocess.run."""
    with patch("subprocess.run") as mock_run:
        yield mock_run


@pytest.fixture
def mock_path():
    """Fixture to mock pathlib.Path."""
    with patch("pathlib.Path") as mock_path_class:
        mock_instance = MagicMock()
        mock_path_class.return_value = mock_instance
        mock_path_class.cwd.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_toml_load():
    """Fixture to mock toml.load."""
    with patch("toml.load") as mock_load:
        yield mock_load


def test_get_project__returns_git_project_if_git_remote_exists(mock_subprocess_run, mock_path):
    """Test that project is identified from git remote."""
    mock_subprocess_run.side_effect = [
        MagicMock(returncode=0, stdout="true"),
        MagicMock(returncode=0, stdout="git@github.com:user/repo.git"),
    ]
    tracker = ProjectTracker(working_dir=mock_path)
    project = tracker.get_project()
    assert project.name == "git@github.com:user/repo.git"
    assert project.source == ProjectSource.GIT


def test_get_project__returns_none_if_not_a_git_repo(mock_subprocess_run, mock_path):
    """Test that get_project returns None if the directory is not a git repository."""
    mock_subprocess_run.return_value = MagicMock(returncode=128, stdout="")
    tracker = ProjectTracker(working_dir=mock_path)
    project = tracker.get_project()
    assert project is None


def test_get_project__returns_pyproject_if_pyproject_exists_and_no_git(mock_subprocess_run, mock_path, mock_toml_load):
    """Test that project is identified from pyproject.toml when no git remote is present."""
    mock_subprocess_run.return_value = MagicMock(returncode=128, stdout="")
    mock_path.is_file.return_value = True
    mock_toml_load.return_value = {"project": {"name": "my-pyproject-name"}}
    tracker = ProjectTracker(working_dir=mock_path)
    project = tracker.get_project()
    assert project.name == "my-pyproject-name"
    assert project.source == ProjectSource.PYPROJECT


def test_get_project__returns_none_if_neither_git_nor_pyproject_exists(mock_subprocess_run, mock_path, mock_toml_load):
    """Test that get_project returns None when no source is found."""
    mock_path.is_dir.return_value = False
    mock_path.is_file.return_value = False
    tracker = ProjectTracker(working_dir=mock_path)
    project = tracker.get_project()
    assert project is None


def test_get_project__returns_git_project_if_both_git_and_pyproject_exist(
    mock_subprocess_run, mock_path, mock_toml_load
):
    """Test that git remote takes precedence over pyproject.toml."""
    mock_subprocess_run.side_effect = [
        MagicMock(returncode=0, stdout="true"),
        MagicMock(returncode=0, stdout="git@github.com:user/repo.git"),
    ]
    mock_path.is_file.return_value = True
    mock_toml_load.return_value = {"project": {"name": "my-pyproject-name"}}
    tracker = ProjectTracker(working_dir=mock_path)
    project = tracker.get_project()
    assert project.name == "git@github.com:user/repo.git"
    assert project.source == ProjectSource.GIT
