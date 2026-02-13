"""Tests for language configuration module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from slopometry.core.language_config import (
    LANGUAGE_CONFIGS,
    PYTHON_CONFIG,
    LanguageConfig,
    get_all_supported_configs,
    get_combined_git_patterns,
    get_combined_ignore_dirs,
    get_language_config,
    should_ignore_path,
)
from slopometry.core.language_models import ProjectLanguage


class TestLanguageConfig:
    """Tests for LanguageConfig dataclass."""

    def test_python_config__has_correct_extensions(self):
        """Verify Python config has .py extension."""
        assert PYTHON_CONFIG.extensions == (".py",)

    def test_python_config__has_git_patterns(self):
        """Verify Python config has *.py git pattern."""
        assert "*.py" in PYTHON_CONFIG.git_patterns

    def test_python_config__has_ignore_dirs(self):
        """Verify Python config has common ignore directories."""
        assert "__pycache__" in PYTHON_CONFIG.ignore_dirs
        assert ".venv" in PYTHON_CONFIG.ignore_dirs
        assert "dist" in PYTHON_CONFIG.ignore_dirs
        assert "build" in PYTHON_CONFIG.ignore_dirs

    def test_python_config__has_test_patterns(self):
        """Verify Python config has test file patterns."""
        assert "test_*.py" in PYTHON_CONFIG.test_patterns

    def test_matches_extension__python_file(self):
        """Verify extension matching for Python files."""
        assert PYTHON_CONFIG.matches_extension("foo.py")
        assert PYTHON_CONFIG.matches_extension(Path("src/bar.py"))
        assert PYTHON_CONFIG.matches_extension("test.PY")  # Case insensitive

    def test_matches_extension__non_python_file(self):
        """Verify extension matching rejects non-Python files."""
        assert not PYTHON_CONFIG.matches_extension("foo.rs")
        assert not PYTHON_CONFIG.matches_extension("foo.js")
        assert not PYTHON_CONFIG.matches_extension("foo.pyc")


class TestLanguageConfigRegistry:
    """Tests for language config registry functions."""

    def test_get_language_config__python(self):
        """Verify getting Python config from registry."""
        config = get_language_config(ProjectLanguage.PYTHON)
        assert config == PYTHON_CONFIG

    def test_get_language_config__unsupported_raises(self):
        """Verify getting unsupported language raises KeyError."""
        # Create a fake language enum value for testing
        # This simulates what would happen if a language is added to enum but not registry
        with pytest.raises(KeyError, match="No configuration found"):
            # We can't easily create a fake enum, so test with a manipulated registry
            original = LANGUAGE_CONFIGS.copy()
            try:
                LANGUAGE_CONFIGS.clear()
                get_language_config(ProjectLanguage.PYTHON)
            finally:
                LANGUAGE_CONFIGS.update(original)

    def test_get_all_supported_configs__returns_all(self):
        """Verify getting all supported configs."""
        configs = get_all_supported_configs()
        assert len(configs) >= 1
        assert PYTHON_CONFIG in configs

    def test_get_combined_git_patterns__all_languages(self):
        """Verify combined git patterns include all supported languages."""
        patterns = get_combined_git_patterns(None)
        assert "*.py" in patterns

    def test_get_combined_git_patterns__specific_language(self):
        """Verify combined git patterns for specific language."""
        patterns = get_combined_git_patterns([ProjectLanguage.PYTHON])
        assert "*.py" in patterns

    def test_get_combined_ignore_dirs__includes_python_dirs(self):
        """Verify combined ignore dirs include Python-specific dirs."""
        dirs = get_combined_ignore_dirs(None)
        assert "__pycache__" in dirs
        assert ".venv" in dirs
        assert "dist" in dirs


class TestShouldIgnorePath:
    """Tests for should_ignore_path function."""

    def test_should_ignore_path__pycache(self):
        """Verify __pycache__ paths are ignored."""
        assert should_ignore_path("__pycache__/foo.pyc")
        assert should_ignore_path("src/__pycache__/bar.pyc")

    def test_should_ignore_path__venv(self):
        """Verify virtual environment paths are ignored."""
        assert should_ignore_path(".venv/lib/python3.12/site-packages/foo.py")
        assert should_ignore_path("venv/bin/activate")

    def test_should_ignore_path__dist(self):
        """Verify dist directory is ignored."""
        assert should_ignore_path("dist/package-1.0.0.tar.gz")
        assert should_ignore_path("dist/package-1.0.0-py3-none-any.whl")

    def test_should_ignore_path__build(self):
        """Verify build directory is ignored."""
        assert should_ignore_path("build/lib/package/module.py")

    def test_should_ignore_path__egg_info(self):
        """Verify *.egg-info directories are ignored."""
        assert should_ignore_path("package.egg-info/PKG-INFO")
        assert should_ignore_path("my_package.egg-info/SOURCES.txt")

    def test_should_ignore_path__pytest_cache(self):
        """Verify pytest cache is ignored."""
        assert should_ignore_path(".pytest_cache/v/cache/lastfailed")

    def test_should_ignore_path__mypy_cache(self):
        """Verify mypy cache is ignored."""
        assert should_ignore_path(".mypy_cache/3.12/module.meta.json")

    def test_should_ignore_path__source_file_not_ignored(self):
        """Verify normal source files are NOT ignored."""
        assert not should_ignore_path("src/module.py")
        assert not should_ignore_path("tests/test_module.py")
        assert not should_ignore_path("package/__init__.py")

    def test_should_ignore_path__specific_language(self):
        """Verify language-specific ignore works."""
        # Python-specific ignores should work when Python is specified
        assert should_ignore_path("__pycache__/foo.py", [ProjectLanguage.PYTHON])


class TestLanguageConfigFrozen:
    """Tests for LanguageConfig immutability."""

    def test_language_config__is_frozen(self):
        """Verify LanguageConfig is immutable."""
        with pytest.raises(ValidationError):
            PYTHON_CONFIG.language = ProjectLanguage.PYTHON  # type: ignore

    def test_language_config__custom_creation(self):
        """Verify custom LanguageConfig can be created."""
        custom = LanguageConfig(
            language=ProjectLanguage.PYTHON,
            extensions=(".custom",),
            git_patterns=("*.custom",),
        )
        assert custom.extensions == (".custom",)
        assert custom.ignore_dirs == ()  # Default empty tuple
