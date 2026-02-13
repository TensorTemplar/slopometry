"""Language-specific configuration for source file detection and caching.

This module provides a registry of language configurations that define:
- Source file extensions and git patterns
- Patterns for files to ignore (build artifacts, caches)
- Test file naming conventions

The design allows easy extension to new languages while keeping type safety.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from slopometry.core.models import ProjectLanguage


class LanguageConfig(BaseModel):
    """Configuration for a programming language's file patterns.

    Attributes:
        language: The ProjectLanguage enum value
        extensions: File extensions for source files (e.g., [".py"])
        git_patterns: Patterns for git diff commands (e.g., ["*.py"])
        ignore_dirs: Directories to ignore (build artifacts, caches)
        test_patterns: Glob patterns for test files
    """

    model_config = {"frozen": True}

    language: "ProjectLanguage"
    extensions: tuple[str, ...]
    git_patterns: tuple[str, ...]
    ignore_dirs: tuple[str, ...] = Field(default_factory=tuple)
    test_patterns: tuple[str, ...] = Field(default_factory=tuple)

    def matches_extension(self, file_path: Path | str) -> bool:
        """Check if a file path matches this language's extensions."""
        suffix = Path(file_path).suffix.lower()
        return suffix in self.extensions


# Language configuration registry
PYTHON_CONFIG = LanguageConfig(
    language=ProjectLanguage.PYTHON,
    extensions=(".py",),
    git_patterns=("*.py",),
    ignore_dirs=(
        # Virtual environments
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env",
        "site-packages",
        # Build artifacts (bdist, wheel, egg)
        "dist",
        "build",
        ".eggs",
        "*.egg-info",
        # Testing caches
        ".pytest_cache",
        ".tox",
        ".nox",
        ".hypothesis",
        # Type checker / linter caches
        ".mypy_cache",
        ".ruff_cache",
        ".pytype",
        # Coverage artifacts
        "htmlcov",
        ".coverage",
        # Jupyter
        ".ipynb_checkpoints",
        # IDE
        ".idea",
        ".vscode",
    ),
    test_patterns=(
        "test_*.py",
        "*_test.py",
        "tests/**/*.py",
    ),
)

RUST_CONFIG = LanguageConfig(
    language=ProjectLanguage.RUST,
    extensions=(".rs",),
    git_patterns=("*.rs",),
    ignore_dirs=(
        "target",  # Cargo build output
        ".cargo",  # Cargo cache
    ),
    test_patterns=(
        "*_test.rs",
        "tests/**/*.rs",
    ),
)

# Registry mapping ProjectLanguage to its config
LANGUAGE_CONFIGS: dict[ProjectLanguage, LanguageConfig] = {
    ProjectLanguage.PYTHON: PYTHON_CONFIG,
    ProjectLanguage.RUST: RUST_CONFIG,
}


def get_language_config(language: ProjectLanguage) -> LanguageConfig:
    """Get the configuration for a specific language.

    Args:
        language: The ProjectLanguage to get config for

    Returns:
        LanguageConfig for the specified language

    Raises:
        KeyError: If language is not supported
    """
    if language not in LANGUAGE_CONFIGS:
        raise KeyError(f"No configuration found for language: {language}")
    return LANGUAGE_CONFIGS[language]


def get_all_supported_configs() -> list[LanguageConfig]:
    """Get configurations for all supported languages.

    Returns:
        List of all registered LanguageConfig objects
    """
    return list(LANGUAGE_CONFIGS.values())


def get_combined_git_patterns(languages: list[ProjectLanguage] | None = None) -> list[str]:
    """Get combined git patterns for multiple languages.

    Args:
        languages: List of languages to include, or None for all supported

    Returns:
        Combined list of git patterns for the specified languages
    """
    if languages is None:
        configs = get_all_supported_configs()
    else:
        configs = [get_language_config(lang) for lang in languages]

    patterns: list[str] = []
    for config in configs:
        patterns.extend(config.git_patterns)
    return patterns


def get_combined_ignore_dirs(languages: list[ProjectLanguage] | None = None) -> set[str]:
    """Get combined ignore directories for multiple languages.

    Args:
        languages: List of languages to include, or None for all supported

    Returns:
        Combined set of directories to ignore
    """
    if languages is None:
        configs = get_all_supported_configs()
    else:
        configs = [get_language_config(lang) for lang in languages]

    ignore_dirs: set[str] = set()
    for config in configs:
        ignore_dirs.update(config.ignore_dirs)
    return ignore_dirs


def should_ignore_path(file_path: Path | str, languages: list[ProjectLanguage] | None = None) -> bool:
    """Check if a file path should be ignored based on language configs.

    Args:
        file_path: Path to check
        languages: List of languages to use for ignore patterns, or None for all

    Returns:
        True if the path should be ignored (is in an ignored directory)
    """
    ignore_dirs = get_combined_ignore_dirs(languages)
    path = Path(file_path)

    # Check if any part of the path is in ignore_dirs
    for part in path.parts:
        if part in ignore_dirs:
            return True
        # Handle glob-like patterns (e.g., "*.egg-info")
        for ignore_pattern in ignore_dirs:
            if ignore_pattern.startswith("*") and part.endswith(ignore_pattern[1:]):
                return True
    return False
