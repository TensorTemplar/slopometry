"""Tests for language detection and guard functionality."""

import subprocess
from pathlib import Path

from slopometry.core.language_detector import (
    EXTENSION_MAP,
    KNOWN_UNSUPPORTED_EXTENSIONS,
    LanguageDetector,
)
from slopometry.core.language_guard import check_language_support
from slopometry.core.models.hook import LanguageGuardResult, ProjectLanguage


class TestLanguageDetector:
    """Tests for LanguageDetector class."""

    def test_detect_languages__detects_python_from_git_tracked_files(self, tmp_path: Path) -> None:
        """Should detect Python when .py files are git-tracked."""
        # Create a git repo with Python files
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)

        detector = LanguageDetector(tmp_path)
        supported, unsupported = detector.detect_languages()

        assert ProjectLanguage.PYTHON in supported
        assert len(unsupported) == 0

    def test_detect_languages__reports_unsupported_languages(self, tmp_path: Path) -> None:
        """Should report unsupported languages like Go, TypeScript (but Rust is now supported)."""
        # Create a git repo with mixed files
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "main.rs").write_text("fn main() {}")
        (tmp_path / "app.go").write_text("package main")
        (tmp_path / "index.ts").write_text("const x: number = 1")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)

        detector = LanguageDetector(tmp_path)
        supported, unsupported = detector.detect_languages()

        assert ProjectLanguage.RUST in supported  # Rust is now supported
        assert "Go" in unsupported
        assert "TypeScript" in unsupported

    def test_detect_languages__handles_empty_repo(self, tmp_path: Path) -> None:
        """Should return empty sets for empty git repo."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)

        detector = LanguageDetector(tmp_path)
        supported, unsupported = detector.detect_languages()

        assert len(supported) == 0
        assert len(unsupported) == 0

    def test_detect_languages__handles_non_git_directory(self, tmp_path: Path) -> None:
        """Should return empty sets for non-git directory."""
        (tmp_path / "main.py").write_text("print('hello')")

        detector = LanguageDetector(tmp_path)
        supported, unsupported = detector.detect_languages()

        assert len(supported) == 0
        assert len(unsupported) == 0

    def test_detect_languages__mixed_supported_and_unsupported(self, tmp_path: Path) -> None:
        """Should correctly categorize both supported and unsupported languages."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "lib.rs").write_text("pub fn foo() {}")
        (tmp_path / "app.go").write_text("package main")  # Go is still unsupported
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)

        detector = LanguageDetector(tmp_path)
        supported, unsupported = detector.detect_languages()

        assert ProjectLanguage.PYTHON in supported
        assert ProjectLanguage.RUST in supported  # Rust is now supported
        assert "Go" in unsupported


class TestCheckLanguageSupport:
    """Tests for check_language_support function."""

    def test_check_language_support__allowed_when_python_present(self, tmp_path: Path) -> None:
        """Should return allowed=True when required Python is detected."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "main.py").write_text("print('hello')")
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)

        result = check_language_support(tmp_path, ProjectLanguage.PYTHON)

        assert result.allowed is True
        assert result.required_language == ProjectLanguage.PYTHON
        assert ProjectLanguage.PYTHON in result.detected_supported

    def test_check_language_support__not_allowed_when_python_missing(self, tmp_path: Path) -> None:
        """Should return allowed=False when required Python is not detected."""
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        (tmp_path / "main.rs").write_text("fn main() {}")
        (tmp_path / "app.go").write_text("package main")  # Go is still unsupported
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)

        result = check_language_support(tmp_path, ProjectLanguage.PYTHON)

        assert result.allowed is False
        assert result.required_language == ProjectLanguage.PYTHON
        assert ProjectLanguage.PYTHON not in result.detected_supported
        assert ProjectLanguage.RUST in result.detected_supported  # Rust is now supported
        assert "Go" in result.detected_unsupported


class TestLanguageGuardResult:
    """Tests for LanguageGuardResult model."""

    def test_format_warning__includes_unsupported_languages(self) -> None:
        """Should format warning message with unsupported language names."""
        result = LanguageGuardResult(
            allowed=True,
            required_language=ProjectLanguage.PYTHON,
            detected_supported={ProjectLanguage.PYTHON},
            detected_unsupported={"Rust", "Go"},
        )

        warning = result.format_warning()

        assert warning is not None
        assert "Rust" in warning
        assert "Go" in warning
        assert "not yet supported" in warning

    def test_format_warning__returns_none_when_no_unsupported(self) -> None:
        """Should return None when no unsupported languages detected."""
        result = LanguageGuardResult(
            allowed=True,
            required_language=ProjectLanguage.PYTHON,
            detected_supported={ProjectLanguage.PYTHON},
            detected_unsupported=set(),
        )

        warning = result.format_warning()

        assert warning is None

    def test_format_warning__sorts_language_names(self) -> None:
        """Should sort language names alphabetically in warning."""
        result = LanguageGuardResult(
            allowed=True,
            required_language=ProjectLanguage.PYTHON,
            detected_supported={ProjectLanguage.PYTHON},
            detected_unsupported={"TypeScript", "Go", "Rust"},
        )

        warning = result.format_warning()

        assert warning is not None
        # Check alphabetical order: Go, Rust, TypeScript
        go_pos = warning.find("Go")
        rust_pos = warning.find("Rust")
        ts_pos = warning.find("TypeScript")
        assert go_pos < rust_pos < ts_pos


class TestExtensionMaps:
    """Tests for extension mapping constants."""

    def test_extension_map__contains_python(self) -> None:
        """Python extension should be in supported map."""
        assert ".py" in EXTENSION_MAP
        assert EXTENSION_MAP[".py"] == ProjectLanguage.PYTHON

    def test_known_unsupported__contains_common_languages(self) -> None:
        """Common unsupported languages should be in unsupported map (Rust is now supported)."""
        assert ".rs" not in KNOWN_UNSUPPORTED_EXTENSIONS  # Rust is now supported
        assert ".rs" in EXTENSION_MAP  # Rust should be in supported map
        assert EXTENSION_MAP[".rs"] == ProjectLanguage.RUST
        assert ".go" in KNOWN_UNSUPPORTED_EXTENSIONS
        assert ".ts" in KNOWN_UNSUPPORTED_EXTENSIONS
        assert ".js" in KNOWN_UNSUPPORTED_EXTENSIONS
