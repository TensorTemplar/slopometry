"""Tests for unified code analyzer using rust-code-analysis."""

from pathlib import Path
from unittest.mock import patch

import pytest

from slopometry.core.code_analyzer import (
    SUPPORTED_EXTENSIONS,
    CodeAnalysisNotInstalledError,
    CodeAnalyzer,
    _analyze_single_file,
    _safe_float,
)
from slopometry.core.models import FileAnalysisResult


class TestSafeFloat:
    """Tests for _safe_float helper function."""

    def test_safe_float__returns_value_for_normal_float(self) -> None:
        """Should return the value for normal floats."""
        assert _safe_float(3.14) == 3.14

    def test_safe_float__returns_zero_for_none(self) -> None:
        """Should return 0.0 for None."""
        assert _safe_float(None) == 0.0

    def test_safe_float__returns_zero_for_nan(self) -> None:
        """Should return 0.0 for NaN."""
        assert _safe_float(float("nan")) == 0.0

    def test_safe_float__returns_zero_for_inf(self) -> None:
        """Should return 0.0 for infinity."""
        assert _safe_float(float("inf")) == 0.0
        assert _safe_float(float("-inf")) == 0.0


class TestSupportedExtensions:
    """Tests for supported extensions constant."""

    def test_supported_extensions__includes_python(self) -> None:
        """Should include Python extension."""
        assert ".py" in SUPPORTED_EXTENSIONS

    def test_supported_extensions__includes_rust(self) -> None:
        """Should include Rust extension."""
        assert ".rs" in SUPPORTED_EXTENSIONS

    def test_supported_extensions__includes_javascript(self) -> None:
        """Should include JavaScript extensions."""
        assert ".js" in SUPPORTED_EXTENSIONS
        assert ".jsx" in SUPPORTED_EXTENSIONS

    def test_supported_extensions__includes_typescript(self) -> None:
        """Should include TypeScript extensions."""
        assert ".ts" in SUPPORTED_EXTENSIONS
        assert ".tsx" in SUPPORTED_EXTENSIONS


class TestCodeAnalyzer:
    """Tests for CodeAnalyzer class."""

    def test_init__raises_when_rca_not_installed(self) -> None:
        """Should raise CodeAnalysisNotInstalledError when rust_code_analysis not installed."""
        with patch.dict("sys.modules", {"rust_code_analysis": None}):
            with patch(
                "slopometry.core.code_analyzer._check_rca_installed",
                side_effect=CodeAnalysisNotInstalledError("Not installed"),
            ):
                with pytest.raises(CodeAnalysisNotInstalledError):
                    CodeAnalyzer()

    def test_is_supported__returns_true_for_python(self) -> None:
        """Should return True for Python files."""
        analyzer = CodeAnalyzer()
        assert analyzer.is_supported(Path("test.py"))

    def test_is_supported__returns_true_for_rust(self) -> None:
        """Should return True for Rust files."""
        analyzer = CodeAnalyzer()
        assert analyzer.is_supported(Path("test.rs"))

    def test_is_supported__returns_false_for_unsupported(self) -> None:
        """Should return False for unsupported extensions."""
        analyzer = CodeAnalyzer()
        assert not analyzer.is_supported(Path("test.txt"))
        assert not analyzer.is_supported(Path("test.md"))

    def test_analyze_file__returns_result_for_valid_python(self, tmp_path: Path) -> None:
        """Should return FileAnalysisResult for valid Python file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo(): pass")

        analyzer = CodeAnalyzer()
        result = analyzer.analyze_file(test_file)

        assert result.path == str(test_file)
        assert result.error is None
        assert isinstance(result.complexity, int)
        assert isinstance(result.tokens, int)

    def test_analyze_file__returns_error_for_missing_file(self, tmp_path: Path) -> None:
        """Should return result with error for missing file."""
        missing_file = tmp_path / "missing.py"

        analyzer = CodeAnalyzer()
        result = analyzer.analyze_file(missing_file)

        assert result.path == str(missing_file)
        assert result.error is not None

    def test_analyze_files__returns_results_for_multiple_files(self, tmp_path: Path) -> None:
        """Should return list of results for multiple files."""
        file1 = tmp_path / "a.py"
        file2 = tmp_path / "b.py"
        file1.write_text("x = 1")
        file2.write_text("y = 2")

        analyzer = CodeAnalyzer()
        results = analyzer.analyze_files([file1, file2])

        assert len(results) == 2
        assert all(isinstance(r, FileAnalysisResult) for r in results)


class TestAnalyzeSingleFile:
    """Tests for module-level _analyze_single_file function."""

    def test_analyze_single_file__returns_result(self, tmp_path: Path) -> None:
        """Should return FileAnalysisResult for valid file."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def bar(): return 42")

        result = _analyze_single_file(test_file)

        assert result.path == str(test_file)
        assert result.error is None

    def test_analyze_single_file__handles_error(self, tmp_path: Path) -> None:
        """Should return result with error for invalid file."""
        missing_file = tmp_path / "missing.py"

        result = _analyze_single_file(missing_file)

        assert result.path == str(missing_file)
        assert result.error is not None
        assert result.complexity == 0
