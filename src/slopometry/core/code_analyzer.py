"""Unified code complexity analysis using rust-code-analysis Python binding."""

import logging
import math
from pathlib import Path

from slopometry.core.models.complexity import FileAnalysisResult
from slopometry.core.tokenizer import count_file_tokens

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".py", ".rs", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".cpp", ".c", ".kt"})


class CodeAnalysisError(Exception):
    """Raised when code analysis fails."""


class CodeAnalysisNotInstalledError(CodeAnalysisError):
    """Raised when rust-code-analysis is not installed."""


def _check_rca_installed() -> None:
    """Check if rust-code-analysis is installed."""
    try:
        import rust_code_analysis  # noqa: F401
    except ImportError as e:
        raise CodeAnalysisNotInstalledError(
            "rust-code-analysis not installed. Install from wheel or build with maturin."
        ) from e


def _safe_float(value: float | None) -> float:
    """Convert NaN/None to 0.0."""
    if value is None:
        return 0.0
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return float(value)


def _analyze_single_file(file_path: Path) -> FileAnalysisResult:
    """Analyze a single source file.

    Module-level function for ProcessPoolExecutor compatibility.
    """
    import rust_code_analysis as rca

    from slopometry.core.tokenizer import count_file_tokens

    try:
        result = rca.analyze_file(str(file_path))
        m = result.metrics

        cc_sum = sum(f.metrics.cyclomatic.sum for f in result.get_functions())

        return FileAnalysisResult(
            path=str(file_path),
            complexity=int(cc_sum),
            volume=_safe_float(m.halstead.volume),
            difficulty=_safe_float(m.halstead.difficulty),
            effort=_safe_float(m.halstead.effort),
            mi=_safe_float(m.mi.mi_original),
            tokens=count_file_tokens(file_path),
        )
    except Exception as e:
        return FileAnalysisResult(
            path=str(file_path),
            complexity=0,
            volume=0.0,
            difficulty=0.0,
            effort=0.0,
            mi=0.0,
            error=str(e),
        )


class CodeAnalyzer:
    """Unified code analyzer using rust-code-analysis Python binding.

    Supports Python, Rust, JavaScript, TypeScript, Go, Java, C/C++, Kotlin.
    """

    def __init__(self) -> None:
        """Initialize the analyzer.

        Raises:
            CodeAnalysisNotInstalledError: If rust-code-analysis is not installed.
        """
        _check_rca_installed()
        import rust_code_analysis as rca

        self._rca = rca

    def analyze_file(self, file_path: Path) -> FileAnalysisResult:
        """Analyze a single source file.

        Args:
            file_path: Path to the source file.

        Returns:
            FileAnalysisResult with metrics.
        """
        try:
            result = self._rca.analyze_file(str(file_path))
            m = result.metrics

            cc_sum = sum(f.metrics.cyclomatic.sum for f in result.get_functions())

            return FileAnalysisResult(
                path=str(file_path),
                complexity=int(cc_sum),
                volume=_safe_float(m.halstead.volume),
                difficulty=_safe_float(m.halstead.difficulty),
                effort=_safe_float(m.halstead.effort),
                mi=_safe_float(m.mi.mi_original),
                tokens=count_file_tokens(file_path),
            )
        except Exception as e:
            logger.warning("Failed to analyze %s: %s", file_path, e)
            return FileAnalysisResult(
                path=str(file_path),
                complexity=0,
                volume=0.0,
                difficulty=0.0,
                effort=0.0,
                mi=0.0,
                error=str(e),
            )

    def analyze_files(self, file_paths: list[Path]) -> list[FileAnalysisResult]:
        """Analyze multiple source files sequentially.

        Args:
            file_paths: List of paths to analyze.

        Returns:
            List of FileAnalysisResult, one per file.
        """
        return [self.analyze_file(fp) for fp in file_paths]

    def is_supported(self, file_path: Path) -> bool:
        """Check if a file extension is supported.

        Args:
            file_path: Path to check.

        Returns:
            True if the file extension is supported.
        """
        return file_path.suffix.lower() in SUPPORTED_EXTENSIONS
