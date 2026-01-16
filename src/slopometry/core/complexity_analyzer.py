"""Cognitive complexity analysis using radon."""

import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from slopometry.core.models import (
    ComplexityDelta,
    ComplexityMetrics,
    ExtendedComplexityMetrics,
    FileAnalysisResult,
)
from slopometry.core.python_feature_analyzer import PythonFeatureAnalyzer
from slopometry.core.settings import settings

logger = logging.getLogger(__name__)

CALCULATOR_VERSION = "2024.1.4"


def _get_tiktoken_encoder() -> Any:
    """Get tiktoken encoder, falling back if o200k_base encoding not available.

    Returns:
        tiktoken Encoder for token counting
    """
    import tiktoken

    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception as e:
        logger.debug(f"Falling back to cl100k_base encoding: {e}")
        return tiktoken.get_encoding("cl100k_base")


def _analyze_single_file_extended(file_path: Path) -> FileAnalysisResult | None:
    """Analyze a single Python file for all metrics.

    Module-level function required for ProcessPoolExecutor pickling.
    Imports are inside function to avoid serialization issues.
    """
    import radon.complexity as cc_lib
    import radon.metrics as metrics_lib

    encoder = _get_tiktoken_encoder()

    try:
        content = file_path.read_text(encoding="utf-8")

        # Suppress SyntaxWarning from radon parsing third-party code with invalid escapes
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            blocks = cc_lib.cc_visit(content)
            complexity = sum(block.complexity for block in blocks)

            hal = metrics_lib.h_visit(content)
            mi = metrics_lib.mi_visit(content, multi=False)
        tokens = len(encoder.encode(content, disallowed_special=()))

        return FileAnalysisResult(
            path=str(file_path),
            complexity=complexity,
            volume=hal.total.volume,
            difficulty=hal.total.difficulty,
            effort=hal.total.effort,
            mi=mi,
            tokens=tokens,
        )
    except (SyntaxError, UnicodeDecodeError, OSError, ValueError) as e:
        return FileAnalysisResult(
            path=str(file_path),
            complexity=0,
            volume=0.0,
            difficulty=0.0,
            effort=0.0,
            mi=0.0,
            tokens=0,
            error=str(e),
        )


class ComplexityAnalyzer:
    """Analyzes cognitive complexity of Python files using radon."""

    def __init__(self, working_directory: Path | None = None):
        """Initialize the analyzer.

        Args:
            working_directory: Directory to analyze. Defaults to current working directory.
        """
        self.working_directory = working_directory or Path.cwd()

    def analyze_complexity(self) -> ComplexityMetrics:
        """Analyze complexity of Python files in the working directory.

        Returns:
            ComplexityMetrics with aggregated complexity data.
        """
        return self._analyze_directory(self.working_directory)

    def analyze_complexity_with_baseline(self, baseline_dir: Path) -> tuple[ComplexityMetrics, ComplexityDelta]:
        """Analyze complexity and compare with baseline from previous commit.

        Args:
            baseline_dir: Directory containing Python files from previous commit

        Returns:
            Tuple of (current_metrics, complexity_delta)
        """
        try:
            current_metrics = self._analyze_directory(self.working_directory)

            baseline_metrics = self._analyze_directory(baseline_dir)

            delta = self._calculate_delta(baseline_metrics, current_metrics)

            return current_metrics, delta

        except Exception as e:
            logger.debug(f"Baseline complexity analysis failed, returning current metrics only: {e}")
            current_metrics = self._analyze_directory(self.working_directory)
            return current_metrics, ComplexityDelta()

    def _analyze_directory(self, directory: Path) -> ComplexityMetrics:
        """Analyze complexity of Python files in a specific directory.

        Files with syntax errors or encoding issues are silently skipped.

        Args:
            directory: Directory to analyze

        Returns:
            ComplexityMetrics with aggregated complexity data.
        """
        import radon.complexity as cc_lib

        from slopometry.core.git_tracker import GitTracker

        tracker = GitTracker(directory)
        python_files = tracker.get_tracked_python_files()

        encoder = _get_tiktoken_encoder()

        files_by_complexity = {}
        all_complexities = []

        files_by_token_count = {}
        all_token_counts = []

        for file_path in python_files:
            if not file_path.exists():
                continue

            try:
                content = file_path.read_text(encoding="utf-8")

                # Suppress SyntaxWarning from radon parsing third-party code
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SyntaxWarning)
                    blocks = cc_lib.cc_visit(content)
                file_complexity = sum(block.complexity for block in blocks)

                relative_path = self._get_relative_path(file_path, directory)
                files_by_complexity[relative_path] = file_complexity
                all_complexities.append(file_complexity)

                token_count = len(encoder.encode(content, disallowed_special=()))
                files_by_token_count[relative_path] = token_count
                all_token_counts.append(token_count)

            except (SyntaxError, UnicodeDecodeError, OSError) as e:
                logger.debug(f"Skipping unparseable file {relative_path}: {e}")
                continue

        total_files = len(all_complexities)
        total_complexity = sum(all_complexities)

        total_tokens = sum(all_token_counts)

        if total_files > 0:
            average_complexity = total_complexity / total_files
            max_complexity = max(all_complexities)
            min_complexity = min(all_complexities)

            average_tokens = total_tokens / total_files
            max_tokens = max(all_token_counts)
            min_tokens = min(all_token_counts)
        else:
            average_complexity = 0.0
            max_complexity = 0
            min_complexity = 0

            average_tokens = 0.0
            max_tokens = 0
            min_tokens = 0

        return ComplexityMetrics(
            total_files_analyzed=total_files,
            total_complexity=total_complexity,
            average_complexity=average_complexity,
            max_complexity=max_complexity,
            min_complexity=min_complexity,
            files_by_complexity=files_by_complexity,
            total_tokens=total_tokens,
            average_tokens=average_tokens,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            files_by_token_count=files_by_token_count,
        )

    def _process_radon_output(self, radon_data: dict[str, Any], reference_dir: Path | None = None) -> ComplexityMetrics:
        """Process radon JSON output into ComplexityMetrics.

        Files with parse errors (e.g., Python version mismatch) are tracked separately.

        Args:
            radon_data: Raw JSON data from radon
            reference_dir: Reference directory for path calculation (defaults to working_directory)

        Returns:
            Processed ComplexityMetrics
        """
        files_by_complexity = {}
        all_complexities = []
        files_with_parse_errors: dict[str, str] = {}

        reference_directory = reference_dir or self.working_directory

        for file_path, functions in radon_data.items():
            if not functions:
                continue

            if isinstance(functions, dict) and "error" in functions:
                relative_path = self._get_relative_path(file_path, reference_directory)
                files_with_parse_errors[relative_path] = functions.get("error", "Unknown parse error")
                continue

            file_complexity = sum(func.get("complexity", 0) for func in functions)

            relative_path = self._get_relative_path(file_path, reference_directory)
            files_by_complexity[relative_path] = file_complexity
            all_complexities.append(file_complexity)

        total_files = len(all_complexities)
        total_complexity = sum(all_complexities)

        if total_files > 0:
            average_complexity = total_complexity / total_files
            max_complexity = max(all_complexities)
            min_complexity = min(all_complexities)
        else:
            average_complexity = 0.0
            max_complexity = 0
            min_complexity = 0

        return ComplexityMetrics(
            total_files_analyzed=total_files,
            total_complexity=total_complexity,
            average_complexity=average_complexity,
            max_complexity=max_complexity,
            min_complexity=min_complexity,
            files_by_complexity=files_by_complexity,
            files_with_parse_errors=files_with_parse_errors,
        )

    def _calculate_delta(
        self,
        baseline_metrics: ComplexityMetrics | ExtendedComplexityMetrics,
        current_metrics: ComplexityMetrics | ExtendedComplexityMetrics,
    ) -> ComplexityDelta:
        """Calculate complexity delta between baseline and current metrics.

        Args:
            baseline_metrics: Complexity metrics from previous commit
            current_metrics: Complexity metrics from current state

        Returns:
            ComplexityDelta showing changes
        """
        baseline_files = set(baseline_metrics.files_by_complexity.keys())
        current_files = set(current_metrics.files_by_complexity.keys())

        files_added = list(current_files - baseline_files)
        files_removed = list(baseline_files - current_files)

        common_files = baseline_files & current_files
        files_changed = {}

        for file_path in common_files:
            baseline_complexity = baseline_metrics.files_by_complexity[file_path]
            current_complexity = current_metrics.files_by_complexity[file_path]
            complexity_change = current_complexity - baseline_complexity

            if complexity_change != 0:
                files_changed[file_path] = complexity_change

        total_complexity_change = current_metrics.total_complexity - baseline_metrics.total_complexity
        avg_complexity_change = current_metrics.average_complexity - baseline_metrics.average_complexity

        delta = ComplexityDelta(
            total_complexity_change=total_complexity_change,
            files_added=files_added,
            files_removed=files_removed,
            files_changed=files_changed,
            net_files_change=len(files_added) - len(files_removed),
            avg_complexity_change=avg_complexity_change,
        )

        if isinstance(current_metrics, ExtendedComplexityMetrics) and isinstance(
            baseline_metrics, ExtendedComplexityMetrics
        ):
            delta.avg_effort_change = current_metrics.average_effort - baseline_metrics.average_effort
            delta.total_effort_change = current_metrics.total_effort - baseline_metrics.total_effort
            delta.avg_mi_change = current_metrics.average_mi - baseline_metrics.average_mi
            delta.total_mi_change = current_metrics.total_mi - baseline_metrics.total_mi

            delta.total_tokens_change = current_metrics.total_tokens - baseline_metrics.total_tokens
            delta.avg_tokens_change = current_metrics.average_tokens - baseline_metrics.average_tokens

            delta.type_hint_coverage_change = current_metrics.type_hint_coverage - baseline_metrics.type_hint_coverage
            delta.docstring_coverage_change = current_metrics.docstring_coverage - baseline_metrics.docstring_coverage
            delta.deprecation_change = current_metrics.deprecation_count - baseline_metrics.deprecation_count

            delta.any_type_percentage_change = (
                current_metrics.any_type_percentage - baseline_metrics.any_type_percentage
            )
            delta.str_type_percentage_change = (
                current_metrics.str_type_percentage - baseline_metrics.str_type_percentage
            )

            delta.orphan_comment_change = current_metrics.orphan_comment_count - baseline_metrics.orphan_comment_count
            delta.untracked_todo_change = current_metrics.untracked_todo_count - baseline_metrics.untracked_todo_count
            delta.inline_import_change = current_metrics.inline_import_count - baseline_metrics.inline_import_count
            delta.dict_get_with_default_change = (
                current_metrics.dict_get_with_default_count - baseline_metrics.dict_get_with_default_count
            )
            delta.hasattr_getattr_change = (
                current_metrics.hasattr_getattr_count - baseline_metrics.hasattr_getattr_count
            )
            delta.nonempty_init_change = current_metrics.nonempty_init_count - baseline_metrics.nonempty_init_count
            delta.test_skip_change = current_metrics.test_skip_count - baseline_metrics.test_skip_count
            delta.swallowed_exception_change = (
                current_metrics.swallowed_exception_count - baseline_metrics.swallowed_exception_count
            )
            delta.type_ignore_change = current_metrics.type_ignore_count - baseline_metrics.type_ignore_count
            delta.dynamic_execution_change = (
                current_metrics.dynamic_execution_count - baseline_metrics.dynamic_execution_count
            )
            delta.single_method_class_change = (
                current_metrics.single_method_class_count - baseline_metrics.single_method_class_count
            )
            delta.deep_inheritance_change = (
                current_metrics.deep_inheritance_count - baseline_metrics.deep_inheritance_count
            )
            delta.passthrough_wrapper_change = (
                current_metrics.passthrough_wrapper_count - baseline_metrics.passthrough_wrapper_count
            )

        return delta

    def _build_files_by_loc(self, python_files: list[Path], target_dir: Path) -> dict[str, int]:
        """Build mapping of file path to code LOC for file filtering.

        Args:
            python_files: List of Python files to analyze
            target_dir: Target directory for relative path calculation

        Returns:
            Dict mapping relative file paths to their code LOC
        """
        from slopometry.core.python_feature_analyzer import _count_loc

        files_by_loc: dict[str, int] = {}
        for file_path in python_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                _, code_loc = _count_loc(content)
                relative_path = self._get_relative_path(file_path, target_dir)
                files_by_loc[relative_path] = code_loc
            except (OSError, UnicodeDecodeError):
                continue
        return files_by_loc

    def _get_relative_path(self, file_path: str | Path, reference_dir: Path | None = None) -> str:
        """Convert absolute path to relative path from reference directory.

        Args:
            file_path: Absolute file path
            reference_dir: Reference directory (defaults to working_directory)

        Returns:
            Relative path string
        """
        try:
            abs_path = Path(file_path).resolve()
            ref_dir = (reference_dir or self.working_directory).resolve()

            if abs_path.is_relative_to(ref_dir):
                return str(abs_path.relative_to(ref_dir))
            else:
                return str(abs_path)
        except (ValueError, OSError):
            return str(file_path)

    def _analyze_files_parallel(
        self, files: list[Path], max_workers: int | None = None
    ) -> list[FileAnalysisResult | None]:
        """Analyze files in parallel using ProcessPoolExecutor.

        Args:
            files: List of Python file paths to analyze
            max_workers: Maximum number of worker processes (default from settings)

        Returns:
            List of FileAnalysisResult objects
        """
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, settings.max_parallel_workers)

        results: list[FileAnalysisResult | None] = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_analyze_single_file_extended, fp): fp for fp in files}

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    file_path = futures[future]
                    logger.warning(f"Failed to analyze {file_path}: {e}")
                    results.append(None)

        return results

    def analyze_extended_complexity(self, directory: Path | None = None) -> ExtendedComplexityMetrics:
        """Analyze with CC, Halstead, and MI metrics.

        Args:
            directory: Directory to analyze. Defaults to working_directory.

        Returns:
            ExtendedComplexityMetrics with all radon metrics.
        """
        target_dir = directory or self.working_directory

        from slopometry.core.git_tracker import GitTracker

        tracker = GitTracker(target_dir)
        python_files = [f for f in tracker.get_tracked_python_files() if f.exists()]

        start_total = time.perf_counter()

        if len(python_files) >= settings.parallel_file_threshold:
            results = self._analyze_files_parallel(python_files)
        else:
            results = [_analyze_single_file_extended(fp) for fp in python_files]

        files_by_complexity: dict[str, int] = {}
        files_by_effort: dict[str, float] = {}
        all_complexities: list[int] = []
        files_with_parse_errors: dict[str, str] = {}
        files_by_token_count: dict[str, int] = {}
        all_token_counts: list[int] = []

        total_volume = 0.0
        total_difficulty = 0.0
        total_effort = 0.0
        hal_file_count = 0
        total_mi = 0.0
        mi_file_count = 0

        for result in results:
            if result is None:
                continue

            relative_path = self._get_relative_path(result.path, target_dir)

            if result.error:
                files_with_parse_errors[relative_path] = result.error
                continue

            files_by_complexity[relative_path] = result.complexity
            files_by_effort[relative_path] = result.effort
            all_complexities.append(result.complexity)

            total_volume += result.volume
            total_difficulty += result.difficulty
            total_effort += result.effort
            hal_file_count += 1

            total_mi += result.mi
            mi_file_count += 1

            files_by_token_count[relative_path] = result.tokens
            all_token_counts.append(result.tokens)

        elapsed_total = time.perf_counter() - start_total
        mode = "parallel" if len(python_files) >= settings.parallel_file_threshold else "sequential"
        logger.debug(f"Complexity analysis ({mode}): {len(python_files)} files in {elapsed_total:.2f}s")

        feature_analyzer = PythonFeatureAnalyzer()

        feature_stats = feature_analyzer.analyze_directory(target_dir)

        total_files = len(all_complexities)
        total_complexity = sum(all_complexities)
        average_complexity = total_complexity / total_files if total_files > 0 else 0.0
        max_complexity = max(all_complexities) if all_complexities else 0
        min_complexity = min(all_complexities) if all_complexities else 0

        total_tokens = sum(all_token_counts)
        average_tokens = total_tokens / total_files if total_files > 0 else 0.0
        max_tokens = max(all_token_counts) if all_token_counts else 0
        min_tokens = min(all_token_counts) if all_token_counts else 0

        average_volume = total_volume / hal_file_count if hal_file_count > 0 else 0.0
        average_difficulty = total_difficulty / hal_file_count if hal_file_count > 0 else 0.0
        average_effort = total_effort / hal_file_count if hal_file_count > 0 else 0.0

        average_mi = total_mi / mi_file_count if mi_file_count > 0 else 0.0

        total_typeable_items = feature_stats.args_count + feature_stats.returns_count
        total_annotated = feature_stats.annotated_args_count + feature_stats.annotated_returns_count
        type_hint_coverage = (total_annotated / total_typeable_items * 100.0) if total_typeable_items > 0 else 0.0

        total_docstringable = feature_stats.functions_count + feature_stats.classes_count
        docstring_coverage = (
            (feature_stats.docstrings_count / total_docstringable * 100.0) if total_docstringable > 0 else 0.0
        )

        total_type_refs = feature_stats.total_type_references
        any_type_percentage = (feature_stats.any_type_count / total_type_refs * 100.0) if total_type_refs > 0 else 0.0
        str_type_percentage = (feature_stats.str_type_count / total_type_refs * 100.0) if total_type_refs > 0 else 0.0

        return ExtendedComplexityMetrics(
            total_complexity=total_complexity,
            average_complexity=average_complexity,
            max_complexity=max_complexity,
            min_complexity=min_complexity,
            total_volume=total_volume,
            average_volume=average_volume,
            total_effort=total_effort,
            average_effort=average_effort,
            total_difficulty=total_difficulty,
            average_difficulty=average_difficulty,
            total_mi=total_mi,
            average_mi=average_mi,
            total_tokens=total_tokens,
            average_tokens=average_tokens,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            files_by_token_count=files_by_token_count,
            type_hint_coverage=type_hint_coverage,
            docstring_coverage=docstring_coverage,
            deprecation_count=feature_stats.deprecations_count,
            any_type_percentage=any_type_percentage,
            str_type_percentage=str_type_percentage,
            total_files_analyzed=total_files,
            files_by_complexity=files_by_complexity,
            files_by_effort=files_by_effort,
            files_with_parse_errors=files_with_parse_errors,
            orphan_comment_count=feature_stats.orphan_comment_count,
            untracked_todo_count=feature_stats.untracked_todo_count,
            inline_import_count=feature_stats.inline_import_count,
            dict_get_with_default_count=feature_stats.dict_get_with_default_count,
            hasattr_getattr_count=feature_stats.hasattr_getattr_count,
            nonempty_init_count=feature_stats.nonempty_init_count,
            test_skip_count=feature_stats.test_skip_count,
            swallowed_exception_count=feature_stats.swallowed_exception_count,
            type_ignore_count=feature_stats.type_ignore_count,
            dynamic_execution_count=feature_stats.dynamic_execution_count,
            orphan_comment_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.orphan_comment_files]
            ),
            untracked_todo_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.untracked_todo_files]
            ),
            inline_import_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.inline_import_files]
            ),
            dict_get_with_default_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.dict_get_with_default_files]
            ),
            hasattr_getattr_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.hasattr_getattr_files]
            ),
            nonempty_init_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.nonempty_init_files]
            ),
            test_skip_files=sorted([self._get_relative_path(p, target_dir) for p in feature_stats.test_skip_files]),
            swallowed_exception_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.swallowed_exception_files]
            ),
            type_ignore_files=sorted([self._get_relative_path(p, target_dir) for p in feature_stats.type_ignore_files]),
            dynamic_execution_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.dynamic_execution_files]
            ),
            single_method_class_count=feature_stats.single_method_class_count,
            deep_inheritance_count=feature_stats.deep_inheritance_count,
            passthrough_wrapper_count=feature_stats.passthrough_wrapper_count,
            single_method_class_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.single_method_class_files]
            ),
            deep_inheritance_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.deep_inheritance_files]
            ),
            passthrough_wrapper_files=sorted(
                [self._get_relative_path(p, target_dir) for p in feature_stats.passthrough_wrapper_files]
            ),
            total_loc=feature_stats.total_loc,
            code_loc=feature_stats.code_loc,
            files_by_loc={
                self._get_relative_path(p, target_dir): loc
                for p, loc in self._build_files_by_loc(python_files, target_dir).items()
            },
        )
