"""Cognitive complexity analysis using radon."""

from pathlib import Path
from typing import Any

from slopometry.core.models import ComplexityDelta, ComplexityMetrics, ExtendedComplexityMetrics
from slopometry.core.python_feature_analyzer import PythonFeatureAnalyzer

CALCULATOR_VERSION = "2024.1.4"


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

        except Exception:
            current_metrics = self._analyze_directory(self.working_directory)
            return current_metrics, ComplexityDelta()

    def _analyze_directory(self, directory: Path) -> ComplexityMetrics:
        """Analyze complexity of Python files in a specific directory.

        Args:
            directory: Directory to analyze

        Returns:
            ComplexityMetrics with aggregated complexity data.
        """
        import radon.complexity as cc_lib

        from slopometry.core.git_tracker import GitTracker

        tracker = GitTracker(directory)
        python_files = tracker.get_tracked_python_files()

        files_by_complexity = {}
        all_complexities = []

        for file_path in python_files:
            try:
                content = file_path.read_text(encoding="utf-8")

                blocks = cc_lib.cc_visit(content)
                file_complexity = sum(block.complexity for block in blocks)

                relative_path = self._get_relative_path(file_path, directory)
                files_by_complexity[relative_path] = file_complexity
                all_complexities.append(file_complexity)

            except (SyntaxError, UnicodeDecodeError, OSError):
                # Note: radon cc_visit can raise SyntaxError on invalid python code
                # We track these as errors just like the CLI parser did
                # But ComplexityMetrics does not expose parse_errors dict in this method's return type currently
                # So we just skip or log
                continue

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
        )

    def _process_radon_output(self, radon_data: dict[str, Any], reference_dir: Path | None = None) -> ComplexityMetrics:
        """Process radon JSON output into ComplexityMetrics.

        Args:
            radon_data: Raw JSON data from radon
            reference_dir: Reference directory for path calculation (defaults to working_directory)

        Returns:
            Processed ComplexityMetrics
        """
        files_by_complexity = {}
        all_complexities = []

        reference_directory = reference_dir or self.working_directory

        for file_path, functions in radon_data.items():
            if not functions:
                continue

            # FIXME: parse errors due to python version mismatch should be handled
            # as explicit negative case instead of implicit ok
            if isinstance(functions, dict) and "error" in functions:
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
        )

    def _calculate_delta(
        self, baseline_metrics: ComplexityMetrics, current_metrics: ComplexityMetrics
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

        # Calculate global average complexity change (not just common files)
        # This prevents penalizing adding many simple files (which increases total but decreases average)
        avg_complexity_change = current_metrics.average_complexity - baseline_metrics.average_complexity

        delta = ComplexityDelta(
            total_complexity_change=total_complexity_change,
            files_added=files_added,
            files_removed=files_removed,
            files_changed=files_changed,
            net_files_change=len(files_added) - len(files_removed),
            avg_complexity_change=avg_complexity_change,
        )

        # Handle Extended Metrics if available (duck typing)
        if hasattr(current_metrics, "average_effort") and hasattr(baseline_metrics, "average_effort"):
            # We use getattr to safely access extended attributes that might not be on the base class type hint
            delta.avg_effort_change = getattr(current_metrics, "average_effort", 0.0) - getattr(
                baseline_metrics, "average_effort", 0.0
            )
            delta.total_effort_change = getattr(current_metrics, "total_effort", 0.0) - getattr(
                baseline_metrics, "total_effort", 0.0
            )

        if hasattr(current_metrics, "average_mi") and hasattr(baseline_metrics, "average_mi"):
            delta.avg_mi_change = getattr(current_metrics, "average_mi", 0.0) - getattr(
                baseline_metrics, "average_mi", 0.0
            )
            delta.total_mi_change = getattr(current_metrics, "total_mi", 0.0) - getattr(
                baseline_metrics, "total_mi", 0.0
            )

        return delta

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

    def analyze_extended_complexity(self, directory: Path | None = None) -> ExtendedComplexityMetrics:
        """Analyze with CC, Halstead, and MI metrics.

        Args:
            directory: Directory to analyze. Defaults to working_directory.

        Returns:
            ExtendedComplexityMetrics with all radon metrics.
        """
        target_dir = directory or self.working_directory
        import radon.complexity as cc_lib
        import radon.metrics as metrics_lib

        from slopometry.core.git_tracker import GitTracker

        tracker = GitTracker(target_dir)
        python_files = tracker.get_tracked_python_files()

        files_by_complexity = {}
        all_complexities = []
        files_with_parse_errors = {}

        total_volume = 0.0
        total_difficulty = 0.0
        total_effort = 0.0
        hal_file_count = 0
        total_mi = 0.0
        mi_file_count = 0

        for file_path in python_files:
            try:
                content = file_path.read_text(encoding="utf-8")

                blocks = cc_lib.cc_visit(content)
                file_complexity = sum(block.complexity for block in blocks)

                relative_path = self._get_relative_path(str(file_path), target_dir)
                files_by_complexity[relative_path] = file_complexity
                all_complexities.append(file_complexity)

                hal = metrics_lib.h_visit(content)
                total = hal.total
                total_volume += total.volume
                total_difficulty += total.difficulty
                total_effort += total.effort
                hal_file_count += 1

                mi_score = metrics_lib.mi_visit(content, multi=False)
                total_mi += mi_score
                mi_file_count += 1

            except (SyntaxError, UnicodeDecodeError, OSError, ValueError) as e:
                # Note: ValueError handles empty files which radon might complain about
                relative_path = self._get_relative_path(str(file_path), target_dir)
                files_with_parse_errors[relative_path] = str(e)
                continue

        feature_analyzer = PythonFeatureAnalyzer()

        feature_stats = feature_analyzer.analyze_directory(target_dir)

        total_files = len(all_complexities)
        total_complexity = sum(all_complexities)
        average_complexity = total_complexity / total_files if total_files > 0 else 0.0
        max_complexity = max(all_complexities) if all_complexities else 0
        min_complexity = min(all_complexities) if all_complexities else 0

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

        return ExtendedComplexityMetrics(
            total_complexity=total_complexity,
            average_complexity=average_complexity,
            max_complexity=max_complexity,
            min_complexity=min_complexity,
            total_volume=total_volume,
            average_volume=average_volume,
            total_effort=total_effort,
            average_effort=average_effort,
            average_difficulty=average_difficulty,
            total_mi=total_mi,
            average_mi=average_mi,
            type_hint_coverage=type_hint_coverage,
            docstring_coverage=docstring_coverage,
            deprecation_count=feature_stats.deprecations_count,
            total_files_analyzed=total_files,
            files_by_complexity=files_by_complexity,
            files_with_parse_errors=files_with_parse_errors,
        )
