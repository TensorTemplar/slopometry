"""Cognitive complexity analysis using radon."""

import json
import subprocess
from pathlib import Path
from typing import Any

from slopometry.core.models import ComplexityDelta, ComplexityMetrics, ExtendedComplexityMetrics

RADON_CMD_PREFIX = ["uvx", "--python", "3.13", "radon"]


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
        try:
            # FIXME: python version hardcode is spurious, while we require the latest AST parsing
            # the version we choose should be informed by either config var or repo python
            result = subprocess.run(
                [*RADON_CMD_PREFIX, "cc", "--json", "--show-complexity", str(directory)],
                capture_output=True,
                text=True,
                timeout=90,
            )

            if result.returncode != 0:
                return ComplexityMetrics()

            radon_data = json.loads(result.stdout)

            return self._process_radon_output(radon_data, directory)

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
            return ComplexityMetrics()

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

        if common_files:
            baseline_avg = sum(baseline_metrics.files_by_complexity[f] for f in common_files) / len(common_files)
            current_avg = sum(current_metrics.files_by_complexity[f] for f in common_files) / len(common_files)
            avg_complexity_change = current_avg - baseline_avg
        else:
            avg_complexity_change = 0.0

        return ComplexityDelta(
            total_complexity_change=total_complexity_change,
            files_added=files_added,
            files_removed=files_removed,
            files_changed=files_changed,
            net_files_change=len(files_added) - len(files_removed),
            avg_complexity_change=avg_complexity_change,
        )

    def _get_relative_path(self, file_path: str, reference_dir: Path | None = None) -> str:
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
            return file_path

    def analyze_extended_complexity(self, directory: Path | None = None) -> ExtendedComplexityMetrics:
        """Analyze with CC, Halstead, and MI metrics.

        Args:
            directory: Directory to analyze. Defaults to working_directory.

        Returns:
            ExtendedComplexityMetrics with all radon metrics.
        """
        target_dir = directory or self.working_directory

        try:
            cc_result = subprocess.run(
                [*RADON_CMD_PREFIX, "cc", "--json", "--show-complexity", str(target_dir)],
                capture_output=True,
                text=True,
                timeout=90,
            )

            hal_result = subprocess.run(
                [*RADON_CMD_PREFIX, "hal", "--json", str(target_dir)],
                capture_output=True,
                text=True,
                timeout=90,
            )

            mi_result = subprocess.run(
                [*RADON_CMD_PREFIX, "mi", "--json", str(target_dir)],
                capture_output=True,
                text=True,
                timeout=90,
            )

            if cc_result.returncode != 0 or hal_result.returncode != 0 or mi_result.returncode != 0:
                return ExtendedComplexityMetrics()

            return self._merge_metrics(
                json.loads(cc_result.stdout),
                json.loads(hal_result.stdout),
                json.loads(mi_result.stdout),
                target_dir,
            )

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
            return ExtendedComplexityMetrics()

    def _merge_metrics(
        self,
        cc_data: dict[str, Any],
        hal_data: dict[str, Any],
        mi_data: dict[str, Any],
        reference_dir: Path,
    ) -> ExtendedComplexityMetrics:
        """Merge metrics from different radon commands into ExtendedComplexityMetrics.

        Args:
            cc_data: Cyclomatic complexity data
            hal_data: Halstead metrics data
            mi_data: Maintainability index data
            reference_dir: Reference directory for path calculation

        Returns:
            Merged ExtendedComplexityMetrics
        """
        files_by_complexity = {}
        all_complexities = []
        files_with_parse_errors = {}

        for file_path, functions in cc_data.items():
            if not functions:
                continue

            if isinstance(functions, dict) and "error" in functions:
                relative_path = self._get_relative_path(file_path, reference_dir)
                files_with_parse_errors[relative_path] = functions["error"]
                continue

            file_complexity = sum(func.get("complexity", 0) for func in functions)
            relative_path = self._get_relative_path(file_path, reference_dir)
            files_by_complexity[relative_path] = file_complexity
            all_complexities.append(file_complexity)

        total_files = len(all_complexities)
        total_complexity = sum(all_complexities)
        average_complexity = total_complexity / total_files if total_files > 0 else 0.0
        max_complexity = max(all_complexities) if all_complexities else 0
        min_complexity = min(all_complexities) if all_complexities else 0

        total_volume = 0.0
        total_difficulty = 0.0
        total_effort = 0.0
        hal_file_count = 0

        for file_path, hal_metrics in hal_data.items():
            if isinstance(hal_metrics, dict) and "total" in hal_metrics:
                total_metrics = hal_metrics["total"]
                if total_metrics:
                    total_volume += total_metrics.get("volume", 0.0)
                    total_difficulty += total_metrics.get("difficulty", 0.0)
                    total_effort += total_metrics.get("effort", 0.0)
                    hal_file_count += 1

        average_volume = total_volume / hal_file_count if hal_file_count > 0 else 0.0
        average_difficulty = total_difficulty / hal_file_count if hal_file_count > 0 else 0.0

        total_mi = 0.0
        mi_file_count = 0

        for file_path, mi_value in mi_data.items():
            if isinstance(mi_value, dict) and "mi" in mi_value:
                mi_score = mi_value["mi"]
            elif isinstance(mi_value, int | float):
                mi_score = float(mi_value)
            else:
                continue

            total_mi += mi_score
            mi_file_count += 1

        average_mi = total_mi / mi_file_count if mi_file_count > 0 else 0.0

        return ExtendedComplexityMetrics(
            total_complexity=total_complexity,
            average_complexity=average_complexity,
            max_complexity=max_complexity,
            min_complexity=min_complexity,
            total_volume=total_volume,
            total_difficulty=total_difficulty,
            total_effort=total_effort,
            average_volume=average_volume,
            average_difficulty=average_difficulty,
            total_mi=total_mi,
            average_mi=average_mi,
            total_files_analyzed=total_files,
            files_by_complexity=files_by_complexity,
            files_with_parse_errors=files_with_parse_errors,
        )
