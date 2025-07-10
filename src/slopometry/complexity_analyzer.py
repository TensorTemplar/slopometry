"""Cognitive complexity analysis using radon."""

import json
import subprocess
from pathlib import Path
from typing import Any

from slopometry.models import ComplexityDelta, ComplexityMetrics


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
            result = subprocess.run(
                ["radon", "cc", "--json", "--show-complexity", str(directory)],
                capture_output=True,
                text=True,
                timeout=30,
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
