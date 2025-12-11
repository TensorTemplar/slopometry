"""Analyzer for Python-specific language features."""

import ast
from pathlib import Path
from typing import NamedTuple


class FeatureStats(NamedTuple):
    """Container for feature statistics."""

    functions_count: int = 0
    classes_count: int = 0

    # Docstrings
    docstrings_count: int = 0  # Number of functions/classes with docstrings

    # Type Hints
    args_count: int = 0
    annotated_args_count: int = 0
    returns_count: int = 0
    annotated_returns_count: int = 0

    # Deprecations
    deprecations_count: int = 0  # Number of deprecated functions/classes or warnings.warn calls


class PythonFeatureAnalyzer:
    """Analyzes Python files for language feature usage."""

    def analyze_directory(self, directory: Path) -> FeatureStats:
        """Analyze all Python files in directory recursively.

        Args:
            directory: Root directory to search

        Returns:
            Aggregated FeatureStats
        """
        aggregated = FeatureStats()

        # Use GitTracker to find files respecting .gitignore
        from slopometry.core.git_tracker import GitTracker

        tracker = GitTracker(directory)
        python_files = tracker.get_tracked_python_files()

        for file_path in python_files:
            try:
                stats = self._analyze_file(file_path)
                aggregated = self._merge_stats(aggregated, stats)
            except (SyntaxError, UnicodeDecodeError, OSError):
                continue

        return aggregated

    def _analyze_file(self, file_path: Path) -> FeatureStats:
        """Analyze a single Python file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except Exception:
            return FeatureStats()

        visitor = FeatureVisitor()
        visitor.visit(tree)
        return visitor.stats

    def _merge_stats(self, s1: FeatureStats, s2: FeatureStats) -> FeatureStats:
        """Merge two stats objects."""
        return FeatureStats(
            functions_count=s1.functions_count + s2.functions_count,
            classes_count=s1.classes_count + s2.classes_count,
            docstrings_count=s1.docstrings_count + s2.docstrings_count,
            args_count=s1.args_count + s2.args_count,
            annotated_args_count=s1.annotated_args_count + s2.annotated_args_count,
            returns_count=s1.returns_count + s2.returns_count,
            annotated_returns_count=s1.annotated_returns_count + s2.annotated_returns_count,
            deprecations_count=s1.deprecations_count + s2.deprecations_count,
        )


class FeatureVisitor(ast.NodeVisitor):
    """AST visitor to collect feature usage statistics."""

    def __init__(self):
        self.functions = 0
        self.classes = 0
        self.docstrings = 0
        self.args = 0
        self.annotated_args = 0
        self.returns = 0
        self.annotated_returns = 0
        self.deprecations = 0

    @property
    def stats(self) -> FeatureStats:
        return FeatureStats(
            functions_count=self.functions,
            classes_count=self.classes,
            docstrings_count=self.docstrings,
            args_count=self.args,
            annotated_args_count=self.annotated_args,
            returns_count=self.returns,
            annotated_returns_count=self.annotated_returns,
            deprecations_count=self.deprecations,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions += 1

        # Docstring check
        if ast.get_docstring(node):
            self.docstrings += 1

        # Return annotation check
        self.returns += 1
        if node.returns:
            self.annotated_returns += 1

        # Arguments check
        for arg in node.args.args:
            if arg.arg == "self" or arg.arg == "cls":
                continue

            self.args += 1
            if arg.annotation:
                self.annotated_args += 1

        # Check for @deprecated or @warnings.deprecated decorator
        for decorator in node.decorator_list:
            if self._is_deprecated_decorator(decorator):
                self.deprecations += 1

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._visit_func_common(node)
        # Check for @deprecated decorator
        for decorator in node.decorator_list:
            if self._is_deprecated_decorator(decorator):
                self.deprecations += 1
        self.generic_visit(node)

    def _visit_func_common(self, node):
        self.functions += 1

        if ast.get_docstring(node):
            self.docstrings += 1

        self.returns += 1
        if node.returns:
            self.annotated_returns += 1

        for arg in node.args.args:
            if arg.arg == "self" or arg.arg == "cls":
                continue

            self.args += 1
            if arg.annotation:
                self.annotated_args += 1

    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes += 1

        if ast.get_docstring(node):
            self.docstrings += 1

        # Check for @deprecated decorator
        for decorator in node.decorator_list:
            if self._is_deprecated_decorator(decorator):
                self.deprecations += 1

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Check for warnings.warn calls."""
        # Check for coverage of warnings.warn(..., DeprecationWarning)
        if self._is_warnings_warn(node.func):
            if len(node.args) > 1:
                category = node.args[1]
                if self._is_deprecation_warning(category):
                    self.deprecations += 1
            # Check kwargs for category
            for keyword in node.keywords:
                if keyword.arg == "category" and self._is_deprecation_warning(keyword.value):
                    self.deprecations += 1

        self.generic_visit(node)

    def _is_deprecated_decorator(self, node: ast.AST) -> bool:
        """Check if decorator is @deprecated or @warnings.deprecated."""
        if isinstance(node, ast.Name):
            return node.id == "deprecated"
        elif isinstance(node, ast.Attribute):
            # Check for warnings.deprecated, typing.deprecated, typing_extensions.deprecated
            if node.attr == "deprecated":
                if isinstance(node.value, ast.Name):
                    return node.value.id in ("warnings", "typing", "typing_extensions")
            return False
        elif isinstance(node, ast.Call):
            return self._is_deprecated_decorator(node.func)
        return False

    def _is_warnings_warn(self, node: ast.AST) -> bool:
        """Check if call is warnings.warn."""
        if isinstance(node, ast.Attribute):
            return node.attr == "warn" and (isinstance(node.value, ast.Name) and node.value.id == "warnings")
        return False

    def _is_deprecation_warning(self, node: ast.AST) -> bool:
        """Check if node represents DeprecationWarning."""
        if isinstance(node, ast.Name):
            return "DeprecationWarning" in node.id or "PendingDeprecationWarning" in node.id
        elif isinstance(node, ast.Attribute):
            return "DeprecationWarning" in node.attr or "PendingDeprecationWarning" in node.attr
        return False
