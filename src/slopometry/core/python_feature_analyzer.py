"""Analyzer for Python-specific language features."""

import ast
import io
import re
import tokenize
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

    # Type Reference Tracking (for detecting overly generic types)
    total_type_references: int = 0  # Total type names found in annotations
    any_type_count: int = 0  # Count of 'Any' type references
    str_type_count: int = 0  # Count of 'str' type references

    # Deprecations
    deprecations_count: int = 0  # Number of deprecated functions/classes or warnings.warn calls

    # Code Smells
    orphan_comment_count: int = 0  # Comments outside docstrings that aren't TODOs or explanatory URLs
    untracked_todo_count: int = 0  # TODO comments without ticket references (JIRA-123, #123) or URLs
    inline_import_count: int = 0  # Import statements not at module level (excluding TYPE_CHECKING guards)


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
        ast_stats = visitor.stats

        # Analyze comments (not in AST)
        orphan_comments, untracked_todos = self._analyze_comments(content)

        return FeatureStats(
            functions_count=ast_stats.functions_count,
            classes_count=ast_stats.classes_count,
            docstrings_count=ast_stats.docstrings_count,
            args_count=ast_stats.args_count,
            annotated_args_count=ast_stats.annotated_args_count,
            returns_count=ast_stats.returns_count,
            annotated_returns_count=ast_stats.annotated_returns_count,
            total_type_references=ast_stats.total_type_references,
            any_type_count=ast_stats.any_type_count,
            str_type_count=ast_stats.str_type_count,
            deprecations_count=ast_stats.deprecations_count,
            orphan_comment_count=orphan_comments,
            untracked_todo_count=untracked_todos,
            inline_import_count=ast_stats.inline_import_count,
        )

    def _analyze_comments(self, content: str) -> tuple[int, int]:
        """Analyze comments in source code using tokenize.

        Returns:
            Tuple of (orphan_comment_count, untracked_todo_count)
        """
        orphan_comments = 0
        untracked_todos = 0

        # Patterns for detection
        todo_pattern = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
        url_pattern = re.compile(r"https?://")
        ticket_pattern = re.compile(r"([A-Z]+-\d+|#\d+)")

        try:
            tokens = tokenize.generate_tokens(io.StringIO(content).readline)
            for tok in tokens:
                if tok.type == tokenize.COMMENT:
                    comment_text = tok.string

                    is_todo = bool(todo_pattern.search(comment_text))
                    has_url = bool(url_pattern.search(comment_text))

                    if is_todo:
                        # Check if it has a ticket reference or URL
                        has_ticket = bool(ticket_pattern.search(comment_text))
                        if not has_ticket and not has_url:
                            untracked_todos += 1
                    elif not has_url:
                        # Not a TODO and no URL - it's an orphan comment
                        orphan_comments += 1
        except tokenize.TokenizeError:
            # If tokenization fails, we just skip comment analysis
            pass

        return orphan_comments, untracked_todos

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
            total_type_references=s1.total_type_references + s2.total_type_references,
            any_type_count=s1.any_type_count + s2.any_type_count,
            str_type_count=s1.str_type_count + s2.str_type_count,
            deprecations_count=s1.deprecations_count + s2.deprecations_count,
            orphan_comment_count=s1.orphan_comment_count + s2.orphan_comment_count,
            untracked_todo_count=s1.untracked_todo_count + s2.untracked_todo_count,
            inline_import_count=s1.inline_import_count + s2.inline_import_count,
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
        # Type reference tracking
        self.total_type_refs = 0
        self.any_type_refs = 0
        self.str_type_refs = 0
        # Inline import tracking
        self.inline_imports = 0
        self._in_type_checking_block = False
        self._scope_depth = 0  # Track nesting level (0 = module level)

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
            total_type_references=self.total_type_refs,
            any_type_count=self.any_type_refs,
            str_type_count=self.str_type_refs,
            deprecations_count=self.deprecations,
            inline_import_count=self.inline_imports,
        )

    def _collect_type_names(self, node: ast.AST | None) -> None:
        """Recursively collect type names from an annotation node.

        Handles various annotation patterns:
        - ast.Name: simple types like int, str, Any
        - ast.Subscript: generic types like list[str], dict[str, Any]
        - ast.BinOp: union types like str | None (Python 3.10+)
        - ast.Tuple: for dict key-value pairs
        - ast.Constant: string annotations like "SomeClass"
        - ast.Attribute: qualified names like typing.Any
        """
        if node is None:
            return

        if isinstance(node, ast.Name):
            self.total_type_refs += 1
            if node.id == "Any":
                self.any_type_refs += 1
            elif node.id == "str":
                self.str_type_refs += 1

        elif isinstance(node, ast.Attribute):
            # Handle typing.Any, typing.Optional, etc.
            self.total_type_refs += 1
            if node.attr == "Any":
                self.any_type_refs += 1
            elif node.attr == "str":
                self.str_type_refs += 1

        elif isinstance(node, ast.Subscript):
            # Handle generic types like list[str], dict[str, Any], Optional[int]
            self._collect_type_names(node.value)  # The generic type itself
            self._collect_type_names(node.slice)  # The type parameter(s)

        elif isinstance(node, ast.BinOp):
            # Handle union types: str | None (Python 3.10+)
            self._collect_type_names(node.left)
            self._collect_type_names(node.right)

        elif isinstance(node, ast.Tuple):
            # Handle multiple type params: dict[str, int] -> (str, int)
            for elt in node.elts:
                self._collect_type_names(elt)

        elif isinstance(node, ast.Constant):
            # Handle string annotations like "ForwardRef"
            if isinstance(node.value, str):
                self.total_type_refs += 1
                if node.value == "Any":
                    self.any_type_refs += 1
                elif node.value == "str":
                    self.str_type_refs += 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions += 1

        # Docstring check
        if ast.get_docstring(node):
            self.docstrings += 1

        # Return annotation check
        self.returns += 1
        if node.returns:
            self.annotated_returns += 1
            self._collect_type_names(node.returns)

        # Arguments check
        for arg in node.args.args:
            if arg.arg == "self" or arg.arg == "cls":
                continue

            self.args += 1
            if arg.annotation:
                self.annotated_args += 1
                self._collect_type_names(arg.annotation)

        # Check for @deprecated or @warnings.deprecated decorator
        for decorator in node.decorator_list:
            if self._is_deprecated_decorator(decorator):
                self.deprecations += 1

        # Track scope for inline import detection
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func_common(node)
        # Check for @deprecated decorator
        for decorator in node.decorator_list:
            if self._is_deprecated_decorator(decorator):
                self.deprecations += 1
        # Track scope for inline import detection
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def _visit_func_common(self, node):
        self.functions += 1

        if ast.get_docstring(node):
            self.docstrings += 1

        self.returns += 1
        if node.returns:
            self.annotated_returns += 1
            self._collect_type_names(node.returns)

        for arg in node.args.args:
            if arg.arg == "self" or arg.arg == "cls":
                continue

            self.args += 1
            if arg.annotation:
                self.annotated_args += 1
                self._collect_type_names(arg.annotation)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes += 1

        if ast.get_docstring(node):
            self.docstrings += 1

        # Check for @deprecated decorator
        for decorator in node.decorator_list:
            if self._is_deprecated_decorator(decorator):
                self.deprecations += 1

        # Track scope for inline import detection
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

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

    def visit_If(self, node: ast.If) -> None:
        """Track TYPE_CHECKING blocks to exclude their imports."""
        if self._is_type_checking_guard(node):
            self._in_type_checking_block = True
            self.generic_visit(node)
            self._in_type_checking_block = False
        else:
            self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Track inline imports (not at module level, not in TYPE_CHECKING)."""
        if self._scope_depth > 0 and not self._in_type_checking_block:
            self.inline_imports += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track inline imports (not at module level, not in TYPE_CHECKING)."""
        if self._scope_depth > 0 and not self._in_type_checking_block:
            self.inline_imports += 1
        self.generic_visit(node)

    def _is_type_checking_guard(self, node: ast.If) -> bool:
        """Check if this is an `if TYPE_CHECKING:` block."""
        if isinstance(node.test, ast.Name):
            return node.test.id == "TYPE_CHECKING"
        if isinstance(node.test, ast.Attribute):
            return node.test.attr == "TYPE_CHECKING"
        return False
