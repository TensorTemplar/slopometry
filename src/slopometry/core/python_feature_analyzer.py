"""Analyzer for Python-specific language features."""

import ast
import io
import logging
import os
import re
import time
import tokenize
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from slopometry.core.models import SmellField
from slopometry.core.settings import settings

logger = logging.getLogger(__name__)


class FeatureStats(BaseModel):
    """Container for feature statistics with actionable guidance in Field descriptions."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Basic counts (no actionable message needed)
    functions_count: int = 0
    classes_count: int = 0
    docstrings_count: int = 0
    args_count: int = 0
    annotated_args_count: int = 0
    returns_count: int = 0
    annotated_returns_count: int = 0
    total_type_references: int = 0
    any_type_count: int = 0
    str_type_count: int = 0
    deprecations_count: int = 0

    # Code smells - self-describing via SmellField metadata
    orphan_comment_count: int = SmellField(
        label="Orphan Comments",
        files_field="orphan_comment_files",
        guidance="Make sure inline code comments add meaningful information about non-obvious design tradeoffs or explain tech debt or performance implications. Consider if these could be docstrings or field descriptors instead",
    )
    untracked_todo_count: int = SmellField(
        label="Untracked TODOs",
        files_field="untracked_todo_files",
        guidance="Untracked TODOs should include ticket references (JIRA-123, #123) or URLs",
    )
    inline_import_count: int = SmellField(
        label="Inline Imports",
        files_field="inline_import_files",
        guidance="Verify if these can be moved to the top of the file (except TYPE_CHECKING guards)",
    )
    dict_get_with_default_count: int = SmellField(
        label="Modeling Gaps (.get() defaults)",
        files_field="dict_get_with_default_files",
        guidance="Consider if these are justified or just indicate modeling gaps - replace with Pydantic BaseSettings or BaseModel with narrower field types or raise explicit errors instead",
    )
    hasattr_getattr_count: int = SmellField(
        label="Modeling Gaps (hasattr/getattr)",
        files_field="hasattr_getattr_files",
        guidance="Consider if these indicate missing domain models - replace with Pydantic BaseModel objects with explicit fields or raise explicit errors instead",
    )
    nonempty_init_count: int = SmellField(
        label="Logic in __init__.py",
        files_field="nonempty_init_files",
        guidance="Consider if implementation code should be moved out of __init__.py files",
    )
    test_skip_count: int = SmellField(
        label="Test Skips",
        files_field="test_skip_files",
        guidance="Request explicit user feedback on whether these test skips are still valid or should be addressed",
    )
    swallowed_exception_count: int = SmellField(
        label="Swallowed Exceptions",
        files_field="swallowed_exception_files",
        guidance="Request explicit user feedback on all swallowed exceptions. Make sure to always log or print the expectation mismatch",
    )
    type_ignore_count: int = SmellField(
        label="Type Ignores",
        files_field="type_ignore_files",
        guidance="Review type: ignore comments - consider fixing the underlying type issue",
    )
    dynamic_execution_count: int = SmellField(
        label="Dynamic Execution",
        files_field="dynamic_execution_files",
        guidance="Review usage of eval/exec/compile - ensure this is necessary and secure",
    )
    single_method_class_count: int = SmellField(
        label="Single-Method Classes",
        files_field="single_method_class_files",
        guidance="Consider using a function instead of a class with only one method besides __init__",
    )
    deep_inheritance_count: int = SmellField(
        label="Deep Inheritance",
        files_field="deep_inheritance_files",
        guidance="Prefer composition over inheritance; >2 base classes increases complexity",
    )
    passthrough_wrapper_count: int = SmellField(
        label="Pass-Through Wrappers",
        files_field="passthrough_wrapper_files",
        guidance="Function that just delegates to another with same args; consider removing indirection",
    )

    # LOC metrics (for file filtering)
    total_loc: int = Field(default=0, description="Total lines of code")
    code_loc: int = Field(default=0, description="Non-blank, non-comment lines")

    # File tracking
    orphan_comment_files: set[str] = Field(default_factory=set)
    untracked_todo_files: set[str] = Field(default_factory=set)
    inline_import_files: set[str] = Field(default_factory=set)
    dict_get_with_default_files: set[str] = Field(default_factory=set)
    hasattr_getattr_files: set[str] = Field(default_factory=set)
    nonempty_init_files: set[str] = Field(default_factory=set)
    test_skip_files: set[str] = Field(default_factory=set)
    swallowed_exception_files: set[str] = Field(default_factory=set)
    type_ignore_files: set[str] = Field(default_factory=set)
    dynamic_execution_files: set[str] = Field(default_factory=set)
    single_method_class_files: set[str] = Field(default_factory=set)
    deep_inheritance_files: set[str] = Field(default_factory=set)
    passthrough_wrapper_files: set[str] = Field(default_factory=set)


def _count_loc(content: str) -> tuple[int, int]:
    """Count total lines and code lines (non-blank, non-comment).

    Returns:
        Tuple of (total_loc, code_loc)
    """
    lines = content.splitlines()
    total = len(lines)
    code = sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
    return total, code


def _analyze_single_file_features(file_path: Path) -> FeatureStats | None:
    """Analyze a single Python file for feature statistics.

    Module-level function required for ProcessPoolExecutor pickling.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
    except Exception as e:
        logger.debug(f"Skipping unparseable file {file_path}: {e}")
        return None

    visitor = FeatureVisitor()
    visitor.visit(tree)
    ast_stats = visitor.stats

    is_test_file = file_path.name.startswith("test_") or "/tests/" in str(file_path)
    orphan_comments, untracked_todos, type_ignores = _analyze_comments_standalone(content, is_test_file)
    nonempty_init = 1 if _is_nonempty_init_standalone(file_path, tree) else 0
    total_loc, code_loc = _count_loc(content)
    path_str = str(file_path)

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
        dict_get_with_default_count=ast_stats.dict_get_with_default_count,
        hasattr_getattr_count=ast_stats.hasattr_getattr_count,
        nonempty_init_count=nonempty_init,
        test_skip_count=ast_stats.test_skip_count,
        swallowed_exception_count=ast_stats.swallowed_exception_count,
        type_ignore_count=type_ignores,
        dynamic_execution_count=ast_stats.dynamic_execution_count,
        single_method_class_count=ast_stats.single_method_class_count,
        deep_inheritance_count=ast_stats.deep_inheritance_count,
        passthrough_wrapper_count=ast_stats.passthrough_wrapper_count,
        total_loc=total_loc,
        code_loc=code_loc,
        orphan_comment_files={path_str} if orphan_comments > 0 else set(),
        untracked_todo_files={path_str} if untracked_todos > 0 else set(),
        inline_import_files={path_str} if ast_stats.inline_import_count > 0 else set(),
        dict_get_with_default_files={path_str} if ast_stats.dict_get_with_default_count > 0 else set(),
        hasattr_getattr_files={path_str} if ast_stats.hasattr_getattr_count > 0 else set(),
        nonempty_init_files={path_str} if nonempty_init > 0 else set(),
        test_skip_files={path_str} if ast_stats.test_skip_count > 0 else set(),
        swallowed_exception_files={path_str} if ast_stats.swallowed_exception_count > 0 else set(),
        type_ignore_files={path_str} if type_ignores > 0 else set(),
        dynamic_execution_files={path_str} if ast_stats.dynamic_execution_count > 0 else set(),
        single_method_class_files={path_str} if ast_stats.single_method_class_count > 0 else set(),
        deep_inheritance_files={path_str} if ast_stats.deep_inheritance_count > 0 else set(),
        passthrough_wrapper_files={path_str} if ast_stats.passthrough_wrapper_count > 0 else set(),
    )


def _analyze_comments_standalone(content: str, is_test_file: bool = False) -> tuple[int, int, int]:
    """Analyze comments in source code using tokenize.

    Standalone version for parallel processing.
    """
    orphan_comments = 0
    untracked_todos = 0
    type_ignores = 0

    todo_pattern = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
    url_pattern = re.compile(r"https?://")
    ticket_pattern = re.compile(r"([A-Z]+-\d+|#\d+)")
    justification_pattern = re.compile(
        r"#\s*(NOTE|REASON|WARNING|WORKAROUND|IMPORTANT|CAVEAT|HACK|NB|PERF|SAFETY|COMPAT):",
        re.IGNORECASE,
    )
    type_ignore_pattern = re.compile(r"#\s*type:\s*ignore")

    try:
        tokens = tokenize.generate_tokens(io.StringIO(content).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                comment_text = tok.string

                is_todo = bool(todo_pattern.search(comment_text))
                has_url = bool(url_pattern.search(comment_text))
                is_justification = bool(justification_pattern.search(comment_text))
                is_type_ignore = bool(type_ignore_pattern.search(comment_text))

                if is_type_ignore:
                    type_ignores += 1
                elif is_todo:
                    has_ticket = bool(ticket_pattern.search(comment_text))
                    if not has_ticket and not has_url:
                        untracked_todos += 1
                elif not has_url and not is_justification and not is_test_file:
                    orphan_comments += 1
    except tokenize.TokenError as e:
        logger.debug(f"Tokenize error during comment analysis: {e}")

    return orphan_comments, untracked_todos, type_ignores


def _is_nonempty_init_standalone(file_path: Path, tree: ast.Module) -> bool:
    """Check if file is __init__.py with implementation code.

    Standalone version for parallel processing.
    """
    if file_path.name != "__init__.py":
        return False

    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                continue

        if isinstance(node, ast.Import | ast.ImportFrom):
            continue

        if isinstance(node, ast.Pass):
            continue

        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                continue

        return True

    return False


class PythonFeatureAnalyzer:
    """Analyzes Python files for language feature usage."""

    def analyze_directory(self, directory: Path) -> FeatureStats:
        """Analyze all Python files in directory recursively.

        Args:
            directory: Root directory to search

        Returns:
            Aggregated FeatureStats
        """
        from slopometry.core.git_tracker import GitTracker

        tracker = GitTracker(directory)
        python_files = [f for f in tracker.get_tracked_python_files() if f.exists()]

        start_total = time.perf_counter()

        # Use parallel processing for large file sets
        if len(python_files) >= settings.parallel_file_threshold:
            results = self._analyze_files_parallel(python_files)
        else:
            results = [_analyze_single_file_features(fp) for fp in python_files]

        # Aggregate results
        aggregated = FeatureStats()
        for stats in results:
            if stats is not None:
                aggregated = self._merge_stats(aggregated, stats)

        elapsed = time.perf_counter() - start_total
        mode = "parallel" if len(python_files) >= settings.parallel_file_threshold else "sequential"
        logger.debug(f"Feature analysis ({mode}): {len(python_files)} files in {elapsed:.2f}s")

        return aggregated

    def _analyze_files_parallel(self, files: list[Path], max_workers: int | None = None) -> list[FeatureStats | None]:
        """Analyze files in parallel using ProcessPoolExecutor.

        Args:
            files: List of Python file paths to analyze
            max_workers: Maximum number of worker processes (default from settings)

        Returns:
            List of FeatureStats objects
        """
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, settings.max_parallel_workers)

        results: list[FeatureStats | None] = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_analyze_single_file_features, fp): fp for fp in files}

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    file_path = futures[future]
                    logger.warning(f"Failed to analyze features for {file_path}: {e}")
                    results.append(None)

        return results

    def _analyze_file(self, file_path: Path) -> FeatureStats:
        """Analyze a single Python file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(file_path))
        except Exception as e:
            logger.debug(f"Skipping unparseable file {file_path}: {e}")
            return FeatureStats()

        visitor = FeatureVisitor()
        visitor.visit(tree)
        ast_stats = visitor.stats

        is_test_file = file_path.name.startswith("test_") or "/tests/" in str(file_path)
        orphan_comments, untracked_todos, type_ignores = self._analyze_comments(content, is_test_file)
        nonempty_init = 1 if self._is_nonempty_init(file_path, tree) else 0
        total_loc, code_loc = _count_loc(content)
        path_str = str(file_path)

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
            dict_get_with_default_count=ast_stats.dict_get_with_default_count,
            hasattr_getattr_count=ast_stats.hasattr_getattr_count,
            nonempty_init_count=nonempty_init,
            test_skip_count=ast_stats.test_skip_count,
            swallowed_exception_count=ast_stats.swallowed_exception_count,
            type_ignore_count=type_ignores,
            dynamic_execution_count=ast_stats.dynamic_execution_count,
            single_method_class_count=ast_stats.single_method_class_count,
            deep_inheritance_count=ast_stats.deep_inheritance_count,
            passthrough_wrapper_count=ast_stats.passthrough_wrapper_count,
            total_loc=total_loc,
            code_loc=code_loc,
            orphan_comment_files={path_str} if orphan_comments > 0 else set(),
            untracked_todo_files={path_str} if untracked_todos > 0 else set(),
            inline_import_files={path_str} if ast_stats.inline_import_count > 0 else set(),
            dict_get_with_default_files={path_str} if ast_stats.dict_get_with_default_count > 0 else set(),
            hasattr_getattr_files={path_str} if ast_stats.hasattr_getattr_count > 0 else set(),
            nonempty_init_files={path_str} if nonempty_init > 0 else set(),
            test_skip_files={path_str} if ast_stats.test_skip_count > 0 else set(),
            swallowed_exception_files={path_str} if ast_stats.swallowed_exception_count > 0 else set(),
            type_ignore_files={path_str} if type_ignores > 0 else set(),
            dynamic_execution_files={path_str} if ast_stats.dynamic_execution_count > 0 else set(),
            single_method_class_files={path_str} if ast_stats.single_method_class_count > 0 else set(),
            deep_inheritance_files={path_str} if ast_stats.deep_inheritance_count > 0 else set(),
            passthrough_wrapper_files={path_str} if ast_stats.passthrough_wrapper_count > 0 else set(),
        )

    def _is_nonempty_init(self, file_path: Path, tree: ast.Module) -> bool:
        """Check if file is __init__.py with implementation code (beyond imports/__all__).

        Acceptable content in __init__.py:
        - Imports (Import, ImportFrom)
        - __all__ assignment
        - Module docstring
        - Pass statements

        Implementation code (flagged as smell):
        - Function definitions
        - Class definitions
        - Other assignments (except __all__)
        - Other expressions
        """
        if file_path.name != "__init__.py":
            return False

        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    continue

            if isinstance(node, ast.Import | ast.ImportFrom):
                continue

            if isinstance(node, ast.Pass):
                continue

            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                    continue

            return True

        return False

    def _analyze_comments(self, content: str, is_test_file: bool = False) -> tuple[int, int, int]:
        """Analyze comments in source code using tokenize.

        Args:
            content: Source code content
            is_test_file: If True, skip orphan comment detection (tests need explanatory comments)

        Returns:
            Tuple of (orphan_comment_count, untracked_todo_count, type_ignore_count)
        """
        orphan_comments = 0
        untracked_todos = 0
        type_ignores = 0

        todo_pattern = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
        url_pattern = re.compile(r"https?://")
        ticket_pattern = re.compile(r"([A-Z]+-\d+|#\d+)")
        justification_pattern = re.compile(
            r"#\s*(NOTE|REASON|WARNING|WORKAROUND|IMPORTANT|CAVEAT|HACK|NB|PERF|SAFETY|COMPAT):",
            re.IGNORECASE,
        )
        type_ignore_pattern = re.compile(r"#\s*type:\s*ignore")

        try:
            tokens = tokenize.generate_tokens(io.StringIO(content).readline)
            for tok in tokens:
                if tok.type == tokenize.COMMENT:
                    comment_text = tok.string

                    is_todo = bool(todo_pattern.search(comment_text))
                    has_url = bool(url_pattern.search(comment_text))
                    is_justification = bool(justification_pattern.search(comment_text))
                    is_type_ignore = bool(type_ignore_pattern.search(comment_text))

                    if is_type_ignore:
                        type_ignores += 1
                    elif is_todo:
                        has_ticket = bool(ticket_pattern.search(comment_text))
                        if not has_ticket and not has_url:
                            untracked_todos += 1
                    elif not has_url and not is_justification and not is_test_file:
                        orphan_comments += 1
        except tokenize.TokenError as e:
            logger.debug(f"Tokenize error during comment analysis: {e}")

        return orphan_comments, untracked_todos, type_ignores

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
            dict_get_with_default_count=s1.dict_get_with_default_count + s2.dict_get_with_default_count,
            hasattr_getattr_count=s1.hasattr_getattr_count + s2.hasattr_getattr_count,
            nonempty_init_count=s1.nonempty_init_count + s2.nonempty_init_count,
            test_skip_count=s1.test_skip_count + s2.test_skip_count,
            swallowed_exception_count=s1.swallowed_exception_count + s2.swallowed_exception_count,
            type_ignore_count=s1.type_ignore_count + s2.type_ignore_count,
            dynamic_execution_count=s1.dynamic_execution_count + s2.dynamic_execution_count,
            single_method_class_count=s1.single_method_class_count + s2.single_method_class_count,
            deep_inheritance_count=s1.deep_inheritance_count + s2.deep_inheritance_count,
            passthrough_wrapper_count=s1.passthrough_wrapper_count + s2.passthrough_wrapper_count,
            total_loc=s1.total_loc + s2.total_loc,
            code_loc=s1.code_loc + s2.code_loc,
            orphan_comment_files=s1.orphan_comment_files | s2.orphan_comment_files,
            untracked_todo_files=s1.untracked_todo_files | s2.untracked_todo_files,
            inline_import_files=s1.inline_import_files | s2.inline_import_files,
            dict_get_with_default_files=s1.dict_get_with_default_files | s2.dict_get_with_default_files,
            hasattr_getattr_files=s1.hasattr_getattr_files | s2.hasattr_getattr_files,
            nonempty_init_files=s1.nonempty_init_files | s2.nonempty_init_files,
            test_skip_files=s1.test_skip_files | s2.test_skip_files,
            swallowed_exception_files=s1.swallowed_exception_files | s2.swallowed_exception_files,
            type_ignore_files=s1.type_ignore_files | s2.type_ignore_files,
            dynamic_execution_files=s1.dynamic_execution_files | s2.dynamic_execution_files,
            single_method_class_files=s1.single_method_class_files | s2.single_method_class_files,
            deep_inheritance_files=s1.deep_inheritance_files | s2.deep_inheritance_files,
            passthrough_wrapper_files=s1.passthrough_wrapper_files | s2.passthrough_wrapper_files,
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
        self.total_type_refs = 0
        self.any_type_refs = 0
        self.str_type_refs = 0
        self.inline_imports = 0
        self._in_type_checking_block = False
        self._scope_depth = 0
        self.dict_get_with_default = 0
        self.hasattr_getattr_calls = 0
        self.test_skips = 0
        self.swallowed_exceptions = 0
        self.dynamic_executions = 0
        # Abstraction smells
        self.single_method_classes = 0
        self.deep_inheritances = 0
        self.passthrough_wrappers = 0

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
            dict_get_with_default_count=self.dict_get_with_default,
            hasattr_getattr_count=self.hasattr_getattr_calls,
            test_skip_count=self.test_skips,
            swallowed_exception_count=self.swallowed_exceptions,
            dynamic_execution_count=self.dynamic_executions,
            single_method_class_count=self.single_method_classes,
            deep_inheritance_count=self.deep_inheritances,
            passthrough_wrapper_count=self.passthrough_wrappers,
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
            self._collect_type_names(node.value)
            self._collect_type_names(node.slice)

        elif isinstance(node, ast.BinOp):
            self._collect_type_names(node.left)
            self._collect_type_names(node.right)

        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                self._collect_type_names(elt)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                self.total_type_refs += 1
                if node.value == "Any":
                    self.any_type_refs += 1
                elif node.value == "str":
                    self.str_type_refs += 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
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

        for decorator in node.decorator_list:
            if self._is_deprecated_decorator(decorator):
                self.deprecations += 1
            if self._is_test_skip_decorator(decorator):
                self.test_skips += 1

        # Pass-through wrapper detection
        if self._is_passthrough(node):
            self.passthrough_wrappers += 1

        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_func_common(node)
        for decorator in node.decorator_list:
            if self._is_deprecated_decorator(decorator):
                self.deprecations += 1
            if self._is_test_skip_decorator(decorator):
                self.test_skips += 1

        # Pass-through wrapper detection
        if self._is_passthrough(node):
            self.passthrough_wrappers += 1

        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def _visit_func_common(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
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

        for decorator in node.decorator_list:
            if self._is_deprecated_decorator(decorator):
                self.deprecations += 1

        # Deep inheritance: >2 base classes
        if len(node.bases) > 2:
            self.deep_inheritances += 1

        # Single-method class: only one method besides __init__
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name != "__init__"]
        if len(methods) == 1:
            self.single_method_classes += 1

        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        """Check for warnings.warn, .get() with defaults, hasattr/getattr, test skips, and dynamic execution."""
        if self._is_warnings_warn(node.func):
            if len(node.args) > 1:
                category = node.args[1]
                if self._is_deprecation_warning(category):
                    self.deprecations += 1
            for keyword in node.keywords:
                if keyword.arg == "category" and self._is_deprecation_warning(keyword.value):
                    self.deprecations += 1

        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            has_default = len(node.args) >= 2 or any(kw.arg == "default" for kw in node.keywords)
            if has_default:
                self.dict_get_with_default += 1

        elif isinstance(node.func, ast.Name):
            if node.func.id in ("hasattr", "getattr"):
                self.hasattr_getattr_calls += 1

        if self._is_test_skip_call(node):
            self.test_skips += 1

        if self._is_dynamic_execution_call(node):
            self.dynamic_executions += 1

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

    def _is_test_skip_call(self, node: ast.Call) -> bool:
        """Check if call is pytest.skip/skipif (not unittest, which is decorator-only)."""
        func = node.func
        if isinstance(func, ast.Attribute):
            if func.attr in ("skip", "skipif"):
                if isinstance(func.value, ast.Name):
                    # Only pytest - unittest.skip is handled as decorator only
                    # to avoid double-counting @unittest.skip("reason")
                    return func.value.id == "pytest"
        return False

    def _is_test_skip_decorator(self, node: ast.AST) -> bool:
        """Check if decorator is @pytest.mark.skip/skipif or @unittest.skip/skipIf."""
        if isinstance(node, ast.Call):
            return self._is_test_skip_decorator(node.func)

        if isinstance(node, ast.Attribute):
            # @pytest.mark.skip or @pytest.mark.skipif
            if node.attr in ("skip", "skipif"):
                if isinstance(node.value, ast.Attribute) and node.value.attr == "mark":
                    if isinstance(node.value.value, ast.Name) and node.value.value.id == "pytest":
                        return True
            # @unittest.skip or @unittest.skipIf
            if node.attr in ("skip", "skipIf"):
                if isinstance(node.value, ast.Name) and node.value.id == "unittest":
                    return True
        return False

    def _is_dynamic_execution_call(self, node: ast.Call) -> bool:
        """Check for eval/exec/compile calls."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in ("eval", "exec", "compile")
        return False

    def _is_passthrough(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if function just returns a call with same arguments.

        A pass-through wrapper is a function whose body is just:
        - return other_func(arg1, arg2, ...)
        where arg1, arg2, ... are exactly the function's parameters (excluding self/cls).
        """
        # Skip if function has docstring (body[0] is docstring, body[1] might be return)
        body = node.body
        if ast.get_docstring(node):
            body = body[1:]

        if len(body) != 1:
            return False

        stmt = body[0]
        if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Call):
            return False

        call = stmt.value

        # Get function's argument names (excluding self/cls)
        func_args = [arg.arg for arg in node.args.args if arg.arg not in ("self", "cls")]

        # Function must have at least one argument to be a meaningful pass-through
        if not func_args:
            return False

        # Get call's positional argument names
        call_args = [arg.id for arg in call.args if isinstance(arg, ast.Name)]

        # Check if call args exactly match function args in order
        return func_args == call_args

    def visit_If(self, node: ast.If) -> None:
        """Track TYPE_CHECKING blocks to exclude their imports."""
        if self._is_type_checking_guard(node):
            self._in_type_checking_block = True
            self.generic_visit(node)
            self._in_type_checking_block = False
        else:
            self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        """Detect swallowed exceptions (except blocks with only pass/continue/empty)."""
        for handler in node.handlers:
            if self._is_swallowed_exception(handler):
                self.swallowed_exceptions += 1
        self.generic_visit(node)

    def _is_swallowed_exception(self, handler: ast.ExceptHandler) -> bool:
        """Check if exception handler just swallows (pass/continue/empty body).

        Not considered swallowed if the handler logs the exception.
        """
        if not handler.body:
            return True

        # Check if any statement in the handler is a logging call
        for stmt in handler.body:
            if self._is_logging_call(stmt):
                return False

        # Single statement that's pass/continue is swallowed
        if len(handler.body) == 1:
            stmt = handler.body[0]
            if isinstance(stmt, ast.Pass | ast.Continue):
                return True

        return False

    def _is_logging_call(self, stmt: ast.stmt) -> bool:
        """Check if a statement is a logging/print call."""
        if not isinstance(stmt, ast.Expr):
            return False
        if not isinstance(stmt.value, ast.Call):
            return False

        call = stmt.value
        func = call.func

        # Check for print() call
        if isinstance(func, ast.Name) and func.id == "print":
            return True

        # Check for attribute calls like logger.warning, logging.info, console.print
        if isinstance(func, ast.Attribute):
            # logger.*, logging.*
            if isinstance(func.value, ast.Name):
                if func.value.id in ("logger", "logging", "log", "console"):
                    return True
            # self.logger.*
            if isinstance(func.value, ast.Attribute):
                if func.value.attr in ("logger", "log"):
                    return True

        return False

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
