"""Tests for PythonFeatureAnalyzer including type usage metrics."""

import ast
import subprocess
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from slopometry.core.python_feature_analyzer import FeatureStats, FeatureVisitor, PythonFeatureAnalyzer

# Frozen commit for baseline testing against this repository
FROZEN_COMMIT = "0b6215b"


class TestFeatureVisitorTypeExtraction:
    """Unit tests for type extraction from AST."""

    def test_collect_type_names__extracts_basic_types(self, tmp_path: Path) -> None:
        """Test that basic type annotations are correctly counted."""
        code = """
def foo(x: int, y: str, z: Any) -> bool:
    pass
"""
        (tmp_path / "test.py").write_text(code)
        (tmp_path / ".git").mkdir()  # Make it look like a git repo

        # Bypass git tracking by directly analyzing the file
        import ast

        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)
        stats = visitor.stats

        # 3 args (int, str, Any) + 1 return (bool) = 4 type refs
        assert stats.total_type_references == 4
        assert stats.any_type_count == 1
        assert stats.str_type_count == 1

    def test_collect_type_names__handles_subscript_types(self, tmp_path: Path) -> None:
        """Test extraction from generic types like list[str], dict[str, Any]."""
        code = """
def foo(items: list[str], mapping: dict[str, Any]) -> tuple[int, str]:
    pass
"""
        import ast

        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)
        stats = visitor.stats

        # items: list + str = 2
        # mapping: dict + str + Any = 3
        # return: tuple + int + str = 3
        # Total = 8
        assert stats.total_type_references == 8
        assert stats.any_type_count == 1
        assert stats.str_type_count == 3  # list[str], dict[str, ...], tuple[..., str]

    def test_collect_type_names__handles_union_types(self, tmp_path: Path) -> None:
        """Test extraction from str | None union syntax (Python 3.10+)."""
        code = """
def foo(name: str | None, value: int | str | None) -> bool | None:
    pass
"""
        import ast

        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)
        stats = visitor.stats

        # None is ast.Constant(value=None), not counted as a type reference
        # name: str = 1
        # value: int + str = 2
        # return: bool = 1
        # Total = 4 (None values are not counted as type names)
        assert stats.total_type_references == 4
        assert stats.str_type_count == 2

    def test_collect_type_names__handles_typing_any(self, tmp_path: Path) -> None:
        """Test extraction from typing.Any qualified name."""
        code = """
import typing

def foo(x: typing.Any) -> typing.Any:
    pass
"""
        import ast

        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)
        stats = visitor.stats

        # typing.Any is counted as one type reference per occurrence
        assert stats.total_type_references == 2
        assert stats.any_type_count == 2

    def test_collect_type_names__handles_optional(self, tmp_path: Path) -> None:
        """Test extraction from Optional[str] pattern."""
        code = """
from typing import Optional

def foo(name: Optional[str]) -> Optional[int]:
    pass
"""
        import ast

        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)
        stats = visitor.stats

        # name: Optional + str = 2
        # return: Optional + int = 2
        # Total = 4
        assert stats.total_type_references == 4
        assert stats.str_type_count == 1

    def test_collect_type_names__empty_function(self, tmp_path: Path) -> None:
        """Test that unannotated functions don't increment type counts."""
        code = """
def foo(x, y, z):
    pass
"""
        import ast

        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)
        stats = visitor.stats

        assert stats.total_type_references == 0
        assert stats.any_type_count == 0
        assert stats.str_type_count == 0
        assert stats.functions_count == 1
        assert stats.args_count == 3
        assert stats.annotated_args_count == 0

    def test_collect_type_names__async_function(self, tmp_path: Path) -> None:
        """Test type extraction from async functions."""
        code = """
async def fetch(url: str) -> dict[str, Any]:
    pass
"""
        import ast

        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)
        stats = visitor.stats

        # url: str = 1
        # return: dict + str + Any = 3
        # Total = 4
        assert stats.total_type_references == 4
        assert stats.any_type_count == 1
        assert stats.str_type_count == 2

    def test_collect_type_names__string_annotation(self, tmp_path: Path) -> None:
        """Test type extraction from string annotations (forward refs)."""
        code = """
def foo(x: "SomeClass") -> "str":
    pass
"""
        import ast

        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)
        stats = visitor.stats

        # "SomeClass" + "str" = 2
        assert stats.total_type_references == 2
        assert stats.str_type_count == 1  # "str" as string annotation


class TestFeatureStatsMerge:
    """Tests for FeatureStats merging."""

    def test_merge_stats__aggregates_type_counts(self) -> None:
        """Test that _merge_stats correctly aggregates type tracking fields."""
        s1 = FeatureStats(
            functions_count=5,
            total_type_references=10,
            any_type_count=2,
            str_type_count=3,
        )
        s2 = FeatureStats(
            functions_count=3,
            total_type_references=8,
            any_type_count=1,
            str_type_count=4,
        )

        analyzer = PythonFeatureAnalyzer()
        merged = analyzer._merge_stats(s1, s2)

        assert merged.functions_count == 8
        assert merged.total_type_references == 18
        assert merged.any_type_count == 3
        assert merged.str_type_count == 7


class TestPythonFeatureAnalyzerIntegration:
    """Integration tests using frozen commit as baseline."""

    @pytest.fixture
    def repo_root(self) -> Path:
        """Get the repository root directory."""
        return Path(__file__).parent.parent

    @pytest.fixture
    def extracted_commit(self, tmp_path: Path, repo_root: Path) -> Path:
        """Extract frozen commit to temp directory for testing."""
        result = subprocess.run(
            ["git", "archive", "--format=tar", FROZEN_COMMIT],
            cwd=repo_root,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            pytest.skip(f"Could not extract frozen commit: {result.stderr.decode()}")

        tar_data = BytesIO(result.stdout)
        with tarfile.open(fileobj=tar_data, mode="r") as tar:
            # Extract only Python files from src directory
            python_members = [m for m in tar.getmembers() if m.name.endswith(".py")]
            tar.extractall(path=tmp_path, members=python_members, filter="data")

        # Create minimal git setup so GitTracker works
        (tmp_path / ".git").mkdir()

        return tmp_path

    def test_analyze_directory__frozen_commit_has_type_references(self, extracted_commit: Path) -> None:
        """Verify that frozen commit has some type annotations to analyze."""
        # Need to bypass git tracking for the test - analyze files directly
        import ast

        total_refs = 0
        any_count = 0
        str_count = 0

        for py_file in extracted_commit.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
                visitor = FeatureVisitor()
                visitor.visit(tree)
                total_refs += visitor.total_type_refs
                any_count += visitor.any_type_refs
                str_count += visitor.str_type_refs
            except (SyntaxError, UnicodeDecodeError):
                continue

        # Baseline sanity checks - slopometry codebase should have:
        # - Many type references (well-typed code)
        # - Some Any usage (but hopefully not too much)
        # - Some str usage
        assert total_refs > 100, f"Expected >100 type refs, got {total_refs}"
        assert any_count >= 0, "Any count should be non-negative"
        assert str_count > 0, "Expected some str type usage"

    def test_any_percentage__below_threshold(self, extracted_commit: Path) -> None:
        """Verify Any type usage is below reasonable threshold in frozen commit."""
        import ast

        total_refs = 0
        any_count = 0

        for py_file in extracted_commit.rglob("*.py"):
            try:
                content = py_file.read_text(encoding="utf-8")
                tree = ast.parse(content)
                visitor = FeatureVisitor()
                visitor.visit(tree)
                total_refs += visitor.total_type_refs
                any_count += visitor.any_type_refs
            except (SyntaxError, UnicodeDecodeError):
                continue

        if total_refs > 0:
            any_percentage = (any_count / total_refs) * 100
            # Slopometry should have less than 20% Any usage
            assert any_percentage < 20, f"Any usage too high: {any_percentage:.1f}%"


class TestComplexityAnalyzerTypeIntegration:
    """Integration tests verifying type metrics flow through ComplexityAnalyzer."""

    def test_analyze_extended_complexity__includes_type_percentages(self, tmp_path: Path) -> None:
        """Test that analyze_extended_complexity includes any/str percentages."""
        # Create test files with known type usage
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        code = '''
from typing import Any

def foo(x: str, y: Any, z: int) -> str:
    """A function with mixed types."""
    return x

def bar(a: str, b: str) -> list[str]:
    """A function using only str."""
    return [a, b]
'''
        (src_dir / "module.py").write_text(code)
        (src_dir / "__init__.py").write_text("")

        # Create minimal git setup
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "test"],
            cwd=tmp_path,
            capture_output=True,
        )

        from slopometry.core.complexity_analyzer import ComplexityAnalyzer

        analyzer = ComplexityAnalyzer(working_directory=tmp_path)
        metrics = analyzer.analyze_extended_complexity()

        # Verify the type percentages are computed
        assert metrics.any_type_percentage >= 0
        assert metrics.str_type_percentage >= 0

        # With our test code:
        # foo: str + Any + int + str = 4 type refs (1 Any, 2 str)
        # bar: str + str + list + str = 4 type refs (0 Any, 3 str)
        # Total: 8 refs, 1 Any (12.5%), 5 str (62.5%)
        # Note: list is counted as a type ref too
        assert metrics.any_type_percentage > 0, "Should have some Any usage"
        assert metrics.str_type_percentage > 0, "Should have str usage"


class TestCommentAnalysis:
    """Tests for comment analysis (orphan comments and untracked TODOs)."""

    def test_analyze_comments__counts_orphan_comments(self) -> None:
        """Orphan comments are non-TODO, non-URL comments."""
        code = """
def foo():
    # This is an orphan comment
    x = 1  # Another orphan comment
    return x
"""
        analyzer = PythonFeatureAnalyzer()
        orphan_count, untracked_todos = analyzer._analyze_comments(code)

        assert orphan_count == 2
        assert untracked_todos == 0

    def test_analyze_comments__excludes_todo_from_orphan_count(self) -> None:
        """TODO/FIXME comments should not count as orphan."""
        code = """
def foo():
    # TODO: implement this later
    # FIXME: this is broken
    # XXX: refactor this
    # HACK: workaround for issue
    pass
"""
        analyzer = PythonFeatureAnalyzer()
        orphan_count, untracked_todos = analyzer._analyze_comments(code)

        assert orphan_count == 0
        # All are untracked since they have no ticket refs or URLs
        assert untracked_todos == 4

    def test_analyze_comments__excludes_url_comments_from_orphan_count(self) -> None:
        """Comments with URLs should not count as orphan."""
        code = """
def foo():
    # See https://example.com/docs for details
    # Reference: http://python.org/pep-8
    pass
"""
        analyzer = PythonFeatureAnalyzer()
        orphan_count, untracked_todos = analyzer._analyze_comments(code)

        assert orphan_count == 0
        assert untracked_todos == 0

    def test_analyze_comments__counts_untracked_todos(self) -> None:
        """TODOs without ticket refs or URLs are untracked."""
        code = """
def foo():
    # TODO: implement this
    # FIXME: broken
    pass
"""
        analyzer = PythonFeatureAnalyzer()
        orphan_count, untracked_todos = analyzer._analyze_comments(code)

        assert untracked_todos == 2

    def test_analyze_comments__tracked_todo_with_jira_pattern(self) -> None:
        """TODO with JIRA-123 pattern should not be untracked."""
        code = """
def foo():
    # TODO PROJ-123: implement this feature
    # FIXME ABC-456: fix the bug
    pass
"""
        analyzer = PythonFeatureAnalyzer()
        orphan_count, untracked_todos = analyzer._analyze_comments(code)

        assert untracked_todos == 0

    def test_analyze_comments__tracked_todo_with_github_issue(self) -> None:
        """TODO with #123 should not be untracked."""
        code = """
def foo():
    # TODO #123: implement this feature
    # FIXME #456 fix the bug
    pass
"""
        analyzer = PythonFeatureAnalyzer()
        orphan_count, untracked_todos = analyzer._analyze_comments(code)

        assert untracked_todos == 0

    def test_analyze_comments__tracked_todo_with_url(self) -> None:
        """TODO with URL should not be untracked."""
        code = """
def foo():
    # TODO: see https://github.com/org/repo/issues/123 for details
    # FIXME: tracked at http://jira.example.com/PROJ-456
    pass
"""
        analyzer = PythonFeatureAnalyzer()
        orphan_count, untracked_todos = analyzer._analyze_comments(code)

        assert untracked_todos == 0

    def test_analyze_comments__mixed_comments(self) -> None:
        """Test with a mix of comment types."""
        code = """
# Module docstring comment (orphan)
def foo():
    # TODO: untracked todo
    # TODO PROJ-123: tracked todo (has ticket)
    # Regular comment (orphan)
    # See https://example.com (has URL, not orphan)
    pass
"""
        analyzer = PythonFeatureAnalyzer()
        orphan_count, untracked_todos = analyzer._analyze_comments(code)

        assert orphan_count == 2  # Module comment + "Regular comment"
        assert untracked_todos == 1  # Only "TODO: untracked todo"


class TestInlineImportDetection:
    """Tests for inline import detection."""

    def test_inline_imports__detects_function_level_import(self) -> None:
        """Import inside function body is inline."""
        code = """
def foo():
    import os
    from pathlib import Path
    return Path(os.getcwd())
"""
        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)

        assert visitor.inline_imports == 2

    def test_inline_imports__excludes_type_checking_imports(self) -> None:
        """Imports inside TYPE_CHECKING block are not flagged."""
        code = """
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import os
    from pathlib import Path

def foo():
    pass
"""
        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)

        assert visitor.inline_imports == 0

    def test_inline_imports__excludes_typing_type_checking(self) -> None:
        """Imports inside typing.TYPE_CHECKING block are not flagged."""
        code = """
import typing

if typing.TYPE_CHECKING:
    import os
    from pathlib import Path

def foo():
    pass
"""
        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)

        assert visitor.inline_imports == 0

    def test_inline_imports__module_level_is_not_inline(self) -> None:
        """Top-level imports are not flagged."""
        code = """
import os
from pathlib import Path

def foo():
    return Path(os.getcwd())
"""
        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)

        assert visitor.inline_imports == 0

    def test_inline_imports__class_level_import(self) -> None:
        """Import inside class body is inline."""
        code = """
class Foo:
    import os

    def method(self):
        pass
"""
        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)

        assert visitor.inline_imports == 1

    def test_inline_imports__conditional_import_not_type_checking(self) -> None:
        """Imports in non-TYPE_CHECKING conditionals are flagged."""
        code = """
import sys

if sys.platform == "win32":
    import winreg

def foo():
    pass
"""
        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)

        # This is a module-level conditional, scope_depth is 0
        # The import is technically at module level but inside an if block
        # Based on our implementation, scope_depth doesn't increase for if blocks
        # Only for function/class definitions
        assert visitor.inline_imports == 0

    def test_inline_imports__nested_function_import(self) -> None:
        """Import in nested function is inline."""
        code = """
def outer():
    def inner():
        import json
        return json
    return inner
"""
        tree = ast.parse(code)
        visitor = FeatureVisitor()
        visitor.visit(tree)

        # inner function has scope_depth=2 (outer=1, inner=2)
        assert visitor.inline_imports == 1


class TestFeatureStatsMergeCodeSmells:
    """Tests for FeatureStats merging of code smell fields."""

    def test_merge_stats__aggregates_code_smell_counts(self) -> None:
        """Test that _merge_stats correctly aggregates code smell fields."""
        s1 = FeatureStats(
            functions_count=5,
            orphan_comment_count=10,
            untracked_todo_count=3,
            inline_import_count=2,
        )
        s2 = FeatureStats(
            functions_count=3,
            orphan_comment_count=5,
            untracked_todo_count=1,
            inline_import_count=4,
        )

        analyzer = PythonFeatureAnalyzer()
        merged = analyzer._merge_stats(s1, s2)

        assert merged.orphan_comment_count == 15
        assert merged.untracked_todo_count == 4
        assert merged.inline_import_count == 6
