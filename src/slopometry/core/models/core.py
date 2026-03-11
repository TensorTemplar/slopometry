"""Core domain models with no external dependencies.

This module contains the foundational types that have NO imports from other model modules.
All cross-module types should import from here to avoid circular reference issues.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from slopometry.core.models.smell import SmellData


class TokenCountError(BaseModel):
    """Error that occurred during token counting."""

    model_config = {"frozen": True}

    message: str
    path: str


class CacheUpdateError(BaseModel):
    """Error that occurred during cache update operation."""

    model_config = {"frozen": True}

    message: str
    session_id: str
    operation: str = "update_coverage"


class SmellCounts(BaseModel):
    """Per-smell occurrence counts from complexity analysis.

    Every field corresponds to a smell in SMELL_REGISTRY. Using explicit fields
    instead of dict[str, int] so access is validated at construction time and
    there is no need for .get() defaults or key existence checks.
    """

    model_config = ConfigDict(frozen=True)

    orphan_comment: int = 0
    untracked_todo: int = 0
    swallowed_exception: int = 0
    test_skip: int = 0
    type_ignore: int = 0
    dynamic_execution: int = 0
    inline_import: int = 0
    dict_get_with_default: int = 0
    hasattr_getattr: int = 0
    nonempty_init: int = 0
    single_method_class: int = 0
    deep_inheritance: int = 0
    passthrough_wrapper: int = 0
    sys_path_manipulation: int = 0
    relative_import: int = 0


class ComplexityMetrics(BaseModel):
    """Cognitive complexity metrics for Python files."""

    total_files_analyzed: int = 0
    total_complexity: int = 0
    average_complexity: float = 0.0
    max_complexity: int = 0
    min_complexity: int = 0
    files_by_complexity: dict[str, int] = Field(
        default_factory=dict, description="Mapping of filename to complexity score"
    )
    files_by_effort: dict[str, float] = Field(
        default_factory=dict, description="Mapping of filename to Halstead effort"
    )
    files_with_parse_errors: dict[str, str] = Field(
        default_factory=dict, description="Files that failed to parse: {filepath: error_message}"
    )

    total_tokens: int = 0
    average_tokens: float = 0.0
    max_tokens: int = 0
    min_tokens: int = 0
    files_by_token_count: dict[str, int] = Field(default_factory=dict, description="Mapping of filename to token count")


class ComplexityDelta(BaseModel):
    """Complexity change comparison between two versions."""

    total_complexity_change: int = 0
    files_added: list[str] = Field(default_factory=list)
    files_removed: list[str] = Field(default_factory=list)
    files_changed: dict[str, int] = Field(default_factory=dict, description="Mapping of filename to complexity delta")
    files_effort_changed: dict[str, float] = Field(
        default_factory=dict, description="Mapping of filename to effort delta"
    )
    net_files_change: int = Field(default=0, description="Net change in number of files (files_added - files_removed)")
    avg_complexity_change: float = 0.0

    total_volume_change: float = 0.0
    avg_volume_change: float = 0.0
    avg_effort_change: float = 0.0
    avg_difficulty_change: float = 0.0
    total_difficulty_change: float = 0.0
    total_effort_change: float = 0.0
    total_mi_change: float = 0.0

    avg_mi_change: float = 0.0

    total_tokens_change: int = 0
    avg_tokens_change: float = 0.0

    qpe_change: float = Field(default=0.0, description="QPE delta (current - baseline)")

    type_hint_coverage_change: float = 0.0
    docstring_coverage_change: float = 0.0
    deprecation_change: int = 0

    any_type_percentage_change: float = 0.0
    str_type_percentage_change: float = 0.0

    orphan_comment_change: int = 0
    untracked_todo_change: int = 0
    inline_import_change: int = 0
    dict_get_with_default_change: int = 0
    hasattr_getattr_change: int = 0
    nonempty_init_change: int = 0
    test_skip_change: int = 0
    swallowed_exception_change: int = 0
    type_ignore_change: int = 0
    dynamic_execution_change: int = 0
    single_method_class_change: int = 0
    deep_inheritance_change: int = 0
    passthrough_wrapper_change: int = 0
    sys_path_manipulation_change: int = 0
    relative_import_change: int = 0

    def get_smell_changes(self) -> dict[str, int]:
        """Return smell name to change value mapping for direct access."""
        from slopometry.core.models.smell import SMELL_REGISTRY

        return {name: getattr(self, f"{name}_change") for name in SMELL_REGISTRY}


class ExtendedComplexityMetrics(ComplexityMetrics):
    """Extended metrics including Halstead and Maintainability Index.

    Core Halstead metrics are required to catch missing parameter bugs early.
    """

    total_volume: float
    total_effort: float
    total_difficulty: float
    average_volume: float
    average_effort: float
    average_difficulty: float

    total_mi: float
    average_mi: float = Field(description="Higher is better (0-100 scale)")
    files_by_mi: dict[str, float] = Field(default_factory=dict, description="Mapping of filename to MI score")

    type_hint_coverage: float = Field(default=0.0, description="Percentage of functions/args with type hints (0-100)")
    docstring_coverage: float = Field(
        default=0.0, description="Percentage of functions/classes with docstrings (0-100)"
    )
    deprecation_count: int = Field(default=0, description="Number of deprecation warnings/markers found")

    any_type_percentage: float = Field(
        default=0.0, description="Percentage of type annotations using Any (0-100). Lower is better."
    )
    str_type_percentage: float = Field(
        default=0.0,
        description="Percentage of type annotations using str (0-100). Consider enums for constrained strings.",
    )

    test_coverage_percent: float | None = Field(
        default=None, description="Pytest test coverage percentage (0-100). None if unavailable."
    )
    test_coverage_source: str | None = Field(
        default=None, description="Source file for coverage data (e.g., 'coverage.xml')"
    )

    orphan_comment_count: int = Field(
        default=0,
        description="Make sure inline code comments add meaningful information about non-obvious design tradeoffs or explain tech debt or performance implications. Consider if these could be docstrings or field descriptors instead",
    )
    untracked_todo_count: int = Field(
        default=0,
        description="Untracked TODOs should include ticket references (JIRA-123, #123) or URLs",
    )
    inline_import_count: int = Field(
        default=0,
        description="Verify if these can be moved to the top of the file (except TYPE_CHECKING guards)",
    )
    dict_get_with_default_count: int = Field(
        default=0,
        description="Consider if these are justified or just indicate modeling gaps - replace with Pydantic BaseSettings or BaseModel with narrower field types or raise explicit errors instead",
    )
    hasattr_getattr_count: int = Field(
        default=0,
        description="Consider if these indicate missing domain models - replace with Pydantic BaseModel objects with explicit fields or raise explicit errors instead",
    )
    nonempty_init_count: int = Field(
        default=0,
        description="Consider if implementation code should be moved out of __init__.py files",
    )
    test_skip_count: int = Field(
        default=0,
        description="BLOCKING: You MUST present a table with columns [Test Name | Intent] for each skip and ask user to confirm skipping is acceptable",
    )
    swallowed_exception_count: int = Field(
        default=0,
        description="BLOCKING: You MUST present a table with columns [Location | Purpose | Justification ] for each and ask user to confirm silent failure is acceptable",
    )
    type_ignore_count: int = Field(
        default=0,
        description="Review type: ignore comments - consider fixing the underlying type issue",
    )
    dynamic_execution_count: int = Field(
        default=0,
        description="Review usage of eval/exec/compile - ensure this is necessary and secure",
    )
    single_method_class_count: int = Field(
        default=0,
        description="Consider using a function instead of a class with only one method besides __init__",
    )
    deep_inheritance_count: int = Field(
        default=0,
        description="Prefer composition over inheritance; >2 base classes increases complexity",
    )
    passthrough_wrapper_count: int = Field(
        default=0,
        description="Function that just delegates to another with same args; consider removing indirection",
    )
    sys_path_manipulation_count: int = Field(
        default=0,
        description="sys.path mutations bypass the package system — restructure package boundaries and use absolute imports from installed packages instead",
    )
    relative_import_count: int = Field(
        default=0,
        description="Prefer absolute imports for clarity and refactor-safety; relative imports create implicit coupling to package structure",
    )

    # LOC metrics (for file filtering in QPE)
    total_loc: int = Field(default=0, description="Total lines of code across all files")
    code_loc: int = Field(default=0, description="Non-blank, non-comment lines")
    files_by_loc: dict[str, int] = Field(
        default_factory=dict, description="Mapping of filepath to code LOC for file filtering"
    )

    orphan_comment_files: list[str] = Field(default_factory=list, description="Files with orphan comments")
    untracked_todo_files: list[str] = Field(default_factory=list, description="Files with untracked TODOs")
    inline_import_files: list[str] = Field(default_factory=list, description="Files with inline imports")
    dict_get_with_default_files: list[str] = Field(default_factory=list, description="Files with .get() defaults")
    hasattr_getattr_files: list[str] = Field(default_factory=list, description="Files with hasattr/getattr")
    nonempty_init_files: list[str] = Field(default_factory=list, description="Files with nonempty __init__")
    test_skip_files: list[str] = Field(default_factory=list, description="Files with test skips")
    swallowed_exception_files: list[str] = Field(default_factory=list, description="Files with swallowed exceptions")
    type_ignore_files: list[str] = Field(default_factory=list, description="Files with type: ignore")
    dynamic_execution_files: list[str] = Field(default_factory=list, description="Files with eval/exec/compile")
    single_method_class_files: list[str] = Field(default_factory=list, description="Files with single-method classes")
    deep_inheritance_files: list[str] = Field(
        default_factory=list, description="Files with deep inheritance (>2 bases)"
    )
    passthrough_wrapper_files: list[str] = Field(default_factory=list, description="Files with pass-through wrappers")
    sys_path_manipulation_files: list[str] = Field(default_factory=list, description="Files with sys.path mutations")
    relative_import_files: list[str] = Field(default_factory=list, description="Files with relative imports")

    def get_smell_counts(self) -> SmellCounts:
        """Return smell counts as a typed model for QPE and display."""
        return SmellCounts(**{smell.name: smell.count for smell in self.get_smells()})

    def get_smells(self) -> list["SmellData"]:
        """Return all smell data as structured objects with direct field access."""
        from slopometry.core.models.smell import SMELL_REGISTRY, SmellData

        return [
            SmellData(
                name=defn.internal_name,
                count=getattr(self, defn.count_field),
                files=getattr(self, defn.files_field),
            )
            for defn in SMELL_REGISTRY.values()
        ]

    def get_smell_files(self) -> dict[str, list[str]]:
        """Return smell name to files mapping for filtering."""
        return {smell.name: smell.files for smell in self.get_smells()}
