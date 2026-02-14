"""Code smell definitions, registry, and data models."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SmellCategory(str, Enum):
    """Category of code smell for organization and filtering."""

    GENERAL = "general"
    PYTHON = "python"


class SmellDefinition(BaseModel):
    """Definition of a code smell with all metadata."""

    model_config = ConfigDict(frozen=True)

    internal_name: str
    label: str
    category: SmellCategory
    weight: float
    guidance: str
    count_field: str
    files_field: str


SMELL_REGISTRY: dict[str, SmellDefinition] = {
    # General smells (language-agnostic concepts)
    "orphan_comment": SmellDefinition(
        internal_name="orphan_comment",
        label="Orphan Comments",
        category=SmellCategory.GENERAL,
        weight=0.01,
        guidance="Make sure inline code comments add meaningful information about non-obvious design tradeoffs or explain tech debt or performance implications. Consider if these could be docstrings or field descriptors instead",
        count_field="orphan_comment_count",
        files_field="orphan_comment_files",
    ),
    "untracked_todo": SmellDefinition(
        internal_name="untracked_todo",
        label="Untracked TODOs",
        category=SmellCategory.GENERAL,
        weight=0.02,
        guidance="Untracked TODOs should include ticket references (JIRA-123, #123) or URLs",
        count_field="untracked_todo_count",
        files_field="untracked_todo_files",
    ),
    "swallowed_exception": SmellDefinition(
        internal_name="swallowed_exception",
        label="Swallowed Exceptions",
        category=SmellCategory.GENERAL,
        weight=0.15,
        guidance="BLOCKING: You MUST present a table with columns [Location | Purpose | Justification ] for each and ask user to confirm silent failure is acceptable",
        count_field="swallowed_exception_count",
        files_field="swallowed_exception_files",
    ),
    "test_skip": SmellDefinition(
        internal_name="test_skip",
        label="Test Skips",
        category=SmellCategory.GENERAL,
        weight=0.10,
        guidance="BLOCKING: You MUST present a table with columns [Test Name | Intent] for each skip and ask user to confirm skipping is acceptable",
        count_field="test_skip_count",
        files_field="test_skip_files",
    ),
    "type_ignore": SmellDefinition(
        internal_name="type_ignore",
        label="Type Ignores",
        category=SmellCategory.GENERAL,
        weight=0.08,
        guidance="Review type: ignore comments - consider fixing the underlying type issue",
        count_field="type_ignore_count",
        files_field="type_ignore_files",
    ),
    "dynamic_execution": SmellDefinition(
        internal_name="dynamic_execution",
        label="Dynamic Execution",
        category=SmellCategory.GENERAL,
        weight=0.12,
        guidance="Review usage of eval/exec/compile - ensure this is necessary and secure",
        count_field="dynamic_execution_count",
        files_field="dynamic_execution_files",
    ),
    "inline_import": SmellDefinition(
        internal_name="inline_import",
        label="Inline Imports",
        category=SmellCategory.PYTHON,
        weight=0.01,
        guidance="Verify if these can be moved to the top of the file (except TYPE_CHECKING guards)",
        count_field="inline_import_count",
        files_field="inline_import_files",
    ),
    "dict_get_with_default": SmellDefinition(
        internal_name="dict_get_with_default",
        label="Dict .get() Defaults",
        category=SmellCategory.PYTHON,
        weight=0.05,
        guidance="Consider if these are justified or just indicate modeling gaps - replace with Pydantic BaseSettings or BaseModel with narrower field types or raise explicit errors instead",
        count_field="dict_get_with_default_count",
        files_field="dict_get_with_default_files",
    ),
    "hasattr_getattr": SmellDefinition(
        internal_name="hasattr_getattr",
        label="hasattr/getattr",
        category=SmellCategory.PYTHON,
        weight=0.10,
        guidance="Consider if these indicate missing domain models - replace with Pydantic BaseModel objects with explicit fields or raise explicit errors instead",
        count_field="hasattr_getattr_count",
        files_field="hasattr_getattr_files",
    ),
    "nonempty_init": SmellDefinition(
        internal_name="nonempty_init",
        label="Non-empty __init__",
        category=SmellCategory.PYTHON,
        weight=0.03,
        guidance="Consider if implementation code should be moved out of __init__.py files",
        count_field="nonempty_init_count",
        files_field="nonempty_init_files",
    ),
    # Abstraction smells (unnecessary complexity)
    "single_method_class": SmellDefinition(
        internal_name="single_method_class",
        label="Single-Method Classes",
        category=SmellCategory.PYTHON,
        weight=0.05,
        guidance="Consider using a function instead of a class with only one method besides __init__",
        count_field="single_method_class_count",
        files_field="single_method_class_files",
    ),
    "deep_inheritance": SmellDefinition(
        internal_name="deep_inheritance",
        label="Deep Inheritance",
        category=SmellCategory.PYTHON,
        weight=0.08,
        guidance="Prefer composition over inheritance; >2 base classes increases complexity",
        count_field="deep_inheritance_count",
        files_field="deep_inheritance_files",
    ),
    "passthrough_wrapper": SmellDefinition(
        internal_name="passthrough_wrapper",
        label="Pass-Through Wrappers",
        category=SmellCategory.PYTHON,
        weight=0.02,
        guidance="Function that just delegates to another with same args; consider removing indirection",
        count_field="passthrough_wrapper_count",
        files_field="passthrough_wrapper_files",
    ),
    "sys_path_manipulation": SmellDefinition(
        internal_name="sys_path_manipulation",
        label="sys.path Manipulation",
        category=SmellCategory.PYTHON,
        weight=0.10,
        guidance="sys.path mutations bypass the package system — restructure package boundaries and use absolute imports from installed packages instead",
        count_field="sys_path_manipulation_count",
        files_field="sys_path_manipulation_files",
    ),
}


def get_smell_label(internal_name: str) -> str:
    """Get display label for a smell from registry."""
    defn = SMELL_REGISTRY.get(internal_name)
    return defn.label if defn else internal_name.replace("_", " ").title()


def get_smells_by_category(category: SmellCategory) -> list[SmellDefinition]:
    """Get all smells in a category, sorted by weight (highest first)."""
    return sorted(
        [d for d in SMELL_REGISTRY.values() if d.category == category],
        key=lambda d: d.weight,
        reverse=True,
    )


def SmellField(
    default: int = 0,
    *,
    label: str,
    files_field: str,
    guidance: str,
) -> Any:
    """Create a Field for a code smell metric with embedded metadata.

    Args:
        default: Default value (always 0 for counts)
        label: Display label for the smell (e.g., "Orphan Comments")
        files_field: Name of the corresponding files list field
        guidance: Actionable message shown in feedback
    """
    return Field(
        default=default,
        description=guidance,
        json_schema_extra={
            "is_smell": True,
            "label": label,
            "files_field": files_field,
        },
    )


class ScopedSmell(BaseModel):
    """A smell classified for a specific session context."""

    model_config = ConfigDict(frozen=True)

    label: str
    name: str
    count: int
    change: int
    actionable_files: list[str]
    guidance: str
    is_blocking: bool


class SmellData(BaseModel):
    """Structured smell data with direct field access (no getattr needed)."""

    model_config = ConfigDict(frozen=True)

    name: str
    count: int
    files: list[str]

    @property
    def definition(self) -> SmellDefinition:
        """Get the smell definition from registry."""
        return SMELL_REGISTRY[self.name]

    @property
    def label(self) -> str:
        """Get display label from registry."""
        return self.definition.label

    @property
    def category(self) -> SmellCategory:
        """Get category from registry."""
        return self.definition.category

    @property
    def weight(self) -> float:
        """Get weight from registry."""
        return self.definition.weight


# Import SmellCounts from core.py to avoid circular imports
# Re-export for backwards compatibility
from slopometry.core.models.core import SmellCounts  # noqa: E402, F401
