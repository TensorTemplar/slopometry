"""Tests for the smell registry and SmellData model."""

import pytest

from slopometry.core.models import (
    SMELL_REGISTRY,
    ComplexityDelta,
    ExtendedComplexityMetrics,
    SmellCategory,
    SmellData,
    get_smell_label,
    get_smells_by_category,
)


class TestSmellRegistry:
    """Tests for SMELL_REGISTRY completeness and consistency."""

    def test_smell_registry__has_all_13_smells(self) -> None:
        """Verify all expected smells are in the registry."""
        expected_smells = {
            "orphan_comment",
            "untracked_todo",
            "swallowed_exception",
            "test_skip",
            "type_ignore",
            "dynamic_execution",
            "inline_import",
            "dict_get_with_default",
            "hasattr_getattr",
            "nonempty_init",
            # Abstraction smells (QPE v2)
            "single_method_class",
            "deep_inheritance",
            "passthrough_wrapper",
        }
        assert set(SMELL_REGISTRY.keys()) == expected_smells

    def test_smell_registry__all_definitions_have_required_fields(self) -> None:
        """Verify all smell definitions have required fields populated."""
        for name, defn in SMELL_REGISTRY.items():
            assert defn.internal_name == name, f"{name}: internal_name mismatch"
            assert defn.label, f"{name}: missing label"
            assert defn.category in SmellCategory, f"{name}: invalid category"
            assert 0 < defn.weight <= 1.0, f"{name}: weight {defn.weight} out of range"
            assert defn.guidance, f"{name}: missing guidance"
            assert defn.count_field.endswith("_count"), f"{name}: invalid count_field"
            assert defn.files_field.endswith("_files"), f"{name}: invalid files_field"

    def test_smell_registry__general_category_smells(self) -> None:
        """Verify expected smells are categorized as GENERAL."""
        general_smells = {
            "orphan_comment",
            "untracked_todo",
            "swallowed_exception",
            "test_skip",
            "type_ignore",
            "dynamic_execution",
        }
        for name in general_smells:
            assert SMELL_REGISTRY[name].category == SmellCategory.GENERAL

    def test_smell_registry__python_category_smells(self) -> None:
        """Verify expected smells are categorized as PYTHON."""
        python_smells = {
            "inline_import",
            "dict_get_with_default",
            "hasattr_getattr",
            "nonempty_init",
            # Abstraction smells (QPE v2)
            "single_method_class",
            "deep_inheritance",
            "passthrough_wrapper",
        }
        for name in python_smells:
            assert SMELL_REGISTRY[name].category == SmellCategory.PYTHON


class TestSmellHelpers:
    """Tests for smell helper functions."""

    def test_get_smell_label__returns_registry_label(self) -> None:
        """Verify get_smell_label returns the registry label."""
        assert get_smell_label("orphan_comment") == "Orphan Comments"
        assert get_smell_label("swallowed_exception") == "Swallowed Exceptions"

    def test_get_smell_label__handles_unknown_smell(self) -> None:
        """Verify get_smell_label handles unknown smells gracefully."""
        assert get_smell_label("unknown_smell") == "Unknown Smell"

    def test_get_smells_by_category__returns_general_smells(self) -> None:
        """Verify get_smells_by_category returns all GENERAL smells."""
        general = get_smells_by_category(SmellCategory.GENERAL)
        assert len(general) == 6
        assert all(d.category == SmellCategory.GENERAL for d in general)

    def test_get_smells_by_category__returns_python_smells(self) -> None:
        """Verify get_smells_by_category returns all PYTHON smells."""
        python = get_smells_by_category(SmellCategory.PYTHON)
        assert len(python) == 7  # 4 original + 3 abstraction smells
        assert all(d.category == SmellCategory.PYTHON for d in python)

    def test_get_smells_by_category__sorted_by_weight_descending(self) -> None:
        """Verify get_smells_by_category returns smells sorted by weight (highest first)."""
        general = get_smells_by_category(SmellCategory.GENERAL)
        weights = [d.weight for d in general]
        assert weights == sorted(weights, reverse=True)


class TestSmellData:
    """Tests for SmellData model."""

    def test_smell_data__provides_definition_via_property(self) -> None:
        """Verify SmellData.definition returns the registry definition."""
        smell = SmellData(name="orphan_comment", count=5, files=["a.py", "b.py"])
        assert smell.definition == SMELL_REGISTRY["orphan_comment"]

    def test_smell_data__provides_label_via_property(self) -> None:
        """Verify SmellData.label returns the display label."""
        smell = SmellData(name="swallowed_exception", count=3, files=["a.py"])
        assert smell.label == "Swallowed Exceptions"

    def test_smell_data__provides_category_via_property(self) -> None:
        """Verify SmellData.category returns the smell category."""
        smell = SmellData(name="inline_import", count=10, files=["a.py"])
        assert smell.category == SmellCategory.PYTHON

    def test_smell_data__provides_weight_via_property(self) -> None:
        """Verify SmellData.weight returns the smell weight."""
        smell = SmellData(name="test_skip", count=2, files=["test.py"])
        assert smell.weight == 0.10

    def test_smell_data__is_frozen(self) -> None:
        """Verify SmellData is immutable."""
        smell = SmellData(name="orphan_comment", count=5, files=["a.py"])
        with pytest.raises(Exception):  # ValidationError in Pydantic
            smell.count = 10  # type: ignore[misc]


class TestExtendedComplexityMetricsSmellMethods:
    """Tests for smell-related methods on ExtendedComplexityMetrics."""

    @pytest.fixture
    def metrics_with_smells(self) -> ExtendedComplexityMetrics:
        """Create metrics with various smells."""
        return ExtendedComplexityMetrics(
            total_files_analyzed=10,
            total_complexity=50,
            average_complexity=5.0,
            max_complexity=15,
            min_complexity=1,
            total_tokens=5000,
            average_tokens=500.0,
            max_tokens=1000,
            min_tokens=100,
            total_volume=10000.0,
            total_effort=50000.0,
            total_difficulty=5.0,
            average_volume=1000.0,
            average_effort=5000.0,
            average_difficulty=0.5,
            total_mi=850.0,
            average_mi=85.0,
            orphan_comment_count=3,
            orphan_comment_files=["a.py", "b.py"],
            swallowed_exception_count=1,
            swallowed_exception_files=["error_handler.py"],
            test_skip_count=0,
            test_skip_files=[],
        )

    def test_get_smells__returns_all_smell_data(self, metrics_with_smells: ExtendedComplexityMetrics) -> None:
        """Verify get_smells returns SmellData for all smells."""
        smells = metrics_with_smells.get_smells()
        assert len(smells) == 13  # 10 original + 3 abstraction smells
        assert all(isinstance(s, SmellData) for s in smells)

    def test_get_smells__includes_correct_counts(self, metrics_with_smells: ExtendedComplexityMetrics) -> None:
        """Verify get_smells returns correct counts."""
        smells = {s.name: s for s in metrics_with_smells.get_smells()}
        assert smells["orphan_comment"].count == 3
        assert smells["swallowed_exception"].count == 1
        assert smells["test_skip"].count == 0

    def test_get_smells__includes_correct_files(self, metrics_with_smells: ExtendedComplexityMetrics) -> None:
        """Verify get_smells returns correct file lists."""
        smells = {s.name: s for s in metrics_with_smells.get_smells()}
        assert smells["orphan_comment"].files == ["a.py", "b.py"]
        assert smells["swallowed_exception"].files == ["error_handler.py"]
        assert smells["test_skip"].files == []

    def test_get_smell_files__returns_name_to_files_mapping(
        self, metrics_with_smells: ExtendedComplexityMetrics
    ) -> None:
        """Verify get_smell_files returns dict mapping smell names to files."""
        smell_files = metrics_with_smells.get_smell_files()
        assert smell_files["orphan_comment"] == ["a.py", "b.py"]
        assert smell_files["swallowed_exception"] == ["error_handler.py"]
        assert smell_files["test_skip"] == []

    def test_get_smell_counts__returns_name_to_count_mapping(
        self, metrics_with_smells: ExtendedComplexityMetrics
    ) -> None:
        """Verify get_smell_counts returns dict mapping smell names to counts."""
        smell_counts = metrics_with_smells.get_smell_counts()
        assert len(smell_counts) == 13  # 10 original + 3 abstraction smells
        assert smell_counts["orphan_comment"] == 3
        assert smell_counts["swallowed_exception"] == 1
        assert smell_counts["test_skip"] == 0


class TestComplexityDeltaSmellChanges:
    """Tests for smell-related methods on ComplexityDelta."""

    def test_get_smell_changes__returns_all_smell_changes(self) -> None:
        """Verify get_smell_changes returns all smell change values."""
        delta = ComplexityDelta(
            orphan_comment_change=2,
            swallowed_exception_change=-1,
            test_skip_change=0,
        )
        changes = delta.get_smell_changes()
        assert len(changes) == 13  # 10 original + 3 abstraction smells
        assert changes["orphan_comment"] == 2
        assert changes["swallowed_exception"] == -1
        assert changes["test_skip"] == 0

    def test_get_smell_changes__default_values_are_zero(self) -> None:
        """Verify get_smell_changes returns 0 for unset changes."""
        delta = ComplexityDelta()
        changes = delta.get_smell_changes()
        assert all(v == 0 for v in changes.values())
