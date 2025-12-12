"""Shared test fixtures and helpers."""

import pytest


@pytest.fixture
def halstead_defaults() -> dict:
    """Default values for required Halstead fields in ExtendedComplexityMetrics.

    Use this fixture to provide sensible defaults when testing functionality
    that doesn't depend on specific Halstead values.
    """
    return {
        "total_volume": 0.0,
        "total_effort": 0.0,
        "total_difficulty": 0.0,
        "average_volume": 0.0,
        "average_effort": 0.0,
        "average_difficulty": 0.0,
        "total_mi": 0.0,
        "average_mi": 0.0,
    }


def make_test_metrics(**overrides) -> dict:
    """Create a dict of ExtendedComplexityMetrics fields with defaults for all required Halstead fields.

    Args:
        **overrides: Any fields to override from defaults.

    Returns:
        Dict ready to be passed to ExtendedComplexityMetrics(**result).
    """
    defaults = {
        "total_volume": 0.0,
        "total_effort": 0.0,
        "total_difficulty": 0.0,
        "average_volume": 0.0,
        "average_effort": 0.0,
        "average_difficulty": 0.0,
        "total_mi": 0.0,
        "average_mi": 0.0,
    }
    defaults.update(overrides)
    return defaults
