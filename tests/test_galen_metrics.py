"""Tests for Galen Rate metrics calculations."""

import pytest

from slopometry.core.models import GALEN_TOKENS_PER_DAY, GALEN_TOKENS_PER_MONTH, GalenMetrics


class TestGalenMetricsCalculations:
    """Test cases for GalenMetrics.calculate() method."""

    def test_calculate__exactly_one_galen(self):
        """1M tokens over 30 days should equal exactly 1 Galen."""
        metrics = GalenMetrics.calculate(
            tokens_changed=GALEN_TOKENS_PER_MONTH, period_days=30.0
        )

        assert metrics.tokens_changed == GALEN_TOKENS_PER_MONTH
        assert metrics.period_days == 30.0
        assert metrics.tokens_per_day == pytest.approx(GALEN_TOKENS_PER_DAY, rel=0.01)
        assert metrics.galen_rate == pytest.approx(1.0, rel=0.01)
        assert metrics.tokens_per_day_to_reach_one_galen is None

    def test_calculate__half_galen(self):
        """500K tokens over 30 days should equal 0.5 Galen."""
        metrics = GalenMetrics.calculate(tokens_changed=500_000, period_days=30.0)

        assert metrics.galen_rate == pytest.approx(0.5, rel=0.01)
        assert metrics.tokens_per_day_to_reach_one_galen is not None
        # Need ~16,667 more tokens/day to reach 1 Galen
        assert metrics.tokens_per_day_to_reach_one_galen == pytest.approx(
            GALEN_TOKENS_PER_DAY / 2, rel=0.01
        )

    def test_calculate__double_galen(self):
        """2M tokens over 30 days should equal 2 Galens."""
        metrics = GalenMetrics.calculate(tokens_changed=2_000_000, period_days=30.0)

        assert metrics.galen_rate == pytest.approx(2.0, rel=0.01)
        assert metrics.tokens_per_day_to_reach_one_galen is None

    def test_calculate__zero_period_returns_zero_rate(self):
        """Zero period days should return 0 Galen rate."""
        metrics = GalenMetrics.calculate(tokens_changed=100_000, period_days=0.0)

        assert metrics.galen_rate == 0.0
        assert metrics.tokens_per_day == 0.0
        assert metrics.tokens_per_day_to_reach_one_galen == GALEN_TOKENS_PER_DAY

    def test_calculate__negative_period_returns_zero_rate(self):
        """Negative period days should return 0 Galen rate."""
        metrics = GalenMetrics.calculate(tokens_changed=100_000, period_days=-1.0)

        assert metrics.galen_rate == 0.0
        assert metrics.period_days == 0.0

    def test_calculate__negative_tokens_uses_absolute_value(self):
        """Negative token delta (code removal) should use absolute value."""
        metrics = GalenMetrics.calculate(tokens_changed=-500_000, period_days=30.0)

        # Galen rate should be based on absolute value of tokens changed
        assert metrics.galen_rate == pytest.approx(0.5, rel=0.01)
        # But the tokens_changed field should preserve the sign
        assert metrics.tokens_changed == -500_000

    def test_calculate__short_period_extrapolates_correctly(self):
        """Tokens over a short period should extrapolate correctly."""
        # 1000 tokens in 1 day = 30,000 tokens/month = 0.03 Galens
        metrics = GalenMetrics.calculate(tokens_changed=1000, period_days=1.0)

        assert metrics.tokens_per_day == 1000.0
        expected_galen_rate = 1000.0 / GALEN_TOKENS_PER_DAY
        assert metrics.galen_rate == pytest.approx(expected_galen_rate, rel=0.01)

    def test_calculate__fractional_days(self):
        """Should handle fractional days (e.g., session duration in hours)."""
        # 2 hours = 1/12 day
        period_days = 2.0 / 24.0
        tokens = 2000

        metrics = GalenMetrics.calculate(tokens_changed=tokens, period_days=period_days)

        expected_per_day = tokens / period_days
        assert metrics.tokens_per_day == pytest.approx(expected_per_day, rel=0.01)


class TestGalenMetricsModel:
    """Test cases for GalenMetrics model properties."""

    def test_model_fields_are_set(self):
        """All required fields should be populated."""
        metrics = GalenMetrics.calculate(tokens_changed=100_000, period_days=10.0)

        assert metrics.tokens_changed == 100_000
        assert metrics.period_days == 10.0
        assert metrics.tokens_per_day == 10_000.0
        assert isinstance(metrics.galen_rate, float)

    def test_tokens_per_day_to_reach_one_galen__below_threshold(self):
        """Should show tokens needed when below 1 Galen."""
        metrics = GalenMetrics.calculate(tokens_changed=100_000, period_days=30.0)

        assert metrics.galen_rate < 1.0
        assert metrics.tokens_per_day_to_reach_one_galen is not None
        assert metrics.tokens_per_day_to_reach_one_galen > 0

    def test_tokens_per_day_to_reach_one_galen__at_or_above_threshold(self):
        """Should be None when at or above 1 Galen."""
        metrics = GalenMetrics.calculate(tokens_changed=1_000_000, period_days=30.0)

        assert metrics.galen_rate >= 1.0
        assert metrics.tokens_per_day_to_reach_one_galen is None


class TestGalenConstants:
    """Test cases for Galen rate constants."""

    def test_galen_tokens_per_month(self):
        """1 Galen should equal 1 million tokens per month."""
        assert GALEN_TOKENS_PER_MONTH == 1_000_000

    def test_galen_tokens_per_day(self):
        """Galen tokens per day should be 1M / 30 days."""
        assert GALEN_TOKENS_PER_DAY == pytest.approx(33_333.33, rel=0.01)
