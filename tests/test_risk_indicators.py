"""합성 관측으로 지표별 결측과 분기·수치 경계를 확인한다."""

import json
from datetime import date

import pytest

from pickplace_ai.risk_indicators import observe_risk_indicators
from pickplace_ai.schemas import Observation


def _row(period, sales, *, stores=10, closings=1):
    return Observation(
        region_code="9999999999",
        region_code_system="test-h10",
        industry_code="fictional-industry",
        period=period,
        available_at=date(2025, 4, 15),
        source_kind="synthetic",
        source_dataset="constructed fixture, not an actual region",
        sales_amount=sales,
        store_count=stores,
        closing_count=closings,
    )


@pytest.mark.parametrize("current,expected", [(120, 0.2), (100, 0), (80, -0.2), (0, -1)])
def test_sales_change_uses_exact_previous_year_and_preserves_sign(current, expected):
    history = [_row("2024Q4", 900), _row("2025Q1", current), _row("2024Q1", 100)]
    result = observe_risk_indicators(history, "2025Q1")["sales_yoy_change"]
    assert result["status"] == "available"
    assert result["value"] == pytest.approx(expected)
    assert result["reason"] is None
    assert result["base_period"] == "2025Q1"
    assert result["comparison_period"] == "2024Q1"
    assert result["current_sales_amount"] == current
    assert result["comparison_sales_amount"] == 100


@pytest.mark.parametrize(
    "history,reason",
    [
        ([_row("2025Q1", None), _row("2024Q1", 100)], "missing_current_sales"),
        ([_row("2025Q1", 100)], "missing_comparison_sales"),
        ([_row("2025Q1", 100), _row("2024Q1", None)], "missing_comparison_sales"),
        ([_row("2025Q1", 100), _row("2024Q1", 0)], "nonpositive_comparison_sales"),
    ],
)
def test_sales_unavailable_does_not_disable_closure_burden(history, reason):
    result = observe_risk_indicators(history, "2025Q1")
    assert result["sales_yoy_change"]["status"] == "unavailable"
    assert result["sales_yoy_change"]["value"] is None
    assert result["sales_yoy_change"]["reason"] == reason
    assert result["closure_burden"]["status"] == "available"
    assert result["closure_burden"]["value"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "stores,closings,reason",
    [
        (None, 1, "missing_store_count"),
        (0, 0, "nonpositive_store_count"),
        (10, None, "missing_closing_count"),
        (10, 11, "closing_count_exceeds_store_count"),
    ],
)
def test_closure_unavailable_does_not_disable_sales(stores, closings, reason):
    history = [_row("2024Q1", 100), _row("2025Q1", 80, stores=stores, closings=closings)]
    result = observe_risk_indicators(history, "2025Q1")
    closure = result["closure_burden"]
    assert closure["status"] == "unavailable"
    assert closure["value"] is None
    assert closure["reason"] == reason
    assert closure["store_count"] == stores
    assert closure["closing_count"] == closings
    assert result["sales_yoy_change"]["status"] == "available"
    assert result["sales_yoy_change"]["value"] == pytest.approx(-0.2)


@pytest.mark.parametrize("closings,expected", [(0, 0), (10, 1)])
def test_zero_closures_and_matching_stock_are_observed_values(closings, expected):
    result = observe_risk_indicators([_row("2025Q1", None, closings=closings)], "2025Q1")
    closure = result["closure_burden"]
    assert closure["status"] == "available"
    assert closure["reason"] is None
    assert closure["value"] == expected


def test_overflow_returns_null_and_serializes_without_nonfinite_values():
    history = [_row("2024Q1", 1e-308), _row("2025Q1", 1e308)]
    result = observe_risk_indicators(history, "2025Q1")
    assert result["sales_yoy_change"]["status"] == "unavailable"
    assert result["sales_yoy_change"]["value"] is None
    assert result["sales_yoy_change"]["reason"] == "nonfinite_derived_indicator"
    assert result["closure_burden"]["status"] == "available"
    json.dumps(result, allow_nan=False)
