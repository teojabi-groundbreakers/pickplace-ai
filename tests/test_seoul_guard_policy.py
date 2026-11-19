"""Research withholding is based only on observable Seoul input history."""

import numpy as np
import pytest

from pickplace_ai import seoul_guard_policy as guard


def test_strict_low_and_abrupt_movement_boundaries():
    low = guard.LOW_REFERENCE_SALES_KRW
    assert guard.assess_sales_history([low, low, low, low]) == []
    assert guard.assess_sales_history([low - 1, low - 1, low - 1, low - 1]) == [
        "low_reference_sales"
    ]
    start = 20_000_000.0
    increase_end = start * (1 + guard.MAX_QOQ_INCREASE)
    assert "rapid_sales_increase" not in guard.assess_sales_history(
        [start, increase_end, increase_end, increase_end]
    )
    assert "rapid_sales_increase" in guard.assess_sales_history(
        [start, increase_end * 1.000001, increase_end * 1.000001, increase_end * 1.000001]
    )
    decrease_end = start * (1 + guard.MIN_QOQ_DECREASE)
    assert "rapid_sales_decrease" not in guard.assess_sales_history(
        [start, decrease_end, decrease_end, decrease_end]
    )
    assert "rapid_sales_decrease" in guard.assess_sales_history(
        [start, decrease_end * 0.999999, decrease_end * 0.999999, decrease_end * 0.999999]
    )


def test_multiple_signals_and_invalid_values():
    assert guard.assess_sales_history([1_000_000, 20_000_000, 2_000_000, 2_000_000]) == [
        "low_reference_sales",
        "low_latest_sales",
        "rapid_sales_increase",
        "rapid_sales_decrease",
    ]
    with pytest.raises(ValueError):
        guard.assess_sales_history([1, 0, 1, 1])
    with pytest.raises(ValueError):
        guard.assess_sales_history([1, 1, np.nan, 1])


def test_policy_is_bound_to_exact_training_source_and_model():
    meta = {
        "source_hashes": {"seoul": guard.SOURCE_SHA256},
        "model_sha256": {"evaluation-pooled.joblib": guard.EVALUATION_MODEL_SHA256},
    }
    assert guard.policy_compatible(meta, guard.FINAL_MODEL_SHA256)
    assert not guard.policy_compatible(meta, "0" * 64)
    meta["source_hashes"]["seoul"] = "0" * 64
    assert not guard.policy_compatible(meta, guard.FINAL_MODEL_SHA256)


def test_latest_sales_threshold_and_zero_are_separate_signals():
    latest = guard.LOW_LATEST_SALES_KRW
    reference = guard.LOW_REFERENCE_SALES_KRW
    assert guard.assess_sales_history([reference, reference, reference, latest]) == []
    assert "low_latest_sales" in guard.assess_sales_history(
        [reference, reference, reference, latest - 1]
    )
    zero = guard.assess_sales_history([reference, reference, reference, 0])
    assert "low_latest_sales" in zero and "rapid_sales_decrease" in zero


def test_explanation_follows_signal_order_and_uses_only_input_periods():
    periods = ["2025Q3", "2025Q4", "2026Q1", "2026Q2"]
    values = [1_000_000, 20_000_000, 2_000_000, 2_000_000]
    details = guard.describe_sales_history(values, periods)
    assert [d["signal"] for d in details] == guard.assess_sales_history(values)
    assert details[0]["to_period"] == "2025Q3" and details[0]["observed_value"] == 1_000_000
    assert details[1]["to_period"] == "2026Q2" and details[1]["observed_value"] == 2_000_000
    assert details[2]["from_period"] == "2025Q3" and details[2]["to_period"] == "2025Q4"
    assert details[2]["observed_value"] == pytest.approx(19.0)
    assert details[2]["unit"] == "ratio" and details[2]["comparison"] == "above"
    assert details[3]["from_period"] == "2025Q4" and details[3]["to_period"] == "2026Q1"
    assert details[3]["observed_value"] == pytest.approx(-0.9)
    assert details[3]["unit"] == "ratio" and details[3]["comparison"] == "below"
    assert values == [1_000_000, 20_000_000, 2_000_000, 2_000_000]


def test_explanation_reports_no_extra_signal_and_breaks_ties_in_time_order():
    periods = ["2025Q3", "2025Q4", "2026Q1", "2026Q2"]
    assert guard.describe_sales_history([20_000_000] * 4, periods) == []
    details = guard.describe_sales_history(
        [20_000_000, 100_000_000, 500_000_000, 500_000_000], periods
    )
    assert details[0]["signal"] == "rapid_sales_increase"
    assert details[0]["from_period"] == "2025Q3" and details[0]["to_period"] == "2025Q4"
    with pytest.raises(ValueError):
        guard.describe_sales_history([20_000_000] * 4, periods[:3])


@pytest.mark.parametrize(
    "periods",
    [
        ["2026Q2", "2026Q1", "2025Q4", "2025Q3"],
        ["2025Q3", "2025Q3", "2025Q4", "2026Q1"],
        ["2025Q3", "2025Q4", "2026Q2", "2026Q3"],
        ["2025Q3", "2025Q4", "2026Q0", "2026Q1"],
    ],
)
def test_explanation_rejects_nonconsecutive_periods(periods):
    with pytest.raises(ValueError):
        guard.describe_sales_history([20_000_000] * 4, periods)
