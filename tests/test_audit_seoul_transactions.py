"""Synthetic tests: derived values do not invent missing data."""

import pandas as pd
import pytest

from scripts.audit_seoul_transactions import KEYS, TIME, guard_diagnostics, positive_ratio, record


def test_count_zero_or_missing_is_not_converted_to_zero():
    assert positive_ratio(10, 0) is None
    assert positive_ratio(10, None) is None
    assert positive_ratio(10, 2) == 5


def test_distribution_uses_exact_raw_fields_and_handles_missing():
    row = {
        "기준_년분기_코드": "20254",
        KEYS[0]: "11710620",
        KEYS[1]: "CS300019",
        "region_name_2025_reference": "동",
        "industry_name_2025_reference": "업종",
        "당월_매출_금액": 100,
        "당월_매출_건수": 4,
        "주중_매출_금액": 80,
    }
    row.update(dict(zip(TIME, [10, 5, 20, 50, 10, 5], strict=True)))
    output = record(pd.Series(row))
    assert output["amount_raw"] == 100
    assert output["amount_divided_by_count"] == 25
    assert output["weekday_amount_fraction"] == pytest.approx(0.8)
    assert output["time_amount_sum_divided_by_total"] == pytest.approx(1)
    assert output["top_time_amount_fraction"] == pytest.approx(0.5)
    row[TIME[1]] = None
    assert record(pd.Series(row))["top_time_amount_fraction"] is None


def test_guard_overlaps_and_retrospective_return_are_reported_separately():
    import numpy as np

    values = np.array([[1_000_000, 20_000_000, 2_000_000, 2_000_000, 1_000_000]])
    result = guard_diagnostics(values, np.array([10.0]))
    assert result["flagged_rows"] == 1
    assert result["overlapping_signals"]["reference_low"]["overlap_rows"] == 1
    assert result["prior_surge_rows"] == 1
    assert result["return_after_surge_rows"] == 0
