"""Synthetic guards for Seoul diagnostic grouping and data boundaries."""

import numpy as np
import pandas as pd
import pytest

from scripts.analyze_unified_seoul import (
    denominator_bins,
    eligibility_counts,
    exploratory_threshold_stats,
    read_name_maps,
)


def test_denominator_boundaries_are_disjoint_and_nonpositive_rejected():
    data = pd.Series([0.1, 999_999, 1_000_000, 9_999_999, 10_000_000, 100_000_000, 1_000_000_000])
    assert denominator_bins(data).astype(str).tolist() == [
        "(0,1M)",
        "(0,1M)",
        "[1M,10M)",
        "[1M,10M)",
        "[10M,100M)",
        "[100M,1B)",
        "[1B,infinity)",
    ]
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            denominator_bins(pd.Series([bad]))


def test_ambiguous_names_preserved_and_missing_history_counted(tmp_path):
    path = tmp_path / "names.csv"
    pd.DataFrame(
        [
            ["11110515", "첫이름", "CS100010", "음료"],
            ["11110515", "옛이름", "CS100010", "카페"],
            ["11110516", "다른동", "CS100011", "서점"],
        ],
        columns=["행정동_코드", "행정동_코드_명", "서비스_업종_코드", "서비스_업종_코드_명"],
    ).to_csv(path, index=False, encoding="cp949")
    regions, industries = read_name_maps(path)
    assert set(regions["11110515"].split(" / ")) == {"첫이름", "옛이름"}
    assert set(industries["CS100010"].split(" / ")) == {"음료", "카페"}
    assert regions["11110516"] == "다른동"
    quarters = pd.period_range("2024Q4", "2025Q4", freq="Q")
    frame = pd.DataFrame(
        [[1, 2, 3, 4, 5], [np.nan, 2, 3, 4, 5], [0, 2, 3, 4, 5], [1, 2, 3, 4, np.nan]],
        columns=quarters,
    )
    assert eligibility_counts(frame, pd.Period("2025Q4", freq="Q")) == (3, 1, 1)


def test_posthoc_threshold_slices_preserve_rows_and_compare_zero_baseline():
    frame = pd.DataFrame(
        {
            "comparison_sales_krw": [10, 20, 30],
            "absolute_error": [0.1, 0.2, 0.4],
            "target": [0.4, -0.5, 1.0],
        }
    )
    result = exploratory_threshold_stats(frame, 20)
    assert result["below"]["rows"] == 1
    assert result["below"]["model_mae_pp"] == pytest.approx(10)
    assert result["at_or_above"]["rows"] == 2
    assert result["at_or_above"]["model_mae_pp"] == pytest.approx(30)
    assert result["at_or_above"]["zero_growth_mae_pp"] == pytest.approx(75)
    assert sum(item["coverage_pct"] for item in result.values()) == pytest.approx(100)
