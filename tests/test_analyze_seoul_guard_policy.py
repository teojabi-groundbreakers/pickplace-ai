"""Synthetic validation of descriptive statistics, group attribution and dates."""

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_seoul_guard_policy as study


def test_summarize_conditional_rates_and_exact_boundaries():
    result = study.summarize(np.array([0.1, 0.2, 0.5, 0.51]), np.array([1, 1, 0, 0], bool))
    assert result["all"]["rows"] == 4
    assert result["withheld"]["error_le_10pp_rows"] == 1
    assert result["retained"]["error_gt_50pp_rows"] == 1
    assert result["retained"]["fraction_pct"] == 50.0
    assert result["retained"]["mae_pp"] == pytest.approx(50.5)
    assert result["all"]["mae_pp"] == pytest.approx(32.75)
    with pytest.raises(ValueError):
        study.summarize(np.array([np.nan]), np.array([True]))
    with pytest.raises(ValueError):
        study.summarize(np.array([0.2]), np.array([True, False]))


def test_group_names_missing_names_and_no_retained_are_preserved():
    frame = pd.DataFrame(
        {
            "code": ["1", "1", "2"],
            "name": ["역동", "역동", None],
            "absolute_error": [0.1, 0.6, 0.2],
            "withheld": [True, False, True],
        }
    )
    group = study.split_groups(frame, "code", "name").set_index("code")
    assert group.loc["1", "rows"] == 2
    assert group.loc["1", "withheld_error_le_10pp_rows"] == 1
    assert group.loc["1", "retained_error_gt_50pp_rows"] == 1
    assert group.loc["2", "name_2025_reference"] == ""
    assert pd.isna(group.loc["2", "retained_mae_pp"])


def test_rolling_cutoff_has_one_quarter_gap():
    assert study.rolling_cutoff("2025Q4") == ("2025Q2", "2025Q3")
    assert study.rolling_cutoff("2024Q1") == ("2023Q3", "2023Q4")


def test_store_history_signal_keeps_missing_and_zero_denominators_unavailable():
    histories = np.array([[10, 20, 20, 0], [0, 10, 10, 10], [10, 10, np.nan, 10]])
    valid, change = study.store_history_signal(histories)
    assert valid.tolist() == [True, False, False]
    assert change[0] == pytest.approx(1.0)
    assert np.isnan(change[1]) and np.isnan(change[2])
    with pytest.raises(ValueError):
        study.store_history_signal(np.ones((2, 3)))


def test_store_history_never_reads_target_quarter():
    periods = pd.period_range("2024Q2", "2025Q2", freq="Q")
    index = pd.MultiIndex.from_tuples(
        [("4111111111", "CS100001")], names=["행정동코드", "중분류업종코드"]
    )
    stores = pd.DataFrame([[10, 11, 12, 13, 99]], index=index, columns=periods)
    frame = pd.DataFrame([{"행정동코드": "4111111111", "중분류업종코드": "CS100001"}])
    first = study.aligned_store_history(stores, frame, "2025Q2")
    stores.loc[index[0], periods[-1]] = 100000
    later = study.aligned_store_history(stores, frame, "2025Q2")
    assert np.array_equal(first, later)
    assert first.tolist() == [[10, 11, 12, 13]]


def test_fixed_research_flags_match_actual_api_policy():
    from pickplace_ai import seoul_guard_policy as policy
    from scripts import audit_seoul_guard as guard

    history = np.array([[1000000, 20000000, 2000000, 2000000], [20000000] * 4])
    thresholds = (
        policy.LOW_REFERENCE_SALES_KRW,
        policy.MAX_QOQ_INCREASE,
        policy.MIN_QOQ_DECREASE,
        policy.LOW_LATEST_SALES_KRW,
    )
    flags = guard.guard_flags(history, *thresholds)
    study.ensure_api_flags_match(history, flags)
    flags["low_latest"][0] = False
    with pytest.raises(ValueError, match="do not match"):
        study.ensure_api_flags_match(history, flags)
