"""Synthetic checks for source handling and pre-target store features."""

import numpy as np
import pandas as pd
import pytest

from scripts import study_seoul_store_features as study


def row(region, period, count, opening=0.0, closing=0.0):
    return {
        "region_code": region,
        "region_code_system": "seoul-adstrd",
        "industry_code": "CS100010",
        "period": period,
        "store_count": count,
        "opening_count": opening,
        "closing_count": closing,
    }


def test_features_use_only_two_prior_quarters_and_never_drop_scope():
    source = pd.DataFrame(
        [
            row("11110515", "2024Q4", 10),
            row("11110515", "2025Q1", 12, 2, 1),
            row("11110515", "2025Q2", 9999, 999, 888),
            row("11110530", "2025Q1", 0),
        ]
    )
    frame = pd.DataFrame(
        [
            {"행정동코드": "11110515", "중분류업종코드": "CS100010", "source": "seoul"},
            {"행정동코드": "11110530", "중분류업종코드": "CS100010", "source": "seoul"},
            {"행정동코드": "4111111111", "중분류업종코드": "Q13", "source": "gyeonggi"},
        ]
    )
    original = frame.copy(deep=True)
    result = study.add_store_features(frame, study.store_lookup(source), "2025Q2")
    assert len(result) == len(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert result.loc[0, "store_current"] == 12
    assert result.loc[0, "store_previous"] == 10
    assert result.loc[0, "store_change"] == pytest.approx(0.2)
    assert result.loc[0, "opening_share"] == pytest.approx(2 / 12)
    assert result.loc[0, "closing_share"] == pytest.approx(1 / 12)
    assert result.loc[1, "store_current"] == 0
    assert (
        result.loc[1, ["store_previous", "store_change", "opening_share", "closing_share"]]
        .isna()
        .all()
    )
    assert result.loc[2, list(study.STORE_FEATURES)].isna().all()
    source.loc[source["period"] == "2025Q2", "store_count"] = 1
    changed_target = study.add_store_features(frame, study.store_lookup(source), "2025Q2")
    pd.testing.assert_frame_equal(result, changed_target)


def test_store_lookup_rejects_duplicate_or_negative_and_preserves_missing():
    rows = pd.DataFrame([row("11110515", "2025Q1", None)])
    assert np.isnan(study.store_lookup(rows).iloc[0]["store_count"])
    with pytest.raises(ValueError, match="Duplicate"):
        study.store_lookup(pd.concat([rows, rows]))
    rows.loc[0, "store_count"] = -1
    with pytest.raises(ValueError, match="Invalid store source amount"):
        study.store_lookup(rows)
    rows.loc[0, "store_count"] = np.inf
    with pytest.raises(ValueError, match="Invalid store source amount"):
        study.store_lookup(rows)


def test_research_targets_are_fixed_before_reused_test():
    assert study.PERIODS == {"train": "2025Q2", "validation": "2025Q4"}
    assert "2026Q2" not in study.PERIODS.values()
