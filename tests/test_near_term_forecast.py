from datetime import date

import numpy as np
import pandas as pd
import pytest

from pickplace_ai.schemas import Observation
from pickplace_ai.training import RecentGrowthRegressor
from scripts.near_term_forecast import (
    build_frames,
    matrix,
    predict,
    scope_training_flags,
    select_and_test,
)

AS_OF = date(2026, 11, 4)


@pytest.fixture
def observations():
    return [
        Observation(
            region_code=f"1111051{r}",
            region_code_system="seoul-adstrd",
            industry_code="CS100010",
            period=f"{year}Q{q}",
            available_at=date(2026, 11, 3) if year < 2026 else AS_OF,
            source_kind="official",
            source_dataset="fixture",
            sales_amount=1000 + (year - 2021) * 100 + q * 20 + r * 10,
            store_count=10,
        )
        for r in (5, 6)
        for year in range(2021, 2027)
        for q in range(1, (2 if year == 2026 else 4) + 1)
    ]


def test_future_targets_cannot_change_predictor_features(observations):
    original, _ = build_frames(observations, AS_OF, ["2025Q2"])
    altered = [
        obs.model_copy(update={"sales_amount": 9_999_999.0})
        if obs.period in {"2025Q3", "2025Q4"}
        else obs
        for obs in observations
    ]
    changed, _ = build_frames(altered, AS_OF, ["2025Q2"])
    pd.testing.assert_frame_equal(matrix(original), matrix(changed))
    assert not np.array_equal(original.target.to_numpy(), changed.target.to_numpy())


def test_missing_quarter_is_not_filled_and_mixed_acquisition_dates_work(observations):
    filtered = [
        obs
        for obs in observations
        if not (obs.region_code == "11110515" and obs.period == "2025Q4")
    ]
    rows, omitted = build_frames(filtered, AS_OF, ["2026Q2"])
    assert len(rows) == 2
    assert omitted["2026Q2:insufficient_history"] == 2
    assert set(rows.target_period) == {"2026Q3", "2026Q4"}
    with pytest.raises(ValueError, match="unavailable"):
        build_frames(observations, date(2026, 11, 3), ["2026Q2"])


def test_recent_growth_baseline_is_raw_ratio_not_log(observations):
    frame, _ = build_frames(observations, AS_OF, ["2025Q2"])
    model = RecentGrowthRegressor().fit(matrix(frame), frame.target)
    np.testing.assert_allclose(predict(model, frame), frame.sales_yoy)


def test_test_labels_do_not_affect_validation_selection(observations):
    origins = [f"{year}Q{q}" for year in range(2022, 2027) for q in range(1, 5)]
    frame, _ = build_frames(observations, AS_OF, origins)
    first = select_and_test(frame, 1)[0]
    modified = frame.copy()
    modified.loc[modified.base_period == "2025Q2", "target"] = 10_000
    assert select_and_test(modified, 1)[0] == first


def test_training_scope_flag_uses_region_and_industry_together():
    training = pd.DataFrame(
        [
            {
                "region_code_system": "seoul-adstrd",
                "region_code": "11110515",
                "industry_code": "CS100010",
            }
        ]
    )
    forecast = pd.DataFrame(
        [
            {
                "region_code_system": "seoul-adstrd",
                "region_code": "11110515",
                "industry_code": "CS100010",
            },
            {
                "region_code_system": "seoul-adstrd",
                "region_code": "11110516",
                "industry_code": "CS100010",
            },
            {
                "region_code_system": "seoul-adstrd",
                "region_code": "11110515",
                "industry_code": "CS100001",
            },
        ]
    )
    assert scope_training_flags(training, forecast) == [True, False, False]
