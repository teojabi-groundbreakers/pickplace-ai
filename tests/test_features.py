from datetime import date, timedelta

import pytest

from pickplace_ai.demo import synthetic_observations
from pickplace_ai.features import FeatureUnavailable, build_training_frame, feature_record
from pickplace_ai.periods import shift_quarter
from pickplace_ai.schemas import PredictionRequest


def test_label_is_next_quarter_over_same_quarter_previous_year():
    rows = synthetic_observations(regions=1, quarters=12)
    frame = build_training_frame(rows)
    base = frame.iloc[0]
    by_period = {row.period: row for row in rows}
    assert base["target"] == pytest.approx(
        by_period[shift_quarter(base.base_period, 1)].sales_amount
        / by_period[shift_quarter(base.base_period, -3)].sales_amount
        - 1
    )


def test_feature_record_does_not_look_at_future_sales():
    rows = synthetic_observations(regions=1, quarters=12)
    current = rows[6]
    original = feature_record(rows, current.period, current.available_at)
    modified = [
        row.model_copy(update={"sales_amount": 1e18}) if row.period > current.period else row
        for row in rows
    ]
    assert original == feature_record(modified, current.period, current.available_at)


def test_calendar_gap_is_not_compressed_into_adjacent_rows():
    rows = synthetic_observations(regions=1, quarters=12)
    current = rows[6]
    gap = [row for row in rows if row.period != rows[4].period]
    with pytest.raises(FeatureUnavailable, match="insufficient_history"):
        feature_record(gap, current.period, current.available_at)


def test_late_published_history_cannot_leak_into_features():
    rows = synthetic_observations(regions=1, quarters=12)
    current = rows[6]
    rows[4] = rows[4].model_copy(update={"available_at": current.available_at + timedelta(days=1)})
    with pytest.raises(FeatureUnavailable, match="insufficient_history"):
        feature_record(rows, current.period, current.available_at)


def test_zero_comparison_sales_excludes_training_label():
    rows = synthetic_observations(regions=1, quarters=12)
    rows[2] = rows[2].model_copy(update={"sales_amount": 0.0})
    frame = build_training_frame(rows)
    assert shift_quarter(rows[2].period, 4) not in set(frame.target_period)


def test_api_request_preserves_codes_and_rejects_future_availability():
    rows = synthetic_observations(regions=1, quarters=12)
    last = rows[-1]
    payload = dict(
        region_code=last.region_code,
        region_code_system=last.region_code_system,
        industry_code=last.industry_code,
        base_period=last.period,
        as_of=last.available_at - timedelta(days=1),
        source_kind="synthetic",
        history=rows,
    )
    with pytest.raises(ValueError, match="not available"):
        PredictionRequest(**payload)


def test_latest_download_date_does_not_fabricate_historical_forecast_origins():
    rows = synthetic_observations(regions=1, quarters=12)
    rows = [row.model_copy(update={"available_at": date(2028, 1, 1)}) for row in rows]
    assert build_training_frame(rows).empty
