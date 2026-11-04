from datetime import date

import joblib
import numpy as np
import pandas as pd
import pytest

from pickplace_ai.forecast_2026 import (
    candidates,
    fit_and_evaluate,
    model_matrix,
    predictions,
    year_rows,
)
from pickplace_ai.schemas import Observation
from scripts.forecast_2026 import quarter_status, run

SNAPSHOT = date(2026, 11, 3)


@pytest.fixture
def observations():
    return [
        Observation(
            region_code=region,
            region_code_system="seoul-adstrd",
            industry_code="CS100010",
            period=f"{year}Q{q}",
            available_at=SNAPSHOT,
            source_kind="official",
            source_dataset="fixture",
            sales_amount=1000 + 50 * (year - 2021) + q * 10 + int(region[-1]) * 50,
            store_count=10,
        )
        for region in ("11110515", "11110516")
        for year in range(2021, 2026)
        for q in range(1, 5)
    ]


def test_future_targets_never_enter_features(observations):
    original, _ = year_rows(observations, 2023, SNAPSHOT, labels=True)
    changed = [
        r.model_copy(update={"sales_amount": 999999999.0}) if r.period.startswith("2024") else r
        for r in observations
    ]
    altered, _ = year_rows(changed, 2023, SNAPSHOT, labels=True)
    assert not np.array_equal(original.target, altered.target)
    pd.testing.assert_frame_equal(model_matrix(original), model_matrix(altered))


def test_incomplete_history_and_zero_denominator_are_reported(observations):
    changed = [
        row.model_copy(update={"sales_amount": 0})
        if row.region_code == "11110515" and row.period == "2023Q2"
        else row
        for row in observations
        if not (row.region_code == "11110516" and row.period == "2022Q4")
    ]
    frame, omitted = year_rows(changed, 2023, SNAPSHOT, labels=False)
    assert len(frame) == 3
    assert omitted == {
        "insufficient_history": 4,
        "missing_or_nonpositive_comparison_sales": 1,
    }
    assert set(frame.comparison_period) == {"2023Q1", "2023Q3", "2023Q4"}


def test_validation_selection_does_not_use_test_labels(observations):
    train, _ = year_rows(observations, 2022, SNAPSHOT, labels=True)
    valid, _ = year_rows(observations, 2023, SNAPSHOT, labels=True)
    test, _ = year_rows(observations, 2024, SNAPSHOT, labels=True)
    first = fit_and_evaluate(train, valid, test)[0]
    altered = test.copy()
    altered.target = 10000
    assert fit_and_evaluate(train, valid, altered)[0] == first


def test_model_reloading_uses_same_feature_transform(observations, tmp_path):
    train, _ = year_rows(observations, 2022, SNAPSHOT, labels=True)
    estimate_frame, _ = year_rows(observations, 2025, SNAPSHOT, labels=False)
    model = candidates()["ridge"]
    model.fit(model_matrix(train), train.target)
    before = predictions(model, estimate_frame)
    path = tmp_path / "model.joblib"
    joblib.dump({"model": model}, path)
    np.testing.assert_allclose(before, predictions(joblib.load(path)["model"], estimate_frame))


def test_date_before_snapshot_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not precede"):
        run(
            tmp_path / "input",
            tmp_path / "output",
            SNAPSHOT,
            tmp_path / "manifest",
            tmp_path,
            date(2026, 11, 2),
        )


def test_completed_and_ongoing_quarters_are_labelled():
    assert quarter_status("2026Q3", date(2026, 11, 4)) == "completed_quarter_unobserved_estimate"
    assert quarter_status("2026Q4", date(2026, 11, 4)) == "in_progress_or_future_quarter_estimate"
