"""3분기 모델의 학습 시점, 적용 경계와 재현 검증을 확인한다."""

import numpy as np
import pandas as pd
import pytest

from scripts import train_seoul_q3_huber_pair as pair


def test_split_uses_one_quarter_lag_and_rejects_already_observed_targets():
    fields = {
        "region_code_system": "seoul-adstrd",
        "region_code": "11110530",
        "horizon": 1,
    }
    frame = pd.DataFrame(
        [
            {**fields, "industry_code": "CS100001", "target_period": period, "target": 0.1}
            for period in ("2026Q1", "2026Q2")
        ]
        + [
            {
                **fields,
                "industry_code": industry,
                "base_period": pair.BASE,
                "target_period": pair.TARGET,
                "comparison_period": pair.COMPARISON,
                "target": np.nan,
            }
            for industry in pair.rolling.SERVICE_INDUSTRIES
        ]
    )
    train, forecast = pair.split_training(frame)
    assert train.target_period.tolist() == ["2026Q1"]
    assert len(forecast) == 3
    frame.loc[frame.target_period.eq(pair.TARGET), "target"] = 0.0
    with pytest.raises(ValueError, match="unobserved"):
        pair.split_training(frame)


def test_pair_uses_transformed_model_strictly_below_threshold(monkeypatch):
    forecast = pd.DataFrame(
        {
            "region_code_system": ["seoul-adstrd"] * 2,
            "region_code": ["11110530", "11110550"],
            "industry_code": ["CS100010"] * 2,
            "base_period": [pair.BASE] * 2,
            "target_period": [pair.TARGET] * 2,
            "comparison_period": [pair.COMPARISON] * 2,
            "comparison_sales": [pair.CUTOFF_KRW - 1, pair.CUTOFF_KRW],
        }
    )
    monkeypatch.setattr(pair.near, "predict", lambda model, frame: np.repeat(model, len(frame)))
    result = pair.predict_pair({pair.tail.RAW: 0.15, pair.tail.ASINH: 0.05}, forecast)
    assert result.uses_transformed_model.tolist() == [True, False]
    assert result[pair.CHALLENGER].tolist() == [0.05, 0.15]


def test_replay_rejects_changed_comparison_sales():
    frozen = pd.DataFrame({"comparison_sales": [100], pair.tail.RAW: [0.1], pair.CHALLENGER: [0.2]})
    pair.require_matching_predictions(frozen, frozen)
    with pytest.raises(ValueError, match="differ from the frozen comparison"):
        pair.require_matching_predictions(frozen.assign(comparison_sales=101), frozen)
