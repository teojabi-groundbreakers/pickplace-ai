"""Synthetic unit and end-to-end checks for pooled growth research."""

import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import pytest

from pickplace_ai.features import GYEONGGI_RESEARCH_FEATURES
from pickplace_ai.unified_features import unified_research_features
from scripts import train_unified_growth as model
from scripts.train_gyeonggi_research import sha


def test_acquisition_dates_use_seoul_calendar():
    assert model.acquisition_day("2026-11-04T23:30:00+00:00") == date(2026, 11, 5)
    with pytest.raises(ValueError, match="timezone"):
        model.acquisition_day("2026-11-04T23:30:00")


def test_same_history_is_not_unit_or_code_converted():
    seoul = unified_research_features(np.array([[10, 20, 30, 60]]), 2, "seoul")
    gyeonggi = unified_research_features(np.array([[1000, 2000, 3000, 6000]]), 2, "gyeonggi")
    np.testing.assert_allclose(
        seoul[list(GYEONGGI_RESEARCH_FEATURES)], gyeonggi[list(GYEONGGI_RESEARCH_FEATURES)]
    )
    assert seoul.source_gyeonggi.iloc[0] == 0
    assert gyeonggi.source_gyeonggi.iloc[0] == 1
    with pytest.raises(ValueError, match="source"):
        unified_research_features(np.ones((1, 4)), 1, "invalid")


def test_predict_rejects_wrong_schema_and_uses_floor():
    class Estimator:
        def predict(self, features):
            return np.full(len(features), -2.0)

    features = unified_research_features(np.ones((1, 4)), 1, "seoul")
    bundle = model.bundle(Estimator())
    assert model.predict_unified_model(bundle, features).tolist() == [-1.0]
    with pytest.raises(ValueError, match="order"):
        model.predict_unified_model(bundle, features.iloc[:, ::-1])
    with pytest.raises(ValueError, match="kind"):
        model.predict_unified_model({**bundle, "artifact_kind": "other"}, features)


def test_fit_both_sources_but_keep_scopes_and_dates(tmp_path):
    # Fictional data includes 2024Q2 through 2026Q2, with no shared source IDs.
    seoul, seoul_manifest = tmp_path / "seoul.csv", tmp_path / "seoul.json"
    gyeonggi, gyeonggi_manifest = tmp_path / "gyeonggi.csv", tmp_path / "gyeonggi.json"
    pd.DataFrame(
        [
            {
                "region_code": "11110000",
                "region_code_system": "seoul-adstrd",
                "industry_code": "CS1",
                "period": str(period),
                "sales_unit": "KRW",
                "sales_amount": 100 + i,
            }
            for i, period in enumerate(pd.period_range("2024Q2", "2026Q2", freq="Q"))
        ]
    ).to_csv(seoul, index=False)
    pd.DataFrame(
        [
            {
                "행정동코드": "4111111111",
                "중분류업종코드": "Q13",
                "기준년월": period.strftime("%Y%m"),
                "매출금액": 10 + i,
            }
            for i, period in enumerate(pd.period_range("2024-04", "2026-06", freq="M"))
        ]
    ).to_csv(gyeonggi, index=False)
    seoul_manifest.write_text(
        json.dumps(
            {
                "output_sha256": sha(seoul),
                "historical_available_at": "2026-11-03",
                "api_sources": [{"acquired_at_utc": "2026-11-04T12:00:00+00:00"}],
            }
        )
    )
    gyeonggi_manifest.write_text(
        json.dumps(
            {
                "purpose": "review_only_not_training_or_prediction",
                "candidate_sha256": sha(gyeonggi),
                "generated_at": "2026-11-09T06:01:10+00:00",
            }
        )
    )
    out = tmp_path / "out"
    kwargs = dict(
        seoul=seoul,
        seoul_manifest=seoul_manifest,
        gyeonggi=gyeonggi,
        gyeonggi_manifest=gyeonggi_manifest,
        output=out,
    )
    with pytest.raises(ValueError, match="acquisition"):
        model.train(**kwargs, snapshot=date(2026, 11, 8))
    result = model.train(**kwargs, snapshot=date(2026, 11, 9))
    assert result["final_rows_by_source"] == {"seoul": 5, "gyeonggi": 5}
    assert result["source_scopes"]["seoul"] == [("11110000", "CS1")]
    assert result["source_scopes"]["gyeonggi"] == [("4111111111", "Q13")]
    assert result["snapshot_review_date"] == "2026-11-09"
    assert result["source_known_by"] == {"seoul": "2026-11-04", "gyeonggi": "2026-11-09"}
    assert result["test_reused"] is True
    assert result["periods"]["reused_test"] == "2026Q2"
    bundle = joblib.load(out / "final-model.joblib")
    assert bundle["artifact_kind"] == model.KIND
    prediction = model.predict_unified_model(
        bundle, unified_research_features(np.ones((1, 4)), 2, "seoul")
    )
    assert np.isfinite(prediction).all()
    with pytest.raises(FileExistsError):
        model.train(**kwargs, snapshot=date(2026, 11, 9))
