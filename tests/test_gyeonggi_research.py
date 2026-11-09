"""Synthetic checks for the Gyeonggi-only retrospective training path."""

import numpy as np
import pandas as pd
import pytest

from pickplace_ai.features import gyeonggi_four_quarter_features
from scripts.train_gyeonggi_research import make_frame, make_quarters


def test_features_use_four_chronological_quarters_and_reject_bad_denominators():
    frame = gyeonggi_four_quarter_features(np.array([[10, 20, 30, 60]]), 2)
    assert frame.iloc[0]["sales_relative_to_t_minus3"] == pytest.approx(5)
    assert frame.iloc[0]["sales_qoq"] == pytest.approx(1)
    assert frame.iloc[0]["sales_prior_qoq"] == pytest.approx(0.5)
    assert frame.iloc[0]["base_quarter"] == 2
    with pytest.raises(ValueError, match="denominators"):
        gyeonggi_four_quarter_features(np.array([[10, 0, 30, 60]]), 2)


def test_only_complete_quarters_and_duplicate_months_rejected():
    monthly = pd.DataFrame(
        [
            {"행정동코드": "4111111111", "중분류업종코드": "Q13", "기준년월": m, "매출금액": v}
            for m, v in [("202401", 1), ("202402", 2), ("202404", 3), ("202405", 4), ("202406", 5)]
        ]
    )
    quarters = make_quarters(monthly)
    assert pd.Period("2024Q1", freq="Q") not in quarters.columns
    assert quarters.iloc[0, 0] == 12
    with pytest.raises(ValueError, match="Duplicate"):
        make_quarters(pd.concat([monthly, monthly.iloc[[0]]]))


def test_frame_target_and_features_never_use_future_values():
    periods = pd.period_range("2024Q2", "2025Q2", freq="Q")
    scope = pd.MultiIndex.from_tuples(
        [("4111111111", "Q13")], names=["행정동코드", "중분류업종코드"]
    )
    quarters = pd.DataFrame([[10.0, 20.0, 30.0, 60.0, 80.0]], index=scope, columns=periods)
    frame = make_frame(quarters, "2025Q2")
    assert frame.iloc[0]["target"] == pytest.approx(7.0)
    assert frame.iloc[0]["sales_qoq"] == pytest.approx(1.0)
    quarters.loc[:, pd.Period("2025Q2", freq="Q")] = 90.0
    other = make_frame(quarters, "2025Q2")
    assert other.iloc[0]["target"] == pytest.approx(8.0)
    assert other.iloc[0]["sales_qoq"] == frame.iloc[0]["sales_qoq"]
    quarters.loc[:, pd.Period("2024Q3", freq="Q")] = np.nan
    with pytest.raises(ValueError, match="No eligible"):
        make_frame(quarters, "2025Q2")


def test_research_runner_preserves_holdout_and_refits_separately(tmp_path):
    import json
    from datetime import date

    from scripts.train_gyeonggi_research import sha, train

    records = []
    for month in pd.period_range("2024-04", "2026-06", freq="M"):
        records.append(
            {
                "기준년월": month.strftime("%Y%m"),
                "행정동코드": "4111111111",
                "중분류업종코드": "Q13",
                "매출금액": 10 + len(records),
            }
        )
    source = tmp_path / "candidate.csv"
    pd.DataFrame(records).to_csv(source, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "purpose": "review_only_not_training_or_prediction",
                "candidate_sha256": sha(source),
                "generated_at": "2026-11-09T06:01:10+00:00",
            }
        )
    )
    out = tmp_path / "out"
    result = train(source, manifest, out, date(2026, 11, 9))
    assert result["model_training_periods"] == {
        "train": "2025Q2",
        "validation": "2025Q4",
        "test": "2026Q2",
    }
    assert result["final_hgb_training_target_periods"] == [
        "2025Q2",
        "2025Q3",
        "2025Q4",
        "2026Q1",
        "2026Q2",
    ]
    assert result["final_hgb_training_rows"] == 5
    assert result["selection"] == min(
        result["validation_metrics"], key=lambda name: result["validation_metrics"][name]["mae_pp"]
    )
    assert result["snapshot_acquired_at"] == "2026-11-09"
    assert (out / "final-learned-model.joblib").is_file()
    with pytest.raises(FileExistsError):
        train(source, manifest, out, date(2026, 11, 9))


def test_research_prediction_checks_and_floor():
    from pickplace_ai.features import GYEONGGI_RESEARCH_FEATURES
    from scripts.train_gyeonggi_research import KIND, predict_research_model

    class FakeModel:
        def predict(self, X):
            return np.array([-2.0] * len(X))

    values = pd.DataFrame(
        [[1.0] * len(GYEONGGI_RESEARCH_FEATURES)], columns=GYEONGGI_RESEARCH_FEATURES
    )
    bundle = {
        "artifact_kind": KIND,
        "model": FakeModel(),
        "feature_columns": list(GYEONGGI_RESEARCH_FEATURES),
    }
    assert predict_research_model(bundle, values).tolist() == [-1.0]
    with pytest.raises(ValueError, match="column"):
        predict_research_model(bundle, values.iloc[:, ::-1])
    values.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        predict_research_model(bundle, values)
