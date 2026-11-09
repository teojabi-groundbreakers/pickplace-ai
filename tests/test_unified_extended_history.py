"""Fictional observations verify source balancing, embargo and final refit."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts import train_gyeonggi_research as base
from scripts import train_unified_extended_history as experiment


class ZeroModel:
    fit_calls = []

    def fit(self, x, y, sample_weight=None):
        self.fit_calls.append(
            (x.copy(), np.asarray(sample_weight) if sample_weight is not None else None)
        )
        return self

    def predict(self, x):
        return np.zeros(len(x))


def test_embargo_excludes_adjacent_target_quarter():
    periods = ["2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4"]
    assert experiment.before_embargo(periods, "2025Q4") == ["2024Q4", "2025Q1", "2025Q2"]
    assert experiment.before_embargo(periods, "2025Q2") == ["2024Q4"]


def test_weighting_balances_unequal_source_rows(monkeypatch):
    from pickplace_ai.unified_features import UNIFIED_RESEARCH_FEATURES

    monkeypatch.setattr(experiment.unified, "model", ZeroModel)
    ZeroModel.fit_calls = []
    f = pd.DataFrame({column: [0.0, 1.0, 2.0] for column in UNIFIED_RESEARCH_FEATURES})
    f["target"] = 0.1
    f["source"] = ["seoul", "gyeonggi", "gyeonggi"]
    experiment.fit_joint(f.iloc[:1], f.iloc[1:])
    _, weights = ZeroModel.fit_calls[-1]
    assert weights[0] == pytest.approx(weights[1] + weights[2])
    with pytest.raises(ValueError, match="nonempty"):
        experiment.fit_joint(f.iloc[:0], f.iloc[1:])


def test_research_preserves_observed_data_and_separates_final_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment.unified, "model", ZeroModel)
    ZeroModel.fit_calls = []
    seoul = tmp_path / "seoul.csv"
    seoul_rows = [
        {
            "region_code": "11110515",
            "region_code_system": "seoul-adstrd",
            "industry_code": "CS100010",
            "period": str(period),
            "sales_unit": "KRW",
            "sales_amount": 1000 + n,
        }
        for n, period in enumerate(pd.period_range("2021Q2", "2026Q2", freq="Q"))
    ]
    pd.DataFrame(seoul_rows).to_csv(seoul, index=False)
    gg = tmp_path / "gyeonggi.csv"
    pd.DataFrame(
        [
            {
                "기준년월": m.strftime("%Y%m"),
                "행정동코드": "4111111111",
                "중분류업종코드": "Q13",
                "매출금액": 100 + n,
            }
            for n, m in enumerate(pd.period_range("2024-04", "2026-06", freq="M"))
        ]
    ).to_csv(gg, index=False)
    sm, gm = tmp_path / "sm.json", tmp_path / "gm.json"
    sm.write_text(
        json.dumps(
            {
                "output_sha256": base.sha(seoul),
                "historical_available_at": "2026-11-03",
                "api_sources": [{"acquired_at_utc": "2026-11-04T10:00:00+00:00"}],
            }
        )
    )
    gm.write_text(
        json.dumps(
            {
                "purpose": "review_only_not_training_or_prediction",
                "candidate_sha256": base.sha(gg),
                "generated_at": "2026-11-09T06:00:00+00:00",
            }
        )
    )
    old_seoul, old_gg = base.sha(seoul), base.sha(gg)
    out = tmp_path / "out"
    result = experiment.train(seoul, sm, gg, gm, out, date(2026, 11, 9))
    assert result["baseline_training_periods"] == {"seoul": ["2025Q2"], "gyeonggi": ["2025Q2"]}
    assert result["extended_training_periods"]["seoul"][-1] == "2025Q2"
    assert result["extended_training_periods"]["gyeonggi"] == ["2025Q2"]
    assert result["source_code_systems"] == experiment.unified.SOURCE_CODE_SYSTEMS
    assert result["source_scopes"] == {
        "seoul": [("11110515", "CS100010")],
        "gyeonggi": [("4111111111", "Q13")],
    }
    assert result["validation_embargo_target"] == "2025Q3"
    assert result["final_training_periods"]["seoul"][-1] == "2026Q2"
    assert result["final_training_periods"]["gyeonggi"][-1] == "2026Q2"
    assert result["final_training_rows"] == {"seoul": 17, "gyeonggi": 5}
    assert [x["target"] for x in result["rolling_seoul_only"]] == list(
        experiment.HISTORICAL_TARGETS
    )
    assert (out / "final-model.joblib").is_file()
    assert (base.sha(seoul), base.sha(gg)) == (old_seoul, old_gg)
    with pytest.raises(FileExistsError):
        experiment.train(seoul, sm, gg, gm, out, date(2026, 11, 9))
