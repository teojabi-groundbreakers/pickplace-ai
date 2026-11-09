"""Synthetic tests for predeclared training-only exclusion."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts import study_seoul_train_outliers as study
from scripts import train_gyeonggi_research as base


class ZeroModel:
    def fit(self, x, y, sample_weight=None):
        return self

    def predict(self, x):
        return np.zeros(len(x))


def test_cutoffs_remove_only_strictly_outside_train_quantiles():
    cuts, masks = study.masks_for_training_target(np.array([-1, 0, 0, 0, 0, 100]))
    assert cuts["p01"] > -1 and cuts["p99"] < 100
    assert masks["all"].tolist() == [True] * 6
    assert masks["exclude_above_p99"].tolist() == [True, True, True, True, True, False]
    assert masks["exclude_outside_p01_p99"].tolist() == [False, True, True, True, True, False]
    _, tied = study.masks_for_training_target(np.zeros(6))
    assert all(mask.all() for mask in tied.values())
    for bad in ([], [float("nan")], [-1.01]):
        with pytest.raises(ValueError):
            study.masks_for_training_target(np.array(bad))


def test_experiment_preserves_all_validation_rows_and_source_files(tmp_path, monkeypatch):
    monkeypatch.setattr(study.unified, "model", ZeroModel)
    seoul, gg = tmp_path / "seoul.csv", tmp_path / "gg.csv"
    rows = []
    for region_index, region in enumerate(("11110515", "11110530")):
        rows.extend(
            {
                "region_code": region,
                "region_code_system": "seoul-adstrd",
                "industry_code": "CS100010",
                "period": str(period),
                "sales_unit": "KRW",
                "sales_amount": 1000 + n * (n + region_index + 1),
            }
            for n, period in enumerate(pd.period_range("2021Q2", "2026Q2", freq="Q"))
        )
    pd.DataFrame(rows).to_csv(seoul, index=False)
    pd.DataFrame(
        [
            {
                "기준년월": month.strftime("%Y%m"),
                "행정동코드": "4111111111",
                "중분류업종코드": "Q13",
                "매출금액": 100 + n,
            }
            for n, month in enumerate(pd.period_range("2024-04", "2026-06", freq="M"))
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
    original = (base.sha(seoul), base.sha(gg))
    out = tmp_path / "study"
    result = study.train(seoul, sm, gg, gm, out, date(2026, 11, 9))
    assert (base.sha(seoul), base.sha(gg)) == original
    assert result["validation_filter"] == "none"
    assert result["seoul_train_target_periods"][-1] == "2025Q2"
    assert result["embargo_target_period"] == "2025Q3"
    assert result["validation_target_period"] == "2025Q4"
    assert result["comparisons"]["exclude_above_p99"]["seoul_excluded_rows"] > 0
    for entry in result["comparisons"].values():
        assert entry["gyeonggi_excluded_rows"] == 0
        assert entry["validation_rows_excluded"] == 0
        assert entry["seoul_validation"]["rows"] == 2
        assert entry["gyeonggi_validation"]["rows"] == 1
    assert len(result["model_sha256"]) == 3
    with pytest.raises(FileExistsError):
        study.train(seoul, sm, gg, gm, out, date(2026, 11, 9))
