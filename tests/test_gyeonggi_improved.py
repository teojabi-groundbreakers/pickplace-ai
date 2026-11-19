"""Synthetic checks of model selection and research-only artifact safety."""

import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.train_gyeonggi_improved import train
from scripts.train_gyeonggi_research import sha


def fixture(tmp_path, name, last_month_amount):
    records = []
    for region in ("4111111111", "4111111112"):
        for i, month in enumerate(pd.period_range("2024-04", "2026-06", freq="M")):
            records.append(
                {
                    "기준년월": month.strftime("%Y%m"),
                    "행정동코드": region,
                    "중분류업종코드": "Q13",
                    "매출금액": last_month_amount if month.strftime("%Y%m") == "202606" else 10 + i,
                }
            )
    source = tmp_path / f"{name}.csv"
    pd.DataFrame(records).to_csv(source, index=False)
    manifest = tmp_path / f"{name}.json"
    manifest.write_text(
        json.dumps(
            {
                "purpose": "review_only_not_training_or_prediction",
                "candidate_sha256": sha(source),
                "generated_at": "2026-11-09T06:01:10+00:00",
            }
        )
    )
    return source, manifest


def test_selection_uses_only_validation_and_final_refit_is_separate(tmp_path):
    first, first_manifest = fixture(tmp_path, "first", 40)
    second, second_manifest = fixture(tmp_path, "second", 4000)
    result1 = train(first, first_manifest, tmp_path / "out1", date(2026, 11, 9))
    result2 = train(second, second_manifest, tmp_path / "out2", date(2026, 11, 9))
    assert result1["selection"] == result2["selection"]
    assert result1["validation_metrics"] == result2["validation_metrics"]
    assert result1["reused_test_metrics"] != result2["reused_test_metrics"]
    assert result1["test_reused"] is True
    assert result1["selection"] == min(
        result1["validation_metrics"], key=lambda k: result1["validation_metrics"][k]["mae_pp"]
    )
    assert result1["final_training_rows"] == 10
    assert result1["final_training_target_periods"][-1] == "2026Q2"
    assert result1["past_publication_dates_verified"] is False
    assert result1["sales_unit"] == "unknown"
    for file, digest in result1["model_sha256"].items():
        assert sha(tmp_path / "out1" / file) == digest
        model = joblib.load(tmp_path / "out1" / file)
        assert model["artifact_kind"] == result1["artifact_kind"]
    assert np.isfinite(result1["reused_test_metrics"]["hgb_absolute"]["mae_pp"])
    with pytest.raises(FileExistsError):
        train(first, first_manifest, tmp_path / "out1", date(2026, 11, 9))


def test_wrong_manifest_rejected_and_no_output_created(tmp_path):
    source, manifest = fixture(tmp_path, "input", 40)
    obj = json.loads(manifest.read_text())
    obj["candidate_sha256"] = "0" * 64
    manifest.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="checksum"):
        train(source, manifest, tmp_path / "out", date(2026, 11, 9))
    assert not (tmp_path / "out").exists()
