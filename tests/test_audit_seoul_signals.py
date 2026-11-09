"""Check pre-prediction-only signals on fictional observations."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import pickplace_ai.features as features_module
import pickplace_ai.unified_features as unified_features_module
from pickplace_ai.unified_features import UNIFIED_RESEARCH_FEATURES
from scripts import audit_seoul_signals as audit
from scripts import train_gyeonggi_research as base
from scripts import train_unified_growth as unified


def test_target_change_never_changes_history_or_flags():
    periods = pd.period_range("2024Q2", "2025Q2", freq="Q")
    keys = pd.MultiIndex.from_tuples([("11110515", "CS100010")], names=base.SCOPE)
    data = pd.DataFrame([[10, 20, 20, 30, 40]], index=keys, columns=periods)
    first, history = audit.frames_and_history(data, "2025Q2")
    before = audit.flag_groups(history, 11, 0.9)
    data.loc[:, pd.Period("2025Q2", freq="Q")] = 400_000
    second, changed_history = audit.frames_and_history(data, "2025Q2")
    after = audit.flag_groups(changed_history, 11, 0.9)
    assert first.target.iloc[0] != second.target.iloc[0]
    np.testing.assert_array_equal(history, changed_history)
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    # Evaluation requires an observed target, though the isolated signals do not.
    np.testing.assert_array_equal(audit.preprediction_signals(history)[0], [10])
    with pytest.raises(ValueError, match="Missing full calendar quarter"):
        audit.frames_and_history(data.drop(columns=[pd.Period("2025Q2", freq="Q")]), "2025Q2")


def test_flag_boundaries_and_disjoint_overlap():
    history = np.array(
        [
            [10, 10, 10, 10],
            [9, 9, 9, 9],
            [10, 10, 10, 15],
            [10, 20, 20, 20],
            [9, 18, 18, 18],
        ],
        dtype=float,
    )
    group = audit.flag_groups(history, low_threshold=10, surge_threshold=0.5)
    assert group["selected_neither_flag"].tolist() == [True, False, True, False, False]
    assert group["low_only"].tolist() == [False, True, False, False, False]
    assert group["surge_only"].tolist() == [False, False, False, True, False]
    assert group["both"].tolist() == [False, False, False, False, True]
    assert group["flagged_either"].tolist() == [False, True, False, True, True]
    assert np.array_equal(
        group["flagged_either"], group["low_only"] | group["surge_only"] | group["both"]
    )


def test_rejects_invalid_denominators_and_thresholds():
    with pytest.raises(ValueError, match="zero"):
        audit.preprediction_signals(np.array([[0, 1, 2, 3]]))
    with pytest.raises(ValueError, match="finite"):
        audit.preprediction_signals(np.array([[1, 2, np.nan, 3]]))
    with pytest.raises(ValueError, match="threshold"):
        audit.flag_groups(np.array([[1, 2, 3, 4]]), np.nan, 0.1)


def test_analyze_output_keeps_inputs_and_rejects_overwrite(tmp_path):
    records = []
    periods = pd.period_range("2024Q2", "2026Q2", freq="Q")
    for index, sales in enumerate(([10, 20, 30, 40, 50, 40, 30, 20, 10], [10] * 9)):
        for period, amount in zip(periods, sales, strict=True):
            records.append(
                {
                    "region_code": f"1111051{index}",
                    "region_code_system": "seoul-adstrd",
                    "industry_code": "CS100010",
                    "period": str(period),
                    "sales_unit": "KRW",
                    "sales_amount": amount,
                }
            )
    source = tmp_path / "source.csv"
    pd.DataFrame(records).to_csv(source, index=False)
    root = tmp_path / "trusted_model"
    root.mkdir()
    predictor = DummyRegressor(strategy="constant", constant=0)
    predictor.fit(pd.DataFrame([[0] * 6], columns=UNIFIED_RESEARCH_FEATURES), [0])
    model = root / "evaluation-pooled.joblib"
    joblib.dump(unified.bundle(predictor), model)
    metadata = {
        "artifact_kind": unified.KIND,
        "purpose": "retrospective_research_only_not_service",
        "source_hashes": {"seoul": base.sha(source)},
        "model_sha256": {model.name: base.sha(model)},
        "code_sha256": {
            "trainer": base.sha(Path(unified.__file__)),
            "feature_code": base.sha(Path(features_module.__file__)),
            "unified_feature_code": base.sha(Path(unified_features_module.__file__)),
            "base_trainer": base.sha(Path(base.__file__)),
        },
    }
    manifest = root / "metadata.json"
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    before = {path: base.sha(path) for path in (source, model, manifest)}
    output = tmp_path / "report"
    result = audit.analyze(seoul=source, model_root=root, output=output)
    assert result["purpose"] == "preprediction_flag_diagnostic_not_a_live_policy"
    assert result["partitions"]["validation"]["all"]["rows"] == 2
    assert result["partitions"]["reused_test"]["all"]["rows"] == 2
    assert (
        json.loads((output / "summary.json").read_text())["source_hashes"]["seoul"]
        == (before[source])
    )
    report_manifest = json.loads((output / "manifest.json").read_text())
    assert report_manifest["summary_sha256"] == base.sha(output / "summary.json")
    assert report_manifest["code_sha256"]["analysis"] == base.sha(Path(audit.__file__))
    assert {path: base.sha(path) for path in before} == before
    with pytest.raises(FileExistsError):
        audit.analyze(seoul=source, model_root=root, output=output)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        audit.analyze(seoul=source, model_root=root, output=tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
