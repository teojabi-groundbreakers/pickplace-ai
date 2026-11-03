from datetime import date

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import scripts.retrospective_experiment as experiment
from pickplace_ai.artifacts import ArtifactError, load_bundle
from pickplace_ai.data import write_observations
from pickplace_ai.demo import synthetic_observations
from pickplace_ai.features import FEATURE_COLUMNS
from pickplace_ai.periods import quarter_index
from scripts.retrospective_experiment import make_frame, predict_bundle, run, split_frame


def snapshot_rows():
    asof = date(2026, 11, 3)
    return [
        row.model_copy(update={"available_at": asof}) for row in synthetic_observations(8, 24, 42)
    ]


def test_snapshot_date_is_not_backdated_and_future_features_are_excluded():
    rows = snapshot_rows()
    frame = make_frame(rows, date(2026, 11, 3))
    assert set(frame.snapshot_available_at) == {"2026-11-03"}
    record = frame.iloc[0]
    changed = []
    for row in rows:
        if row.period > record.base_period:
            row = row.model_copy(update={"sales_amount": row.sales_amount * 7})
        changed.append(row)
    changed_frame = make_frame(changed, date(2026, 11, 3))
    changed_record = changed_frame.loc[
        (changed_frame.region_code == record.region_code)
        & (changed_frame.base_period == record.base_period)
    ].iloc[0]
    assert changed_record.sales_amount == record.sales_amount
    assert changed_record.sales_lag1 == record.sales_lag1
    assert changed_record.target != record.target


def test_snapshot_availability_mismatch_rejected():
    rows = snapshot_rows()
    rows[0] = rows[0].model_copy(update={"available_at": date(2024, 1, 1)})
    with pytest.raises(ValueError, match="preserve"):
        make_frame(rows, date(2026, 11, 3))


def test_split_preserves_target_quarters_and_full_embargoes():
    frame = make_frame(snapshot_rows(), date(2026, 11, 3))
    partitions, periods, embargoes = split_frame(frame)
    assert len(embargoes) == 2
    assert quarter_index(periods["validation"][0]) - quarter_index(periods["train"][-1]) == 2
    assert quarter_index(periods["test"][0]) - quarter_index(periods["validation"][-1]) == 2
    assert all(set(part.target_period).isdisjoint(embargoes) for part in partitions.values())


def test_experiment_artifact_is_rejected_by_production_loader(tmp_path):
    path = tmp_path / "input.csv"
    write_observations(snapshot_rows(), path)
    output = tmp_path / "bundle"
    meta, results = run(path, output, date(2026, 11, 3), source_kind="synthetic")
    assert meta["release_status"] == "retrospective_research_only"
    assert meta["snapshot_available_at"] == "2026-11-03"
    assert meta["selection"] == min(
        results["validation"], key=lambda name: results["validation"][name]["mae"]
    )
    with pytest.raises(ArtifactError):
        load_bundle(output)
    with pytest.raises(FileExistsError):
        run(path, output, date(2026, 11, 3), source_kind="synthetic")


def test_recent_growth_selection_saves_a_working_predictor(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    write_observations(snapshot_rows(), input_path)
    original_metrics = experiment._metrics
    calls = iter([10, 0, 20, 30])

    def forced_first_validation(actual, prediction):
        result = original_metrics(actual, prediction)
        try:
            result["mae"] = next(calls)
        except StopIteration:
            pass
        return result

    monkeypatch.setattr(experiment, "_metrics", forced_first_validation)
    output = tmp_path / "recent"
    metadata, _ = run(input_path, output, date(2026, 11, 3), source_kind="synthetic")
    assert metadata["selection"] == "recent_growth"
    bundle = joblib.load(output / "model.joblib")
    sample = pd.DataFrame([{name: 0.1 for name in FEATURE_COLUMNS}])
    assert predict_bundle(bundle, sample)[0] == 0.1
    learned_bundle = joblib.load(output / "learned-model.joblib")
    assert learned_bundle["selection"] == metadata["learned_model_selection"]
    assert len(predict_bundle(learned_bundle, sample)) == 1


def test_predict_bundle_applies_floor_and_rejects_wrong_schema():
    sample = pd.DataFrame([{name: 0.1 for name in FEATURE_COLUMNS}])
    estimator = DummyRegressor(strategy="constant", constant=-2).fit(sample, [0])
    bundle = {"artifact_kind": experiment.ARTIFACT_KIND, "pipeline": estimator}
    assert predict_bundle(bundle, sample)[0] == -1.0
    with pytest.raises(ValueError, match="Feature columns"):
        predict_bundle(bundle, sample.drop(columns=["quarter"]))
