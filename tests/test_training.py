"""Tests for temporal leakage, artifact integrity, and honest provenance."""

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from pickplace_ai.artifacts import ArtifactError, ModelBundle, load_bundle
from pickplace_ai.features import FEATURE_COLUMNS, build_training_frame
from pickplace_ai.periods import quarter_end, shift_quarter
from pickplace_ai.schemas import Observation
from pickplace_ai.training import observation_sha256, temporal_split, train


def observations() -> list[Observation]:
    result = []
    for region in range(4):
        for index in range(24):
            period = shift_quarter("2019Q1", index)
            result.append(
                Observation(
                    region_code=f"999999{region:04d}",
                    region_code_system="synthetic-test",
                    industry_code="coffee",
                    period=period,
                    available_at=quarter_end(period) + timedelta(days=30),
                    source_kind="synthetic",
                    source_dataset="generated test fixture, not official statistics",
                    sales_amount=float((1_000_000 + region * 50_000) * (1.02**index)),
                    store_count=20 + region,
                    resident_population=10_000 + region * 100,
                    working_population=5_000 + region * 50,
                    floating_population=20_000 + region * 200,
                )
            )
    return result


def test_quarters_and_publication_cutoffs_do_not_leak():
    frame = build_training_frame(observations())
    split = temporal_split(frame)
    train_periods = set(split["train"]["target_period"])
    validation_periods = set(split["validation"]["target_period"])
    test_periods = set(split["test"]["target_period"])
    assert train_periods.isdisjoint(validation_periods | test_periods)
    assert validation_periods.isdisjoint(test_periods)
    assert max(split["train"]["label_available_at"]) <= min(split["validation"]["as_of"])
    assert max(split["validation"]["label_available_at"]) <= min(split["test"]["as_of"])


def test_delayed_training_label_is_excluded():
    frame = build_training_frame(observations())
    initial = temporal_split(frame)
    delayed_index = initial["train"].index[-1]
    frame.loc[delayed_index, "label_available_at"] = max(initial["test"]["as_of"]) + timedelta(
        days=100
    )
    split = temporal_split(frame)
    assert delayed_index not in split["train"].index


def test_no_validation_labels_available_before_test_fails():
    frame = build_training_frame(observations())
    initial = temporal_split(frame)
    indices = initial["validation"].index
    frame.loc[indices, "label_available_at"] = max(initial["test"]["as_of"]) + timedelta(days=100)
    with pytest.raises(ValueError, match="empty"):
        temporal_split(frame)


def test_late_feature_snapshot_is_excluded_even_if_label_is_old():
    frame = build_training_frame(observations())
    initial = temporal_split(frame)
    delayed_index = initial["train"].index[-1]
    frame.loc[delayed_index, "as_of"] = max(initial["test"]["as_of"]) + timedelta(days=100)
    split = temporal_split(frame)
    assert delayed_index not in split["train"].index


def test_training_exports_experimental_bundle_and_finite_prediction(tmp_path):
    rows = observations()
    output = tmp_path / "model"
    metadata = train(rows, output)
    bundle = load_bundle(output)
    assert metadata["source_kind"] == "synthetic"
    assert metadata["release_status"] == "experimental"
    assert len(metadata["dataset_sha256"]) == 64
    assert (
        metadata["training_cutoff"]["latest_label_available_at"]
        <= metadata["training_cutoff"]["first_validation_origin"]
    )
    assert metadata["selected_model"] == min(
        metadata["validation_metrics"],
        key=lambda name: metadata["validation_metrics"][name]["mae"],
    )
    frame = build_training_frame(rows)
    features = {name: float(frame.iloc[-1][name]) for name in FEATURE_COLUMNS}
    assert np.isfinite(bundle.predict(features))
    assert bundle.predict(features) >= -1
    report = json.loads((output / "evaluation.json").read_text())
    assert report["test_by_scope"]
    assert report["dataset_sha256"] == metadata["dataset_sha256"]
    assert "experimental" in (output / "MODEL_CARD.md").read_text()
    assert metadata["dataset_sha256"] in (output / "MODEL_CARD.md").read_text()
    with pytest.raises(ValueError, match="overwrite"):
        train(rows, output)


def test_model_checksum_detects_modified_artifact(tmp_path):
    output = tmp_path / "model"
    train(observations(), output)
    with (output / "model.joblib").open("ab") as stream:
        stream.write(b"unexpected modification")
    with pytest.raises(ArtifactError, match="checksum"):
        load_bundle(output)


def test_version_and_feature_schema_are_checked_before_loading(tmp_path):
    output = tmp_path / "model"
    train(observations(), output)
    path = output / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["feature_columns"] = list(reversed(metadata["feature_columns"]))
    path.write_text(json.dumps(metadata))
    with pytest.raises(ArtifactError, match="columns"):
        load_bundle(output)


def test_provenance_cannot_be_mixed(tmp_path):
    rows = observations()
    rows[0] = rows[0].model_copy(update={"source_kind": "official"})
    with pytest.raises(ValueError, match="mix"):
        train(rows, tmp_path / "model")


def test_dataset_hash_preserves_identity_without_depending_on_row_order():
    rows = observations()
    assert observation_sha256(rows) == observation_sha256(list(reversed(rows)))
    modified = [*rows]
    modified[0] = rows[0].model_copy(update={"sales_amount": rows[0].sales_amount + 1})
    assert observation_sha256(rows) != observation_sha256(modified)


def test_reference_sensitivity_has_growth_units_and_ignores_missing_inputs():
    class KnownPrediction:
        def predict(self, X):
            return X["sales_amount"].to_numpy() * 0.1

    features = dict.fromkeys(FEATURE_COLUMNS, float("nan"))
    features["sales_amount"] = 3.0
    references = dict.fromkeys(FEATURE_COLUMNS, 1.0)
    bundle = ModelBundle(KnownPrediction(), {"reference_features": references})
    explanation = bundle.explain(features)
    assert len(explanation) == 1
    assert explanation[0]["feature"] == "sales_amount"
    assert explanation[0]["delta_growth"] == pytest.approx(0.2)
    assert explanation[0]["direction"] == "positive"
    assert explanation[0]["method"] == "single_feature_reference_sensitivity"
    bundle.metadata["reference_features"]["sales_amount"] = None
    assert bundle.explain(features) == []


def test_test_targets_do_not_choose_estimator(monkeypatch, tmp_path):
    from pickplace_ai import training

    frame = build_training_frame(observations())
    expected = temporal_split(frame)
    test_indices = expected["test"].index
    altered = frame.copy()
    altered.loc[test_indices, "target"] = 100.0
    monkeypatch.setattr(training, "build_training_frame", lambda rows: frame)
    first = train(observations(), tmp_path / "first")
    monkeypatch.setattr(training, "build_training_frame", lambda rows: altered)
    second = train(observations(), tmp_path / "second")
    assert first["selected_model"] == second["selected_model"]
    assert first["validation_metrics"] == second["validation_metrics"]
    assert first["test_metrics"] != second["test_metrics"]


def test_short_history_cannot_create_spurious_split():
    with pytest.raises(ValueError):
        temporal_split(pd.DataFrame())
