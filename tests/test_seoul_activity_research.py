"""Check the source window and relative activity feature transformation."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pickplace_ai import seoul_activity_features as activity
from scripts import train_seoul_activity_research as research


def test_activity_covariates_use_duration_adjusted_time_bands() -> None:
    bands = np.array([[90, 50, 30, 30, 40, 30]], dtype=float)
    result = activity.activity_covariates(np.array([120]), np.array([270]), np.array([4]), bands)
    assert result.loc[0, "resident_log1p"] == pytest.approx(np.log1p(120))
    assert result.loc[0, "floating_log1p"] == pytest.approx(np.log1p(270))
    assert result.loc[0, "day_overnight_hourly_log_ratio"] == pytest.approx(
        np.log1p(60 / 6) - np.log1p(120 / 9)
    )
    assert result.loc[0, "evening_day_hourly_log_ratio"] == pytest.approx(
        np.log1p(40 / 4) - np.log1p(60 / 6)
    )


def test_activity_covariates_rejects_inconsistent_source_total() -> None:
    with pytest.raises(ValueError, match="reconcile"):
        activity.activity_covariates(
            np.array([120]),
            np.array([400]),
            np.array([4]),
            np.array([[90, 50, 30, 30, 40, 30]]),
        )


def test_target_frame_uses_only_pre_target_covariates() -> None:
    periods = pd.period_range("2021Q1", "2022Q1", freq="Q")
    quarters = pd.DataFrame(
        [[100, 110, 120, 130, 140]],
        index=pd.MultiIndex.from_tuples([("11110515", "CS100001")], names=list(activity.SCOPE)),
        columns=periods,
    )
    values = {name: 1.0 for name in (*activity.POPULATION_FEATURES, *activity.STORE_FEATURES)}
    source = pd.DataFrame(
        [
            {
                "period": "2021Q4",
                "region_code": "11110515",
                "industry_code": "CS100001",
                **values,
            },
            {
                "period": "2022Q1",
                "region_code": "11110515",
                "industry_code": "CS100001",
                **{name: 999.0 for name in values},
            },
        ]
    )
    frame, history = research.make_target_frame(quarters, source, "2022Q1")
    assert frame.loc[0, "base_period"] == "2021Q4"
    assert frame.loc[0, "resident_log1p"] == 1.0
    assert frame.loc[0, "target"] == pytest.approx(140 / 100 - 1)
    np.testing.assert_array_equal(history, [[100, 110, 120, 130]])


def test_industry_encoder_is_fitted_only_on_training_codes() -> None:
    base = {name: 1.0 for name in activity.FEATURE_SETS["sales_only"]}
    training = pd.DataFrame(
        [{**base, "중분류업종코드": "CS100001", "target": 0.1} for _ in range(30)]
    )
    bundle = research.fit(training, "sales_only")
    assert bundle["encoder"].categories_[0].tolist() == ["CS100001"]
    evaluation = training.iloc[:1].copy()
    evaluation["중분류업종코드"] = "CS100002"
    encoded = research.encoded_features(evaluation, "sales_only", bundle["encoder"])
    assert np.isnan(encoded.loc[0, "seoul_industry"])
    assert np.isfinite(research.predict(bundle, evaluation, "sales_only")).all()


@pytest.mark.parametrize("change_during", ["loading", "training", "serialization", None])
def test_changed_training_input_is_never_published(tmp_path, monkeypatch, change_during) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "snapshot.sqlite3").write_bytes(b"fictional snapshot")
    (snapshot / "manifest.json").write_text('{"fixture": true}')
    floating = tmp_path / "floating.csv"
    floating.write_text("fictional floating input")
    original_hash = activity.sha256(floating)
    output = tmp_path / "artifacts" / "model"

    source = pd.DataFrame(
        [
            {
                "period": "2021Q4",
                "region_code": "11110515",
                "resident_total": 120,
                "floating_total": 270,
                "store_count": 4,
                **dict(zip(activity.TIME_COLUMNS, [90, 50, 30, 30, 40, 30], strict=True)),
            }
        ]
    )
    frame = pd.DataFrame(
        [
            {
                **{name: 1.0 for name in activity.FEATURE_SETS["sales_population_stores"]},
                "중분류업종코드": "CS100001",
                "target": 0.1,
            }
        ]
    )
    history = np.array([[100.0, 110.0, 120.0, 130.0]])

    def load_snapshot(*_args):
        if change_during == "loading":
            floating.write_text("changed while loading")
        return pd.DataFrame(), source, {"created_at_utc": "2026-11-11T00:00:00+00:00"}

    def fit(*_args):
        if change_during == "training":
            floating.write_text("changed while training")
        return {
            "model": "fictional model",
            "encoder": SimpleNamespace(categories_=[np.array(["CS100001"])]),
        }

    original_dump = research.joblib.dump

    def dump(*args, **kwargs):
        result = original_dump(*args, **kwargs)
        if change_during == "serialization":
            floating.write_text("changed while serializing")
        return result

    # 무거운 학습은 합성 fixture로 대체하고 입력 변경 감지와 발행 경계만 검사한다.
    monkeypatch.setattr(activity, "load_snapshot", load_snapshot)
    monkeypatch.setattr(research, "make_target_frame", lambda *_args: (frame.copy(), history))
    monkeypatch.setattr(research, "compare_fold", lambda *_args: ({}, {}))
    monkeypatch.setattr(research, "fit", fit)
    monkeypatch.setattr(research.joblib, "dump", dump)

    if change_during is not None:
        with pytest.raises(ValueError, match="Source changed during training"):
            research.train(snapshot, floating, output)
        assert not output.exists()
    else:
        report = research.train(snapshot, floating, output)
        saved = json.loads((output / "metadata.json").read_text())
        assert saved["source_hashes"]["floating_csv"] == original_hash
        assert report["source_hashes"] == saved["source_hashes"]
        assert (output / "final-model.joblib").is_file()
    assert list(output.parent.glob(".seoul-activity-*")) == []
