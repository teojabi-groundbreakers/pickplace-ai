"""Contract checks for the isolated Seoul industry experiment."""

import joblib
import numpy as np
import pandas as pd
import pytest

from pickplace_ai import seoul_industry_research as research
from scripts.train_seoul_industry_research import evaluate


def sample_frame():
    size = 120
    frame = pd.DataFrame(
        {key: np.arange(size, dtype=float) / size for key in research.UNIFIED_RESEARCH_FEATURES}
    )
    frame["source"] = ["seoul"] * 60 + ["gyeonggi"] * 60
    frame["source_gyeonggi"] = [0.0] * 60 + [1.0] * 60
    frame[research.INDUSTRY] = ["A", "B"] * 30 + ["A", "G"] * 30
    frame["target"] = np.sin(np.arange(size) / 12)
    return frame


def test_encoder_is_train_only_and_gyeonggi_is_missing():
    frame = sample_frame()
    bundle = research.fit_model(frame)
    future = frame.iloc[[0, 1, 60, 61]].copy()
    future.loc[future.index[0], research.INDUSTRY] = "never-seen"
    transformed = research.encode_features(future, bundle["encoder"])
    assert np.isnan(transformed.iloc[0]["seoul_industry"])
    assert np.isfinite(transformed.iloc[1]["seoul_industry"])
    assert transformed.iloc[2:]["seoul_industry"].isna().all()
    assert "never-seen" not in bundle["encoder"].categories_[0]
    assert np.isfinite(research.predict_model(bundle, future)).all()


def test_saved_bundle_preserves_predictions_and_rejects_wrong_kind(tmp_path):
    frame = sample_frame()
    bundle = research.fit_model(frame)
    path = tmp_path / "trusted-local.joblib"
    joblib.dump(bundle, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(
        research.predict_model(bundle, frame), research.predict_model(restored, frame)
    )
    with pytest.raises(ValueError, match="model kind"):
        research.predict_model({"kind": "other"}, frame)


def test_metrics_compare_identical_guard_mask_and_all_rows():
    seoul = sample_frame().iloc[:4].copy()
    gg = sample_frame().iloc[60:64].copy()
    history = np.array(
        [[10, 10, 10, 10], [1, 1, 1, 1], [10, 10, 10, 10], [10, 10, 10, 10]], dtype=float
    )
    models = {
        "zero": (None, lambda _, f: np.zeros(len(f))),
        "one": (None, lambda _, f: np.ones(len(f))),
    }
    results = evaluate(models, seoul, gg, history, (5, 2, -0.7, 5))
    for result in results.values():
        assert result["seoul_guard_pass"]["rows"] == 3
        assert result["seoul_guard_pass"]["coverage_pct"] == 75
        assert result["seoul_all"]["rows"] == 4
        assert result["gyeonggi_all"]["rows"] == 4
        assert result["seoul_withheld_rows"] == 1


@pytest.mark.parametrize(
    "mode",
    [
        "nan",
        "infinite",
        "invalid",
        "empty",
        "source",
        "source_indicator",
        "missing_industry",
        "blank_industry",
    ],
)
def test_bad_input_fails_closed(mode):
    frame = sample_frame()
    bundle = research.fit_model(frame)
    test = frame.copy()
    if mode == "nan":
        test.loc[0, research.UNIFIED_RESEARCH_FEATURES[0]] = np.nan
    elif mode == "infinite":
        test.loc[0, research.UNIFIED_RESEARCH_FEATURES[0]] = np.inf
    elif mode == "invalid":
        test[research.UNIFIED_RESEARCH_FEATURES[0]] = "not-a-number"
    elif mode == "empty":
        test = test.iloc[:0]
    elif mode == "source":
        test.loc[0, "source"] = "other"
    elif mode == "source_indicator":
        test.loc[0, "source_gyeonggi"] = 1.0
    elif mode == "missing_industry":
        test.loc[0, research.INDUSTRY] = None
    elif mode == "blank_industry":
        test.loc[0, research.INDUSTRY] = "  "
    with pytest.raises(ValueError):
        research.predict_model(bundle, test)


def test_training_rejects_too_many_industries():
    frame = pd.concat([sample_frame()] * 6, ignore_index=True)
    mask = frame.source == "seoul"
    frame.loc[mask, research.INDUSTRY] = [f"x{i}" for i in range(mask.sum())]
    with pytest.raises(ValueError, match="limit"):
        research.fit_model(frame)


def test_predictions_are_finite():
    frame = sample_frame()
    bundle = research.fit_model(frame)

    class BadEstimator:
        def predict(self, values):
            return np.full(len(values), np.nan)

    bundle["model"] = BadEstimator()
    with pytest.raises(ValueError, match="Invalid model prediction"):
        research.predict_model(bundle, frame)
