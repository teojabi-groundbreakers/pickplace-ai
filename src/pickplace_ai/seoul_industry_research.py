"""Isolated exploratory pooled model with Seoul-only industry categories.

The encoder is fitted on training rows only and stored with the estimator. This
is not compatible with or loaded by the serving API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from pickplace_ai.unified_features import UNIFIED_RESEARCH_FEATURES

INDUSTRY = "중분류업종코드"
FEATURES = (*UNIFIED_RESEARCH_FEATURES, "seoul_industry")


def encode_features(frame: pd.DataFrame, encoder: OrdinalEncoder) -> pd.DataFrame:
    """Apply a fitted encoder; unknown Seoul and all Gyeonggi codes become NaN."""
    if frame.empty:
        raise ValueError("Features must be nonempty")
    if not set(frame.source).issubset({"seoul", "gyeonggi"}):
        raise ValueError("Unsupported source")
    result = frame[list(UNIFIED_RESEARCH_FEATURES)].copy()
    try:
        valid = np.isfinite(result.to_numpy(dtype=float)).all()
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be numeric and finite") from exc
    if not valid:
        raise ValueError("Features must be numeric and finite")
    if not np.array_equal(
        result["source_gyeonggi"].to_numpy(), (frame.source == "gyeonggi").to_numpy(dtype=float)
    ):
        raise ValueError("Source indicator mismatch")
    result["seoul_industry"] = np.nan
    mask = frame.source == "seoul"
    if mask.any():
        codes = frame.loc[mask, INDUSTRY]
        if codes.isna().any() or (codes.astype(str).str.strip() == "").any():
            raise ValueError("Seoul industry codes are required")
        result.loc[mask, "seoul_industry"] = encoder.transform(codes.to_frame("industry")).ravel()
    return result[list(FEATURES)]


def fit_model(frame: pd.DataFrame) -> dict:
    """Fit both encoder and estimator only on the supplied training rows."""
    if set(frame.source) != {"seoul", "gyeonggi"}:
        raise ValueError("Both sources are required")
    seoul = frame.source == "seoul"
    if (
        frame.loc[seoul, INDUSTRY].isna().any()
        or (frame.loc[seoul, INDUSTRY].astype(str).str.strip() == "").any()
    ):
        raise ValueError("Seoul industry codes are required")
    if frame.loc[seoul, INDUSTRY].nunique() > 255:
        raise ValueError("Seoul industry categories exceed estimator limit")
    if not np.isfinite(frame.target.to_numpy(dtype=float)).all():
        raise ValueError("Targets must be finite")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan).fit(
        frame.loc[seoul, INDUSTRY].to_frame("industry")
    )
    counts = frame.source.value_counts().to_dict()
    weights = np.array([1.0 / counts[source] for source in frame.source])
    weights *= len(weights) / weights.sum()
    estimator = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1,
        early_stopping=False,
        random_state=42,
        loss="absolute_error",
        categorical_features=[False] * len(UNIFIED_RESEARCH_FEATURES) + [True],
    ).fit(encode_features(frame, encoder), frame.target, sample_weight=weights)
    return {"kind": "seoul_industry_unified_research_v1", "encoder": encoder, "model": estimator}


def predict_model(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict research growth ratios with the exact training-time transformation."""
    if bundle.get("kind") != "seoul_industry_unified_research_v1":
        raise ValueError("Unexpected research model kind")
    prediction = np.asarray(
        bundle["model"].predict(encode_features(frame, bundle["encoder"])), dtype=float
    )
    if prediction.ndim != 1 or len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise ValueError("Invalid model prediction")
    return np.maximum(prediction, -1)
