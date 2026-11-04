"""Year-ahead Seoul snapshot experiment. This is not the production model.

All origins use information through Q4 of the preceding year. Snapshot dates
are preserved: historical scoring is a retrospective diagnostic, not a
point-in-time backtest. The common feature_record implementation is used both
for model fitting and for the 2026 forecasts.
"""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pickplace_ai.data import validate_observations
from pickplace_ai.features import FEATURE_COLUMNS, FeatureUnavailable, feature_record
from pickplace_ai.periods import quarter_end, shift_quarter

FEATURES = FEATURE_COLUMNS + ["comparison_sales", "target_quarter", "industry_code"]
LOG_FIELDS = {
    "sales_amount",
    "sales_lag1",
    "sales_lag4",
    "sales_mean4",
    "comparison_sales",
}
NUMERIC = [c for c in FEATURES if c != "industry_code"]
ARTIFACT_KIND = "seoul_year_ahead_snapshot_forecast_v1"


def year_rows(observations, base_year, snapshot_at, *, labels):
    """Build four target quarters per scope using no observations after base Q4.

    When labels=True the following year's sales are accessed solely to create
    target labels; they are never passed to feature_record or feature inputs.
    """
    validate_observations(observations)
    groups = defaultdict(dict)
    for row in observations:
        if row.available_at != snapshot_at:
            raise ValueError("All rows must preserve the snapshot availability date")
        if quarter_end(row.period) > snapshot_at:
            raise ValueError("Snapshot contains an incomplete observation quarter")
        if (
            row.region_code_system != "seoul-adstrd"
            or len(row.region_code) != 8
            or not row.region_code.startswith("11")
        ):
            raise ValueError("Expected Seoul administrative-dong observations")
        groups[(row.region_code_system, row.region_code, row.industry_code)][row.period] = row
    base_period = f"{base_year}Q4"
    records, omitted = [], Counter()
    for (system, region, industry), periods in sorted(groups.items()):
        if not any(f"{base_year}Q{q}" in periods for q in range(1, 5)):
            continue
        history = [
            periods[p]
            for p in (shift_quarter(base_period, x) for x in range(-4, 1))
            if p in periods
        ]
        try:
            features = feature_record(history, base_period, snapshot_at)
        except FeatureUnavailable as exc:
            omitted[exc.reason] += 4
            continue
        for q in range(1, 5):
            comparison = periods.get(f"{base_year}Q{q}")
            target = periods.get(f"{base_year + 1}Q{q}")
            if (
                comparison is None
                or comparison.sales_amount is None
                or comparison.sales_amount <= 0
            ):
                omitted["missing_or_nonpositive_comparison_sales"] += 1
                continue
            if labels and (target is None or target.sales_amount is None):
                omitted["missing_target_sales"] += 1
                continue
            row = {
                **features,
                "comparison_sales": comparison.sales_amount,
                "target_quarter": q,
                "industry_code": industry,
                "region_code": region,
                "region_code_system": system,
                "base_period": base_period,
                "comparison_period": comparison.period,
                "target_period": f"{base_year + 1}Q{q}",
            }
            if labels:
                row["target"] = target.sales_amount / comparison.sales_amount - 1
                if not np.isfinite(row["target"]):
                    raise ValueError("Non-finite target growth")
            records.append(row)
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No eligible records for base year {base_year}")
    return frame, dict(omitted)


def model_matrix(frame):
    """Apply the same input transform during fitting and prediction."""
    matrix = frame[FEATURES].copy()
    for field in LOG_FIELDS:
        if (matrix[field].dropna() < 0).any():
            raise ValueError(f"Negative input in {field}")
        matrix[field] = np.log1p(matrix[field])
    for field in ("sales_yoy", "sales_qoq"):
        matrix[field] = np.sign(matrix[field]) * np.log1p(np.abs(matrix[field]))
    if np.isinf(matrix[NUMERIC].to_numpy(dtype=float)).any():
        raise ValueError("Infinite input")
    return matrix


def candidates():
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            ("industry", OneHotEncoder(handle_unknown="ignore"), ["industry_code"]),
        ]
    )
    return {
        "zero_growth": DummyRegressor(strategy="constant", constant=0),
        "ridge": Pipeline([("features", clone(preprocess)), ("model", Ridge(alpha=10))]),
        "huber_sgd": Pipeline(
            [
                ("features", clone(preprocess)),
                (
                    "model",
                    SGDRegressor(
                        loss="huber",
                        epsilon=1.0,
                        alpha=0.01,
                        max_iter=1000,
                        tol=1e-4,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def predictions(estimator, frame):
    result = np.asarray(estimator.predict(model_matrix(frame)), dtype=float)
    if result.ndim != 1 or len(result) != len(frame) or not np.isfinite(result).all():
        raise ValueError("Non-finite or invalid model predictions")
    # Negative sales are impossible; the target is a ratio so -100% is the floor.
    return np.maximum(result, -1.0)


def metrics(frame, estimate):
    y = frame.target.to_numpy(float)
    error = np.abs(y - estimate)
    return {
        "rows": len(frame),
        "mae_ratio": float(mean_absolute_error(y, estimate)),
        "mae_pp": float(mean_absolute_error(y, estimate) * 100),
        "median_absolute_error_ratio": float(median_absolute_error(y, estimate)),
        "median_absolute_error_pp": float(median_absolute_error(y, estimate) * 100),
        "comparison_sales_below_1m_krw": int((frame.comparison_sales < 1_000_000).sum()),
        "absolute_actual_growth_over_100pct": int((np.abs(y) > 1).sum()),
        "absolute_error_over_100pct": int((error > 1).sum()),
    }


def fit_and_evaluate(train, validation, test):
    """Select solely on 2024, train through 2024, then score untouched 2025."""
    validation_metrics, validation_predictions = {}, {}
    for name, model in candidates().items():
        model.fit(model_matrix(train), train.target)
        validation_predictions[name] = predictions(model, validation)
        validation_metrics[name] = metrics(validation, validation_predictions[name])
    winner = min(validation_metrics, key=lambda name: validation_metrics[name]["mae_ratio"])
    test_metrics, test_predictions = {}, {}
    for name, model in candidates().items():
        combined = pd.concat([train, validation], ignore_index=True)
        model.fit(model_matrix(combined), combined.target)
        test_predictions[name] = predictions(model, test)
        test_metrics[name] = metrics(test, test_predictions[name])
    return (
        winner,
        validation_metrics,
        validation_predictions,
        test_metrics,
        test_predictions,
    )
