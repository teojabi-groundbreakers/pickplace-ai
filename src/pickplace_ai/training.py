"""Publication-aware quarter splits, validation-only selection, and artifact export."""

import hashlib
import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_training_frame
from pickplace_ai.periods import quarter_index
from pickplace_ai.schemas import Observation

SCOPE_COLUMNS = ["region_code", "region_code_system", "industry_code"]


def observation_sha256(observations: list[Observation]) -> str:
    """Fingerprint canonical inputs independently of CSV row or JSON key order."""
    rows = sorted(
        (row.model_dump(mode="json") for row in observations),
        key=lambda row: (
            row["region_code_system"],
            row["region_code"],
            row["industry_code"],
            quarter_index(row["period"]),
        ),
    )
    canonical = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class RecentGrowthRegressor(RegressorMixin, BaseEstimator):
    """Persist the latest observed year-over-year growth; no target fitting."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RecentGrowthRegressor":
        self.n_features_in_ = len(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(X["sales_yoy"].to_numpy(dtype=float), nan=0.0)


def temporal_split(
    frame: pd.DataFrame,
    validation_quarters: int = 2,
    test_quarters: int = 2,
) -> dict[str, pd.DataFrame]:
    """Keep whole target quarters together and embargo unavailable labels.

    The model is selected before the first test prediction. Validation labels must
    already be known then, and training labels must be known at first validation.
    """
    if validation_quarters < 1 or test_quarters < 1:
        raise ValueError("validation_quarters and test_quarters must both be positive")
    if frame.empty:
        raise ValueError("No eligible training rows; check sales history and availability dates")
    required = {
        *FEATURE_COLUMNS,
        *SCOPE_COLUMNS,
        "target",
        "source_kind",
        "base_period",
        "target_period",
        "as_of",
        "label_available_at",
    }
    if required - set(frame.columns):
        raise ValueError("Training frame is missing required schema columns")
    working = frame.copy()
    working["as_of"] = pd.to_datetime(working["as_of"]).dt.date
    working["label_available_at"] = pd.to_datetime(working["label_available_at"]).dt.date
    quarters = sorted(working["target_period"].unique(), key=quarter_index)
    if len(quarters) < validation_quarters + test_quarters + 2:
        raise ValueError("Need at least two training quarters plus validation and test quarters")
    test_periods = quarters[-test_quarters:]
    validation_periods = quarters[-test_quarters - validation_quarters : -test_quarters]
    train_periods = quarters[: -test_quarters - validation_quarters]
    train_frame = working.loc[working["target_period"].isin(train_periods)].copy()
    validation_frame = working.loc[working["target_period"].isin(validation_periods)].copy()
    test_frame = working.loc[working["target_period"].isin(test_periods)].copy()
    validation_origin = min(validation_frame["as_of"])
    test_origin = min(test_frame["as_of"])
    train_frame = train_frame.loc[
        (train_frame["label_available_at"] <= validation_origin)
        & (train_frame["as_of"] <= validation_origin)
    ].copy()
    validation_frame = validation_frame.loc[
        (validation_frame["label_available_at"] <= test_origin)
        & (validation_frame["as_of"] <= test_origin)
    ].copy()
    if train_frame.empty or validation_frame.empty or test_frame.empty:
        raise ValueError("Publication-aware split leaves an empty train/validation/test partition")
    for partition in (train_frame, validation_frame, test_frame):
        if not np.isfinite(partition["target"].to_numpy(dtype=float)).all():
            raise ValueError("Training labels must be finite")
        if np.isinf(partition[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).any():
            raise ValueError("Training features cannot contain infinite values")
    return {"train": train_frame, "validation": validation_frame, "test": test_frame}


def _predict(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    predictions = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
    if len(predictions) != len(X) or not np.isfinite(predictions).all():
        raise ValueError("Candidate returned invalid predictions")
    return np.maximum(predictions, -1.0)


def _metrics(actual: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(actual, predictions))
    rmse = float(np.sqrt(mean_squared_error(actual, predictions)))
    return {"mae": mae, "rmse": rmse, "mae_pp": mae * 100, "rmse_pp": rmse * 100}


def _scope_metrics(frame: pd.DataFrame, predictions: np.ndarray) -> list[dict[str, Any]]:
    scored = frame.copy()
    scored["prediction"] = predictions
    result = []
    for keys, rows in scored.groupby(SCOPE_COLUMNS, sort=True):
        result.append(
            {
                **dict(zip(SCOPE_COLUMNS, keys, strict=True)),
                "n": len(rows),
                **_metrics(rows["target"].to_numpy(), rows["prediction"].to_numpy()),
            }
        )
    return result


def _partition_description(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "target_periods": sorted(frame["target_period"].unique(), key=quarter_index),
        "earliest_as_of": str(min(frame["as_of"])),
        "latest_as_of": str(max(frame["as_of"])),
        "latest_label_available_at": str(max(frame["label_available_at"])),
    }


def _learners(model_type: str, seed: int) -> dict[str, Any]:
    def imputer() -> SimpleImputer:
        return SimpleImputer(strategy="median", keep_empty_features=True)

    candidates: dict[str, Any] = {
        "zero_growth": Pipeline(
            [("impute", imputer()), ("model", DummyRegressor(strategy="constant", constant=0))]
        ),
        "recent_growth": RecentGrowthRegressor(),
        "ridge": Pipeline(
            [("impute", imputer()), ("scale", StandardScaler()), ("model", Ridge(alpha=10))]
        ),
    }
    if model_type == "hist":
        candidates["hist"] = Pipeline(
            [
                ("impute", imputer()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=100,
                        max_leaf_nodes=15,
                        learning_rate=0.05,
                        l2_regularization=1,
                        early_stopping=False,
                        random_state=seed,
                    ),
                ),
            ]
        )
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except (ImportError, OSError) as exc:
            raise ValueError(
                "LightGBM runtime unavailable. Install the lightgbm extra and its OpenMP "
                "runtime, or explicitly choose --model hist."
            ) from exc
        candidates["lightgbm"] = Pipeline(
            [
                ("impute", imputer()),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=100,
                        num_leaves=15,
                        learning_rate=0.05,
                        reg_lambda=1,
                        random_state=seed,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    else:
        raise ValueError("model_type must be 'hist' or 'lightgbm'")
    return candidates


def train(
    observations: list[Observation],
    output_dir: Path,
    model_type: str = "hist",
    validation_quarters: int = 2,
    test_quarters: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit on TRAIN only, select on validation only, and export experimental artifacts."""
    if not observations:
        raise ValueError("No observations supplied")
    source_kinds = {row.source_kind for row in observations}
    if len(source_kinds) != 1:
        raise ValueError("Cannot mix official and synthetic data in one training run")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "Output directory must be absent or empty; do not overwrite a model version"
        )
    frame = build_training_frame(observations)
    partitions = temporal_split(frame, validation_quarters, test_quarters)
    training = partitions["train"]
    validation = partitions["validation"]
    testing = partitions["test"]
    X_train = training[list(FEATURE_COLUMNS)]
    y_train = training["target"].to_numpy()
    X_validation = validation[list(FEATURE_COLUMNS)]
    y_validation = validation["target"].to_numpy()
    estimators = _learners(model_type, seed)
    validation_results: dict[str, Any] = {}
    for name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        validation_results[name] = _metrics(y_validation, _predict(estimator, X_validation))
    # Test observations and test targets play no part in selection.
    selected = min(validation_results, key=lambda name: validation_results[name]["mae"])
    test_results = {}
    selected_test_predictions = None
    for name, estimator in estimators.items():
        predictions = _predict(estimator, testing[list(FEATURE_COLUMNS)])
        test_results[name] = _metrics(testing["target"].to_numpy(), predictions)
        if name == selected:
            selected_test_predictions = predictions
    assert selected_test_predictions is not None
    training_scopes = set(training[SCOPE_COLUMNS].itertuples(index=False, name=None))
    testing_scopes = set(testing[SCOPE_COLUMNS].itertuples(index=False, name=None))
    supported = sorted(training_scopes & testing_scopes)
    if not supported:
        raise ValueError("No scope has both training and test observations")
    now = datetime.now(UTC)
    source_kind = source_kinds.pop()
    model_version = f"growth-{source_kind}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    versions = package_versions()
    if selected == "lightgbm":
        versions["lightgbm"] = importlib.metadata.version("lightgbm")
    best_baseline = min(
        validation_results[name]["mae"] for name in ("zero_growth", "recent_growth")
    )
    medians = X_train.median()
    metadata: dict[str, Any] = {
        "model_version": model_version,
        "trained_at": now.isoformat(),
        "source_kind": source_kind,
        "release_status": "experimental",
        "selected_model": selected,
        "requested_model": model_type,
        "selection_rule": "lowest validation MAE, including baselines; no test selection",
        "improves_validation_baseline": validation_results[selected]["mae"] < best_baseline,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "reference_features": {
            name: float(medians[name]) if np.isfinite(medians[name]) else None
            for name in FEATURE_COLUMNS
        },
        "explanation_method": "single_feature_reference_sensitivity",
        "explanation_limitations": [
            "Changes one input feature to its TRAIN median and measures a prediction difference.",
            "Sensitivity deltas are neither SHAP values nor additive or causal contributions.",
            "Changing one derived feature can break consistency with correlated features.",
            "Missing observations and all-missing training reference features are excluded.",
        ],
        "target": "sales[t+1] / sales[t-3] - 1",
        "prediction_floor": -1.0,
        "versions": versions,
        "supported_scopes": [dict(zip(SCOPE_COLUMNS, scope, strict=True)) for scope in supported],
        "split": {name: _partition_description(rows) for name, rows in partitions.items()},
        "training_rows": len(training),
        "eligible_rows": len(frame),
        "input_observations": len(observations),
        "dataset_sha256": observation_sha256(observations),
        "dataset_hash_schema": "canonical-observation-v1",
        "input_period_range": {
            "first": min((row.period for row in observations), key=quarter_index),
            "last": max((row.period for row in observations), key=quarter_index),
        },
        "source_datasets": sorted({row.source_dataset for row in observations}),
        "training_cutoff": {
            "latest_snapshot_available_at": str(max(training["as_of"])),
            "latest_label_available_at": str(max(training["label_available_at"])),
            "first_validation_origin": str(min(validation["as_of"])),
            "first_test_origin": str(min(testing["as_of"])),
        },
        "excluded_after_temporal_cutoffs": len(frame)
        - sum(len(partition) for partition in partitions.values()),
        "seed": seed,
        "validation_metrics": validation_results,
        "validation_mae_improvement_pp": (best_baseline - validation_results[selected]["mae"])
        * 100,
        "test_metrics": test_results[selected],
        "trained_through_target_period": max(training["target_period"], key=quarter_index),
        "limitations": [
            "Synthetic metrics do not measure real-world commercial forecasting performance."
            if source_kind == "synthetic"
            else "Historical release vintages and source comparability require independent review.",
            "Aggregate-area growth is not individual store revenue or startup success probability.",
            "No uncertainty percentage or calibrated confidence is supplied.",
            "Only listed code-system/region/industry scopes are eligible for inference.",
            "This bundle is experimental and is not automatically promoted for production.",
        ],
    }
    evaluation = {
        "model_version": model_version,
        "source_kind": source_kind,
        "dataset_sha256": metadata["dataset_sha256"],
        "dataset_hash_schema": metadata["dataset_hash_schema"],
        "selection": selected,
        "split": metadata["split"],
        "training_cutoff": metadata["training_cutoff"],
        "validation": validation_results,
        "test": test_results,
        "test_by_scope": _scope_metrics(testing, selected_test_predictions),
        "metric_units": {
            "mae": "growth ratio",
            "rmse": "growth ratio",
            "*_pp": "percentage points",
        },
        "notes": metadata["limitations"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = output_dir / "model.joblib.tmp"
    joblib.dump({"pipeline": estimators[selected]}, temporary_model)
    temporary_model.replace(output_dir / "model.joblib")
    metadata["model_sha256"] = file_sha256(output_dir / "model.joblib")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    card = (
        f"# PickPlace Growth Model Card\n\n"
        f"- Version: `{model_version}`\n"
        f"- Source: **{source_kind}**\n"
        f"- Status: **experimental**\n"
        f"- Canonical input SHA-256: `{metadata['dataset_sha256']}`\n"
        f"- Input hash schema: `{metadata['dataset_hash_schema']}`\n"
        f"- Input observations: {len(observations)}\n"
        f"- Eligible labeled samples: {len(frame)}\n"
        f"- Excluded by temporal cutoffs: {metadata['excluded_after_temporal_cutoffs']}\n"
        f"- Selected estimator: `{selected}`\n"
        f"- Target: next-quarter year-over-year aggregate estimated-sales growth\n"
        f"- Selection: validation MAE only; test never selects the estimator\n"
        f"- Train-only preprocessing; publication-aware whole-quarter splits\n"
        f"- Explanation: single-feature sensitivity to TRAIN medians, not SHAP\n"
        f"- Output unit: growth ratio (0.05 = 5%); minimum -1\n"
        f"- Test MAE: {metadata['test_metrics']['mae_pp']:.4f} percentage points\n\n"
        "## Evaluation periods\n\n"
        "| Partition | Samples | Target quarters |\n"
        "|---|---:|---|\n"
        + "\n".join(
            f"| {name} | {values['rows']} | {', '.join(values['target_periods'])} |"
            for name, values in metadata["split"].items()
        )
        + "\n\n## Training cutoff\n\n"
        + "\n".join(f"- {name}: `{value}`" for name, value in metadata["training_cutoff"].items())
        + "\n\n## Limitations\n\n"
        + "\n".join(f"- {note}" for note in metadata["limitations"])
        + "\n\n## Explanation limitations\n\n"
        + "\n".join(f"- {note}" for note in metadata["explanation_limitations"])
        + "\n\nReview `evaluation.json`, data provenance, supported scopes, and publication dates "
        "before any production release. Do not relabel synthetic artifacts as official.\n"
    )
    (output_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    return metadata
