"""Current-snapshot, quarterly Gyeonggi research only; never a live backtest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

import pickplace_ai.features as features_module
from pickplace_ai.artifacts import package_versions
from pickplace_ai.features import GYEONGGI_RESEARCH_FEATURES, gyeonggi_four_quarter_features

KIND = "gyeonggi_quarter_growth_snapshot_research_v1"
PERIODS = {"train": "2025Q2", "validation": "2025Q4", "test": "2026Q2"}
EMBARGO = ("2025Q3", "2026Q1")
SCOPE = ["행정동코드", "중분류업종코드"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_quarters(monthly: pd.DataFrame) -> pd.DataFrame:
    """Only exactly three different valid calendar months create a quarter."""
    cols = [*SCOPE, "기준년월", "매출금액"]
    if not set(cols).issubset(monthly.columns):
        raise ValueError("Missing candidate columns")
    data = monthly[cols].copy()
    if data[SCOPE + ["기준년월"]].isna().any().any():
        raise ValueError("Missing candidate key")
    if data.duplicated(SCOPE + ["기준년월"]).any():
        raise ValueError("Duplicate candidate monthly key")
    if not data["기준년월"].str.fullmatch(r"\d{4}(0[1-9]|1[0-2])").all():
        raise ValueError("Invalid candidate month")
    data["매출금액"] = pd.to_numeric(data["매출금액"], errors="raise")
    if not np.isfinite(data["매출금액"]).all() or data["매출금액"].lt(0).any():
        raise ValueError("Invalid sales amount")
    data["period"] = pd.PeriodIndex(pd.to_datetime(data["기준년월"], format="%Y%m"), freq="Q")
    by_quarter = data.groupby([*SCOPE, "period"], sort=True)["매출금액"].agg(["size", "sum"])
    return by_quarter.loc[by_quarter["size"] == 3, "sum"].unstack("period")


def make_frame(quarters: pd.DataFrame, target_period: str) -> pd.DataFrame:
    target = pd.Period(target_period, freq="Q")
    periods = [target - i for i in (4, 3, 2, 1, 0)]
    if any(p not in quarters.columns for p in periods):
        raise ValueError(f"Missing full calendar quarter for {target_period}")
    data = quarters[periods].dropna()
    keep = (data.iloc[:, :3] > 0).all(axis=1)
    data = data.loc[keep]
    if data.empty:
        raise ValueError(f"No eligible scopes for {target_period}")
    x = data.iloc[:, :4].to_numpy(dtype=float)
    feats = gyeonggi_four_quarter_features(x, (target - 1).quarter)
    scopes = data.index.to_frame(index=False).reset_index(drop=True)
    result = pd.concat([scopes, feats], axis=1)
    result["target"] = data.iloc[:, 4].to_numpy(dtype=float) / x[:, 0] - 1
    result["target_period"] = target_period
    if not np.isfinite(result["target"]).all():
        raise ValueError("Nonfinite target")
    return result


def predict_research_model(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    """Apply the same checked prediction floor in evaluation and later research."""
    if bundle.get("artifact_kind") != KIND:
        raise ValueError("Not a Gyeonggi retrospective research bundle")
    if list(bundle.get("feature_columns", ())) != list(GYEONGGI_RESEARCH_FEATURES):
        raise ValueError("Unexpected bundle feature schema")
    if list(features.columns) != list(GYEONGGI_RESEARCH_FEATURES):
        raise ValueError("Features must use the expected column names and order")
    if features.empty or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Features must be nonempty and finite")
    result = np.asarray(bundle["model"].predict(features), dtype=float)
    if result.ndim != 1 or len(result) != len(features) or not np.isfinite(result).all():
        raise ValueError("Invalid prediction shape or non-finite prediction")
    return np.maximum(result, -1.0)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    error = np.abs(y - pred)
    return {
        "rows": len(y),
        "mae_pp": float(mean_absolute_error(y, pred) * 100),
        "median_absolute_error_pp": float(np.median(error) * 100),
        "p90_absolute_error_pp": float(np.quantile(error, 0.90) * 100),
    }


def train(candidate: Path, review_manifest: Path, output: Path, snapshot: date) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Never overwrite a research artifact")
    manifest = json.loads(review_manifest.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "review_only_not_training_or_prediction":
        raise ValueError("Unexpected candidate review manifest")
    if manifest.get("candidate_sha256") != sha(candidate):
        raise ValueError("Candidate checksum mismatch")
    review_at = datetime.fromisoformat(manifest["generated_at"]).date()
    if snapshot < review_at or snapshot < date(2026, 6, 30):
        raise ValueError("Snapshot cannot precede acquisition or include future quarters")
    input_hashes = {
        "candidate_sha256": sha(candidate),
        "review_manifest_sha256": sha(review_manifest),
    }
    monthly = pd.read_csv(
        candidate, dtype={"기준년월": str, "행정동코드": str, "중분류업종코드": str}
    )
    quarters = make_quarters(monthly)
    frames = {name: make_frame(quarters, period) for name, period in PERIODS.items()}
    train_frame, val, test = (frames[k] for k in ("train", "validation", "test"))
    train_scopes = set(map(tuple, train_frame[SCOPE].to_numpy()))
    for frame in (val, test):
        frame["scope_seen_in_training"] = [
            tuple(x) in train_scopes for x in frame[SCOPE].to_numpy()
        ]
    xcols = list(GYEONGGI_RESEARCH_FEATURES)
    estimators = {
        "zero_growth": DummyRegressor(strategy="constant", constant=0),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=100,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=1,
            early_stopping=False,
            random_state=42,
        ),
    }
    validation = {}
    for name, model in estimators.items():
        model.fit(train_frame[xcols], train_frame.target)
        prediction = predict_research_model(
            {"artifact_kind": KIND, "model": model, "feature_columns": xcols}, val[xcols]
        )
        validation[name] = metrics(val.target.to_numpy(), prediction)
    selected = min(validation, key=lambda k: validation[k]["mae_pp"])
    # Compute each candidate once on the same untouched test data; never reselect.
    test_predictions = {
        name: predict_research_model(
            {"artifact_kind": KIND, "model": model, "feature_columns": xcols}, test[xcols]
        )
        for name, model in estimators.items()
    }
    test_metrics = {
        name: metrics(test.target.to_numpy(), pred) for name, pred in test_predictions.items()
    }
    prediction = test_predictions[selected]
    evaluated = test_metrics[selected]
    supported = test["scope_seen_in_training"].to_numpy()
    supported_metrics = (
        metrics(test.target.to_numpy()[supported], prediction[supported])
        if supported.any()
        else None
    )
    industry = []
    for code in sorted(test["중분류업종코드"].unique()):
        mask = test["중분류업종코드"].to_numpy() == code
        industry.append(
            {"industry_code": code, **metrics(test.target.to_numpy()[mask], prediction[mask])}
        )
    # Fit a distinct final research HGB only after evaluation and selection.
    final_periods = ["2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
    final_frame = pd.concat([make_frame(quarters, q) for q in final_periods], ignore_index=True)
    final_hgb = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1,
        early_stopping=False,
        random_state=42,
    )
    final_hgb.fit(final_frame[xcols], final_frame.target)
    final_scopes = set(map(tuple, final_frame[SCOPE].to_numpy()))
    metadata = {
        "artifact_kind": KIND,
        "purpose": "retrospective_research_only_not_service",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_acquired_at": snapshot.isoformat(),
        "source_hashes": input_hashes,
        "code_sha256": {
            "trainer": sha(Path(__file__)),
            "feature_code": sha(Path(features_module.__file__)),
        },
        "source_candidate_manifest": manifest,
        "source_region_code_system": "gyeonggi-admdong-10-unverified",
        "h10_verified": False,
        "sales_unit": "unknown",
        "monthly_amount_additivity_verified": False,
        "past_publication_dates_verified": False,
        "features": xcols,
        "prediction_method": "scripts.train_gyeonggi_research.predict_research_model",
        "prediction_floor": -1.0,
        "prediction_scope_policy": "Caller must check region and industry against training scopes.",
        "target": "quarter_sales[t+1]/quarter_sales[t-3]-1",
        "quarter_aggregation": "sum three distinct monthly values, assumption unverified",
        "selection_rule": "lowest validation MAE; test excluded from selection",
        "selection": selected,
        "model_training_periods": PERIODS,
        "embargo_periods": EMBARGO,
        "rows": {k: len(v) for k, v in frames.items()},
        "industries": {k: int(v["중분류업종코드"].nunique()) for k, v in frames.items()},
        "validation_metrics": validation,
        "test_metrics_selected": evaluated,
        "test_metrics_all_candidates_diagnostic_not_for_selection": test_metrics,
        "final_hgb_training_target_periods": final_periods,
        "final_hgb_training_rows": len(final_frame),
        "final_hgb_training_scopes": [list(scope) for scope in sorted(final_scopes)],
        "test_metrics_seen_in_training": supported_metrics,
        "training_scopes": [list(scope) for scope in sorted(train_scopes)],
        "test_seen_scopes": int(supported.sum()),
        "test_unseen_scopes": int((~supported).sum()),
        "test_metrics_by_industry": industry,
        "target_tail": {
            "test_max": float(test.target.max()),
            "test_p99": float(test.target.quantile(0.99)),
        },
        "versions": package_versions(),
        "limitations": [
            "One latest corrected snapshot; historical vintages and availability are unknown.",
            "One train, one validation, one test quarter: limited temporal generalization.",
            "Final HGB was refitted on all five target quarters after evaluation; "
            "test metrics do not describe that final model.",
            "2022-2023 rows are unused: full quarters for this experiment start in 2024Q2.",
            "Code equality does not verify geographic boundaries, H10 or source definitions.",
            "Card sales unit, methodology and monthly additive consistency are unverified.",
            "2024Q1 duplicate keys remain excluded; incomplete quarters are never imputed.",
            "Not API compatible, production-approved or real-time accuracy.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gg-research-", dir=output.parent) as temp:
        directory = Path(temp) / "out"
        directory.mkdir()
        for name, obj in (
            ("selected-model.joblib", estimators[selected]),
            ("learned-model.joblib", estimators["hist_gradient_boosting"]),
            ("final-learned-model.joblib", final_hgb),
        ):
            joblib.dump(
                {"artifact_kind": KIND, "model": obj, "feature_columns": xcols}, directory / name
            )
        metadata["model_sha256"] = {
            name: sha(directory / name)
            for name in (
                "selected-model.joblib",
                "learned-model.joblib",
                "final-learned-model.joblib",
            )
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if input_hashes != {
            "candidate_sha256": sha(candidate),
            "review_manifest_sha256": sha(review_manifest),
        }:
            raise ValueError("Input changed during training")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Never overwrite a research artifact")
        os.rename(directory, output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    result = train(args.candidate, args.review_manifest, args.output, args.snapshot)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("selection", "rows", "validation_metrics", "test_metrics_selected")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
