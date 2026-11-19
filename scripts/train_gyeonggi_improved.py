"""Retrospective comparison of robust and squared-loss Gyeonggi models."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

import pickplace_ai.features as features_module
from pickplace_ai.artifacts import package_versions
from pickplace_ai.features import GYEONGGI_RESEARCH_FEATURES
from scripts import train_gyeonggi_research as base

PERIODS = {"train": "2025Q2", "validation": "2025Q4", "reused_test": "2026Q2"}
FINAL_PERIODS = ("2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2")


def models() -> dict:
    common = dict(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1,
        early_stopping=False,
        random_state=42,
    )
    return {
        "zero_growth": DummyRegressor(strategy="constant", constant=0),
        "hgb_squared": HistGradientBoostingRegressor(loss="squared_error", **common),
        "hgb_absolute": HistGradientBoostingRegressor(loss="absolute_error", **common),
    }


def bundle(model):
    return {
        "artifact_kind": base.KIND,
        "model": model,
        "feature_columns": list(GYEONGGI_RESEARCH_FEATURES),
    }


def train(candidate: Path, review_manifest: Path, output: Path, snapshot: date) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Never overwrite a research artifact")
    input_hashes = {
        "candidate_sha256": base.sha(candidate),
        "review_manifest_sha256": base.sha(review_manifest),
    }
    manifest = json.loads(review_manifest.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "review_only_not_training_or_prediction":
        raise ValueError("Unexpected candidate review manifest")
    if manifest.get("candidate_sha256") != input_hashes["candidate_sha256"]:
        raise ValueError("Candidate checksum mismatch")
    review_at = datetime.fromisoformat(manifest["generated_at"]).date()
    if snapshot < review_at or snapshot < date(2026, 6, 30):
        raise ValueError("Snapshot cannot precede acquisition or include future quarters")
    monthly = pd.read_csv(
        candidate, dtype={"기준년월": str, "행정동코드": str, "중분류업종코드": str}
    )
    quarters = base.make_quarters(monthly)
    frames = {name: base.make_frame(quarters, period) for name, period in PERIODS.items()}
    fit, val, test = (frames[k] for k in ("train", "validation", "reused_test"))
    xcols = list(GYEONGGI_RESEARCH_FEATURES)
    scopes = set(map(tuple, fit[base.SCOPE].to_numpy()))
    test_seen = np.array([tuple(row) in scopes for row in test[base.SCOPE].to_numpy()])
    estimators = models()
    val_metrics = {}
    for name, model in estimators.items():
        model.fit(fit[xcols], fit.target)
        predictions = base.predict_research_model(bundle(model), val[xcols])
        val_metrics[name] = base.metrics(val.target.to_numpy(), predictions)
    selected = min(val_metrics, key=lambda name: val_metrics[name]["mae_pp"])
    # All hypotheses above are fixed before touching the reused historical test.
    # This test was previously inspected and is never an independent holdout.
    test_metrics = {}
    for name, model in estimators.items():
        predictions = base.predict_research_model(bundle(model), test[xcols])
        test_metrics[name] = base.metrics(test.target.to_numpy(), predictions)
    final = pd.concat([base.make_frame(quarters, p) for p in FINAL_PERIODS], ignore_index=True)
    final_selected = clone(estimators[selected]).fit(final[xcols], final.target)
    final_robust = clone(estimators["hgb_absolute"]).fit(final[xcols], final.target)
    final_scopes = set(map(tuple, final[base.SCOPE].to_numpy()))
    eval_dir = {
        "evaluation-zero.joblib": estimators["zero_growth"],
        "evaluation-squared.joblib": estimators["hgb_squared"],
        "evaluation-absolute.joblib": estimators["hgb_absolute"],
        "final-selected.joblib": final_selected,
        "final-absolute.joblib": final_robust,
    }
    metadata = {
        "artifact_kind": base.KIND,
        "experiment_kind": "gyeonggi_robust_loss_research_v1",
        "purpose": "retrospective_research_only_not_service",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_acquired_at": snapshot.isoformat(),
        "source_hashes": input_hashes,
        "code_sha256": {
            "improved_trainer": base.sha(Path(__file__)),
            "base_trainer": base.sha(Path(base.__file__)),
            "feature_code": base.sha(Path(features_module.__file__)),
        },
        "source_region_code_system": "gyeonggi-admdong-10-unverified",
        "h10_verified": False,
        "sales_unit": "unknown",
        "monthly_amount_additivity_verified": False,
        "past_publication_dates_verified": False,
        "model_training_periods": PERIODS,
        "embargo_periods": ["2025Q3", "2026Q1"],
        "test_reused": True,
        "validation_metrics": val_metrics,
        "reused_test_metrics": test_metrics,
        "selection_rule": "lowest validation MAE, reused test not used in selection",
        "selection": selected,
        "rows": {k: len(v) for k, v in frames.items()},
        "industries": {k: int(v["중분류업종코드"].nunique()) for k, v in frames.items()},
        "test_seen_scopes": int(test_seen.sum()),
        "test_unseen_scopes": int((~test_seen).sum()),
        "training_scopes": [list(s) for s in sorted(scopes)],
        "final_training_scopes": [list(s) for s in sorted(final_scopes)],
        "final_training_target_periods": FINAL_PERIODS,
        "final_training_rows": len(final),
        "features": xcols,
        "target": "quarter_sales[t+1]/quarter_sales[t-3]-1",
        "prediction_method": "scripts.train_gyeonggi_research.predict_research_model",
        "prediction_floor": -1.0,
        "prediction_scope_policy": "Caller must check region and industry against training scopes.",
        "quarter_aggregation": "sum three distinct monthly values, assumption unverified",
        "versions": package_versions(),
        "limitations": [
            "Validation quarter used to compare multiple model variants; it is exploratory.",
            "2026Q2 test was previously inspected, so reused test is not independent.",
            "Historical snapshots and publication dates are unknown; not a live backtest.",
            "One train, one validation, one reused test quarter; weak temporal evidence.",
            "Final models refit using all five target quarters including reused test labels; "
            "reported historical metrics do not measure final model performance.",
            "2022-2023 not used; 2024Q1 excluded, missing months never imputed.",
            "Source code system H10, geography, sales unit, monthly sum unverified.",
            "Not production-approved or API compatible; not a success probability.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gyeonggi-robust-", dir=output.parent) as temp:
        path = Path(temp) / "out"
        path.mkdir()
        for filename, model in eval_dir.items():
            joblib.dump(bundle(model), path / filename)
        metadata["model_sha256"] = {name: base.sha(path / name) for name in eval_dir}
        (path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if input_hashes != {
            "candidate_sha256": base.sha(candidate),
            "review_manifest_sha256": base.sha(review_manifest),
        }:
            raise ValueError("Input changed during training")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Never overwrite a research artifact")
        os.rename(path, output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    result = train(args.candidate, args.review_manifest, args.output, args.snapshot)
    keys = ("selection", "rows", "validation_metrics", "reused_test_metrics")
    print(json.dumps({k: result[k] for k in keys}, ensure_ascii=False))


if __name__ == "__main__":
    main()
