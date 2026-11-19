"""Explore pre-prediction warnings; no service policy or input exclusions change."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pickplace_ai.features as features_module
import pickplace_ai.unified_features as unified_features_module
from pickplace_ai.unified_features import UNIFIED_RESEARCH_FEATURES
from scripts import train_gyeonggi_research as base
from scripts import train_unified_growth as unified

LOW_QUANTILE = 0.05
SURGE_QUANTILE = 0.95
TRAIN_TARGET = "2025Q2"
PERIODS = {"validation": "2025Q4", "reused_test": "2026Q2"}


def preprediction_signals(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use only four input quarters; first three must be valid denominators."""
    x = np.asarray(history, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError("Expected finite four-quarter histories")
    if (x < 0).any() or (x[:, :3] <= 0).any():
        raise ValueError("Invalid sales or division by zero")
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        maximum_growth = np.max(x[:, 1:] / x[:, :-1] - 1, axis=1)
    return x[:, 0].copy(), maximum_growth


def thresholds(history: np.ndarray) -> tuple[float, float]:
    if len(history) == 0:
        raise ValueError("Empty training history")
    sales, growth = preprediction_signals(history)
    return float(np.quantile(sales, LOW_QUANTILE)), float(np.quantile(growth, SURGE_QUANTILE))


def flag_groups(
    history: np.ndarray, low_threshold: float, surge_threshold: float
) -> dict[str, np.ndarray]:
    if not np.isfinite([low_threshold, surge_threshold]).all():
        raise ValueError("Invalid flag thresholds")
    sales, growth = preprediction_signals(history)
    low = sales < low_threshold
    surge = growth > surge_threshold
    return {
        "all": np.ones(len(sales), dtype=bool),
        "selected_neither_flag": ~(low | surge),
        "flagged_either": low | surge,
        "low_only": low & ~surge,
        "surge_only": ~low & surge,
        "both": low & surge,
    }


def frames_and_history(quarters: pd.DataFrame, period: str) -> tuple[pd.DataFrame, np.ndarray]:
    frame = unified.make_frame(quarters, period, "seoul")
    keys = pd.MultiIndex.from_frame(frame[base.SCOPE])
    target = pd.Period(period, freq="Q")
    history = quarters.loc[keys, [target - n for n in (4, 3, 2, 1)]].to_numpy(dtype=float)
    preprediction_signals(history)
    return frame, history


def _metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    actual, values = target[mask], prediction[mask]
    error = np.abs(actual - values)
    return {
        "rows": int(mask.sum()),
        "coverage_pct": float(mask.mean() * 100),
        "model_mae_pp": float(error.mean() * 100) if len(error) else None,
        "model_median_pp": float(np.median(error) * 100) if len(error) else None,
        "model_p90_pp": float(np.quantile(error, 0.9) * 100) if len(error) else None,
        "zero_growth_mae_pp": float(np.abs(actual).mean() * 100) if len(error) else None,
    }


def analyze(*, seoul: Path, model_root: Path, output: Path) -> dict:
    """Write a diagnostic report without mutating data, model, or existing output."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new output directory")
    metadata_path = model_root / "metadata.json"
    model_path = model_root / "evaluation-pooled.joblib"
    paths = {"seoul": seoul, "metadata": metadata_path, "model": model_path}
    hashes = {name: base.sha(path) for name, path in paths.items()}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_kind") != unified.KIND or metadata.get("purpose") != (
        "retrospective_research_only_not_service"
    ):
        raise ValueError("Not a compatible research artifact")
    if metadata["source_hashes"]["seoul"] != hashes["seoul"]:
        raise ValueError("Seoul input digest mismatch")
    if metadata["model_sha256"][model_path.name] != hashes["model"]:
        raise ValueError("Evaluation model digest mismatch")
    code_paths = {
        "trainer": Path(unified.__file__),
        "feature_code": Path(features_module.__file__),
        "unified_feature_code": Path(unified_features_module.__file__),
        "base_trainer": Path(base.__file__),
        "analysis": Path(__file__),
    }
    code_hashes = {name: base.sha(path) for name, path in code_paths.items()}
    for name in ("trainer", "feature_code", "unified_feature_code", "base_trainer"):
        if code_hashes[name] != metadata["code_sha256"].get(name):
            raise ValueError("Research artifact code mismatch")
    # The artifact must be trusted locally; a hash alone does not make pickle safe.
    model = joblib.load(model_path)
    quarters = unified.seoul_quarters(seoul)
    _, train_history = frames_and_history(quarters, TRAIN_TARGET)
    low_threshold, surge_threshold = thresholds(train_history)
    result = {
        "purpose": "preprediction_flag_diagnostic_not_a_live_policy",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_hashes": hashes,
        "code_sha256": code_hashes,
        "thresholds": {
            "training_target_period": TRAIN_TARGET,
            "training_rows": len(train_history),
            "reference_sales_quantile": LOW_QUANTILE,
            "reference_sales_below_krw": low_threshold,
            "maximum_historical_qoq_growth_quantile": SURGE_QUANTILE,
            "maximum_historical_qoq_growth_above_ratio": surge_threshold,
        },
        "definition": {
            "low": "q-4 sales below threshold (fixed from training inputs)",
            "surge": "maximum of three qoq growths in q-4..q-1 above threshold",
            "policy": "low OR surge; target q and actual error are not used by flags",
        },
        "limitations": [
            "Threshold choice was explored after inspecting error distribution; not preregistered.",
            "Validation and reused test are revised historical snapshots; not live tests.",
            "Evaluated scopes require known target labels and four full input quarters; "
            "coverage is conditional and may be selection-biased.",
            "This evaluation script cannot accept an unlabeled future target. "
            "Real-time flagging needs a separate input-only path and prospective validation.",
            "2026Q2 was already inspected; results are reused diagnostics.",
            "Selected-subset metrics are conditional and cannot represent whole-model improvement.",
            "No automatic withholding, source deletion, label exclusion, or API policy change.",
            "Flags are hypotheses, not proof of errors or causes of sales changes.",
        ],
        "partitions": {},
    }
    for name, period in PERIODS.items():
        frame, history = frames_and_history(quarters, period)
        prediction = unified.predict_unified_model(model, frame[list(UNIFIED_RESEARCH_FEATURES)])
        target = frame.target.to_numpy(dtype=float)
        groups = flag_groups(history, low_threshold, surge_threshold)
        result["partitions"][name] = {
            "target_period": period,
            **{label: _metrics(target, prediction, mask) for label, mask in groups.items()},
        }
    if hashes != {name: base.sha(path) for name, path in paths.items()}:
        raise ValueError("Input changed while processing")
    if code_hashes != {name: base.sha(path) for name, path in code_paths.items()}:
        raise ValueError("Code changed while processing")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seoul-signal-audit-", dir=output.parent) as temp:
        staged = Path(temp) / "result"
        staged.mkdir()
        summary_path = staged / "summary.json"
        summary_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        report_manifest = {
            "purpose": result["purpose"],
            "source_hashes": hashes,
            "code_sha256": code_hashes,
            "summary_sha256": base.sha(summary_path),
        }
        (staged / "manifest.json").write_text(
            json.dumps(report_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError("Use a new output directory")
        os.rename(staged, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seoul", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(seoul=args.seoul, model_root=args.model_root, output=args.output)
    print(json.dumps({"purpose": result["purpose"], "thresholds": result["thresholds"]}))


if __name__ == "__main__":
    main()
