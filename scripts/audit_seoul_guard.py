"""Explore pre-target surge, drop and low-sales flags on known labels.

The post-hoc thresholds are research evidence, not validated error detection.
Only trusted local model files should be loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np

import pickplace_ai.unified_features as shared
from pickplace_ai import features
from scripts import audit_seoul_signals as signal
from scripts import train_gyeonggi_research as base
from scripts import train_unified_growth as unified


def minimum_growth(history: np.ndarray) -> np.ndarray:
    """Calculate the steepest quarter-over-quarter fall from four inputs only."""
    signal.preprediction_signals(history)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        return np.min(history[:, 1:] / history[:, :-1] - 1, axis=1)


def training_thresholds(history: np.ndarray) -> tuple[float, float, float, float]:
    """Fit thresholds only on already-observed four-quarter training inputs."""
    low, rise = signal.thresholds(history)
    fall = float(np.quantile(minimum_growth(history), 0.05))
    latest_low = float(np.quantile(history[:, 3], 0.05))
    return low, rise, fall, latest_low


def guard_flags(
    history: np.ndarray, low: float, rise: float, fall: float, latest_low: float
) -> dict[str, np.ndarray]:
    """Apply strict bounds using only the observed quarters, never a target."""
    if not np.isfinite([low, rise, fall, latest_low]).all():
        raise ValueError("Thresholds must be finite")
    lows, highs = signal.preprediction_signals(history)
    return {
        "low_reference": lows < low,
        "rapid_increase": highs > rise,
        "rapid_decrease": minimum_growth(history) < fall,
        "low_latest": history[:, 3] < latest_low,
    }


def run(source: Path, model_root: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output already exists")
    paths = {
        "seoul": source,
        "metadata": model_root / "metadata.json",
        "evaluation_model": model_root / "evaluation-pooled.joblib",
        "final_model": model_root / "final-model.joblib",
    }
    hashes = {name: base.sha(path) for name, path in paths.items()}
    metadata = json.loads(paths["metadata"].read_text())
    if metadata.get("artifact_kind") != unified.KIND or metadata.get("purpose") != (
        "retrospective_research_only_not_service"
    ):
        raise ValueError("Unexpected research artifact")
    if metadata["source_hashes"]["seoul"] != hashes["seoul"]:
        raise ValueError("Source mismatch")
    for name in ("evaluation_model", "final_model"):
        if metadata["model_sha256"][paths[name].name] != hashes[name]:
            raise ValueError("Model checksum mismatch")
    codes = {
        "trainer": Path(unified.__file__),
        "feature_code": Path(features.__file__),
        "unified_feature_code": Path(shared.__file__),
        "base_trainer": Path(base.__file__),
        "original_signal_study": Path(signal.__file__),
        "this_analysis": Path(__file__),
    }
    code_hashes = {key: base.sha(value) for key, value in codes.items()}
    for key in ("trainer", "feature_code", "unified_feature_code", "base_trainer"):
        if metadata["code_sha256"][key] != code_hashes[key]:
            raise ValueError("Code checksum mismatch")
    # A digest detects accidental changes, not malicious pickle execution.
    bundle = joblib.load(paths["evaluation_model"])
    quarters = unified.seoul_quarters(source)
    _, train_history = signal.frames_and_history(quarters, "2025Q2")
    low, rise, fall, latest_low = training_thresholds(train_history)
    report = {
        "purpose": "retrospective_guard_threshold_research_not_future_validation",
        "training_target_period": "2025Q2",
        "training_rows": len(train_history),
        "thresholds": {
            "low_sales_below_krw": low,
            "max_previous_qoq_increase_above_ratio": rise,
            "min_previous_qoq_decrease_below_ratio": fall,
            "latest_sales_below_krw": latest_low,
            "low_sales_quantile": 0.05,
            "increase_quantile": 0.95,
            "decrease_quantile": 0.05,
            "latest_sales_quantile": 0.05,
            "comparisons": "strict; equality remains eligible",
        },
        "definition": (
            "OR of low target-year-ago quarter sales, max previous-quarter increase, "
            "min previous-quarter decrease, and low latest quarter sales, all "
            "calculated only from q-4..q-1."
        ),
        "source_sha256": hashes,
        "code_sha256": code_hashes,
        "limitations": [
            "Thresholds selected after inspecting earlier results; not preregistered.",
            "Known-target samples only; conditional selection may be biased.",
            "2026Q2 was previously inspected and is not independent test data.",
            "Current revised data does not establish historical publication dates.",
            "Flags are risk signals, not proof of invalid data or fraud.",
            "Performance of the final retrained model was not independently evaluated.",
        ],
        "periods": {},
    }
    for name, period in (("validation", "2025Q4"), ("reused_test", "2026Q2")):
        frame, histories = signal.frames_and_history(quarters, period)
        prediction = unified.predict_unified_model(
            bundle, frame[list(shared.UNIFIED_RESEARCH_FEATURES)]
        )
        flags = guard_flags(histories, low, rise, fall, latest_low)
        small, surge, decline = (
            flags[key] for key in ("low_reference", "rapid_increase", "rapid_decrease")
        )
        old_flags, new_flags = small | surge, small | surge | decline
        latest_sales = flags["low_latest"]
        four_flags = new_flags | latest_sales
        all_rows = np.ones(len(frame), dtype=bool)
        report["periods"][name] = {
            "target_period": period,
            "all": signal._metrics(frame.target.to_numpy(), prediction, all_rows),
            "old_selected": signal._metrics(frame.target.to_numpy(), prediction, ~old_flags),
            "three_signals_selected": signal._metrics(
                frame.target.to_numpy(), prediction, ~new_flags
            ),
            "three_signals_flagged": signal._metrics(
                frame.target.to_numpy(), prediction, new_flags
            ),
            "four_signals_selected": signal._metrics(
                frame.target.to_numpy(), prediction, ~four_flags
            ),
            "four_signals_flagged": signal._metrics(
                frame.target.to_numpy(), prediction, four_flags
            ),
            "decline_only_new_rows": int((decline & ~old_flags).sum()),
            "decline_any_rows": int(decline.sum()),
            "latest_only_new_rows": int((latest_sales & ~new_flags).sum()),
            "latest_any_rows": int(latest_sales.sum()),
        }
    if hashes != {name: base.sha(path) for name, path in paths.items()}:
        raise ValueError("Sources changed during analysis")
    if code_hashes != {name: base.sha(path) for name, path in codes.items()}:
        raise ValueError("Code changed during analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seoul-decline-", dir=output.parent) as temp:
        staged = Path(temp) / "result"
        staged.mkdir()
        summary = staged / "summary.json"
        summary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        (staged / "manifest.json").write_text(
            json.dumps({"summary_sha256": base.sha(summary)}, indent=2) + "\n"
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output already exists")
        os.rename(staged, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seoul", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seoul, args.model_root, args.output)
    print(json.dumps({"thresholds": report["thresholds"], "periods": report["periods"]}))


if __name__ == "__main__":
    main()
