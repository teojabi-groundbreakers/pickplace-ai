"""Compare Seoul-only industry encoding without changing the existing API.

All comparisons are retrospective research on previously examined current
snapshots. The separate final refit is not evaluated or serving-ready.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from pickplace_ai import features, seoul_guard_policy
from pickplace_ai import seoul_industry_research as candidate
from pickplace_ai import unified_features as shared
from pickplace_ai.artifacts import package_versions
from scripts import audit_seoul_guard as guard
from scripts import audit_seoul_signals as signal
from scripts import train_gyeonggi_research as base
from scripts import train_unified_extended_history as extended
from scripts import train_unified_growth as unified


def metrics(frame: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray) -> dict:
    if not mask.any():
        raise ValueError("No evaluation rows")
    result = unified.metric(frame.target.to_numpy()[mask], prediction[mask])
    result["rows"] = int(mask.sum())
    result["coverage_pct"] = float(mask.mean() * 100)
    return result


def evaluate(
    models: dict, sf: pd.DataFrame, gf: pd.DataFrame, history: np.ndarray, cutoffs: tuple
) -> dict:
    flags = guard.guard_flags(history, *cutoffs)
    passed = ~np.logical_or.reduce(list(flags.values()))
    result = {}
    for name, (model, predict) in models.items():
        seoul_prediction = predict(model, sf)
        gg_prediction = predict(model, gf)
        result[name] = {
            "seoul_guard_pass": metrics(sf, seoul_prediction, passed),
            "seoul_all": metrics(sf, seoul_prediction, np.ones(len(sf), dtype=bool)),
            "gyeonggi_all": metrics(gf, gg_prediction, np.ones(len(gf), dtype=bool)),
            "seoul_withheld_rows": int((~passed).sum()),
            "seoul_withheld_pct": float((~passed).mean() * 100),
        }
    return result


def baseline_predict(model, frame: pd.DataFrame) -> np.ndarray:
    return unified.predict_unified_model(
        unified.bundle(model), frame[list(candidate.UNIFIED_RESEARCH_FEATURES)]
    )


def train(
    seoul: Path,
    seoul_manifest: Path,
    gyeonggi: Path,
    gyeonggi_manifest: Path,
    output: Path,
    report_path: Path,
    snapshot: date,
) -> dict:
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError("Never overwrite existing output")
    inputs = {
        "seoul": seoul,
        "seoul_manifest": seoul_manifest,
        "gyeonggi": gyeonggi,
        "gyeonggi_manifest": gyeonggi_manifest,
    }
    hashes = {name: base.sha(path) for name, path in inputs.items()}
    sm = json.loads(seoul_manifest.read_text())
    gm = json.loads(gyeonggi_manifest.read_text())
    if sm.get("output_sha256") != hashes["seoul"]:
        raise ValueError("Seoul checksum mismatch")
    if (
        gm.get("candidate_sha256") != hashes["gyeonggi"]
        or gm.get("purpose") != "review_only_not_training_or_prediction"
    ):
        raise ValueError("Gyeonggi checksum mismatch")
    api_sources = sm["api_sources"]
    if not api_sources:
        raise ValueError("Missing API acquisition provenance")
    seoul_known = max(
        date.fromisoformat(sm["historical_available_at"]),
        *(unified.acquisition_day(x["acquired_at_utc"]) for x in api_sources),
    )
    gyeonggi_known = unified.acquisition_day(gm["generated_at"])
    if (
        snapshot < max(seoul_known, gyeonggi_known)
        or snapshot > datetime.now(ZoneInfo("Asia/Seoul")).date()
    ):
        raise ValueError("Invalid snapshot date")
    sq = unified.seoul_quarters(seoul)
    monthly = pd.read_csv(
        gyeonggi, dtype={"기준년월": str, "행정동코드": str, "중분류업종코드": str}
    )
    gq = base.make_quarters(monthly)
    seoul_periods = extended.target_periods("2022Q2", "2026Q2")
    seoul_frames = {p: unified.make_frame(sq, p, "seoul") for p in seoul_periods}
    gg_frames = {p: unified.make_frame(gq, p, "gyeonggi") for p in extended.FINAL_GYEONGGI_TARGETS}
    eligible = extended.before_embargo(seoul_periods, "2025Q4")
    if eligible[-1] != "2025Q2":
        raise ValueError("Unexpected training cutoff")
    long_seoul = pd.concat([seoul_frames[p] for p in eligible], ignore_index=True)
    gg_train = gg_frames["2025Q2"]
    eval_model = candidate.fit_model(pd.concat([long_seoul, gg_train], ignore_index=True))
    eval_models = {
        "baseline_joint": (extended.fit_joint(seoul_frames["2025Q2"], gg_train), baseline_predict),
        "extended_joint": (extended.fit_joint(long_seoul, gg_train), baseline_predict),
        "seoul_industry_joint": (eval_model, candidate.predict_model),
    }
    _, training_history = signal.frames_and_history(sq, "2025Q2")
    cutoffs = guard.training_thresholds(training_history)
    fixed = (
        seoul_guard_policy.LOW_REFERENCE_SALES_KRW,
        seoul_guard_policy.MAX_QOQ_INCREASE,
        seoul_guard_policy.MIN_QOQ_DECREASE,
        seoul_guard_policy.LOW_LATEST_SALES_KRW,
    )
    if cutoffs != fixed:
        raise ValueError("Training cutoffs do not match current guard policy")
    comparisons = {}
    for period in ("2025Q4", "2026Q2"):
        frame, history = signal.frames_and_history(sq, period)
        comparisons[period] = evaluate(eval_models, frame, gg_frames[period], history, cutoffs)
    final_s = pd.concat([seoul_frames[p] for p in seoul_periods], ignore_index=True)
    final_g = pd.concat([gg_frames[p] for p in extended.FINAL_GYEONGGI_TARGETS], ignore_index=True)
    final = candidate.fit_model(pd.concat([final_s, final_g], ignore_index=True))
    report = {
        "artifact_kind": "seoul_industry_unified_research_v1",
        "purpose": "retrospective_research_only_not_service",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_review_date": snapshot.isoformat(),
        "source_known_by": {
            "seoul": seoul_known.isoformat(),
            "gyeonggi": gyeonggi_known.isoformat(),
        },
        "source_sha256": hashes,
        "gyeonggi_source_eligibility": {
            "manifest_purpose": gm["purpose"],
            "review_only": True,
            "approved_for_training_or_prediction": False,
            "explanation": (
                "Unverified candidate used for requested retrospective simulation; "
                "not approved training data."
            ),
        },
        "code_sha256": {
            "trainer": base.sha(Path(__file__)),
            "model_features": base.sha(Path(candidate.__file__)),
            "unified_features": base.sha(Path(shared.__file__)),
            "features": base.sha(Path(features.__file__)),
            "gyeonggi_trainer": base.sha(Path(base.__file__)),
            "unified_trainer": base.sha(Path(unified.__file__)),
            "extended_trainer": base.sha(Path(extended.__file__)),
            "signal_audit": base.sha(Path(signal.__file__)),
            "guard_audit": base.sha(Path(guard.__file__)),
            "guard_policy": base.sha(Path(seoul_guard_policy.__file__)),
        },
        "features": list(candidate.FEATURES),
        "training_periods": {"seoul": eligible, "gyeonggi": ["2025Q2"]},
        "evaluation_guard_thresholds": list(cutoffs),
        "comparisons_previously_inspected": comparisons,
        "final_training_periods": {
            "seoul": seoul_periods,
            "gyeonggi": list(extended.FINAL_GYEONGGI_TARGETS),
        },
        "final_training_rows": {"seoul": len(final_s), "gyeonggi": len(final_g)},
        "versions": package_versions(),
        "limitations": [
            "2025Q4 and 2026Q2 were previously inspected; neither is an independent test.",
            "2026Q2 is reused diagnosis, not model selection or independent testing.",
            "Current-snapshot retrospective simulation; historical dates are not known.",
            "Seoul-only industry uses original source codes; Gyeonggi category is missing.",
            "Existing guard thresholds are exploratory and do not authorize serving this model.",
            "Final refit includes 2026Q2 and has no independent performance estimate.",
            "Gyeonggi candidates remain unverified and unapproved for training or prediction.",
            "Gyeonggi units and boundaries need review.",
            "Research simulation is not definitive performance evidence; do not serve or deploy.",
            "Not connected to the existing API or approved for production.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seoul-industry-", dir=output.parent) as tmp:
        dest = Path(tmp) / "output"
        dest.mkdir()
        joblib.dump(eval_model, dest / "evaluation-model.joblib")
        joblib.dump(final, dest / "final-model.joblib")
        report["model_sha256"] = {p.name: base.sha(p) for p in dest.iterdir()}
        (dest / "metadata.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if hashes != {name: base.sha(path) for name, path in inputs.items()}:
            raise ValueError("Inputs changed during training")
        if (
            output.exists()
            or output.is_symlink()
            or report_path.exists()
            or report_path.is_symlink()
        ):
            raise FileExistsError("Output already exists")
        os.rename(dest, output)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seoul", "seoul-manifest", "gyeonggi", "gyeonggi-manifest", "output", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--snapshot", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    result = train(
        args.seoul,
        args.seoul_manifest,
        args.gyeonggi,
        args.gyeonggi_manifest,
        args.output,
        args.report,
        args.snapshot,
    )
    print(json.dumps(result["comparisons_previously_inspected"], ensure_ascii=False))


if __name__ == "__main__":
    main()
