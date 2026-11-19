"""Train a joint exploratory model using Seoul rows passing the fixed input guard.

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

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
        return {
            "rows": 0,
            "mae_pp": None,
            "median_absolute_error_pp": None,
            "p90_absolute_error_pp": None,
            "coverage_pct": 0.0,
        }
    result = unified.metric(frame.target.to_numpy()[mask], prediction[mask])
    result["rows"] = int(mask.sum())
    result["coverage_pct"] = float(mask.mean() * 100)
    return result


def select_eligible_seoul(frame: pd.DataFrame, history: np.ndarray, cutoffs: tuple) -> np.ndarray:
    """Choose by pre-target input history only, never by target outcome."""
    if len(frame) != len(history):
        raise ValueError("History alignment mismatch")
    flags = guard.guard_flags(history, *cutoffs)
    return ~np.logical_or.reduce(list(flags.values()))


def evaluate(
    models: dict, sf: pd.DataFrame, gf: pd.DataFrame, history: np.ndarray, cutoffs: tuple
) -> dict:
    passed = select_eligible_seoul(sf, history, cutoffs)
    result = {}
    for name, (model, predict) in models.items():
        seoul_prediction = predict(model, sf)
        gg_prediction = predict(model, gf)
        result[name] = {
            "seoul_guard_pass": metrics(sf, seoul_prediction, passed),
            "seoul_all": metrics(sf, seoul_prediction, np.ones(len(sf), dtype=bool)),
            "seoul_guard_withheld": metrics(sf, seoul_prediction, ~passed),
            "gyeonggi_all": metrics(gf, gg_prediction, np.ones(len(gf), dtype=bool)),
            "seoul_withheld_rows": int((~passed).sum()),
            "seoul_withheld_pct": float((~passed).mean() * 100),
        }
    return result


def baseline_predict(model, frame: pd.DataFrame) -> np.ndarray:
    return unified.predict_unified_model(
        unified.bundle(model), frame[list(candidate.UNIFIED_RESEARCH_FEATURES)]
    )


def fit_seoul_only_industry(frame: pd.DataFrame) -> dict:
    """Fit Seoul-only fold without borrowing any Gyeonggi or later categories."""
    if frame.empty or not frame.source.eq("seoul").all():
        raise ValueError("Expected nonempty Seoul-only training rows")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan).fit(
        frame[candidate.INDUSTRY].to_frame("industry")
    )
    if len(encoder.categories_[0]) > 255:
        raise ValueError("Too many Seoul industries")
    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1,
        early_stopping=False,
        random_state=42,
        categorical_features=[False] * len(candidate.UNIFIED_RESEARCH_FEATURES) + [True],
    ).fit(candidate.encode_features(frame, encoder), frame.target)
    return {"kind": "seoul_industry_unified_research_v1", "encoder": encoder, "model": estimator}


def rolling_seoul_only(
    sq: pd.DataFrame, periods: list[str], frames: dict[str, pd.DataFrame]
) -> list[dict]:
    """Retrospective fold thresholds use only the latest prior training inputs."""
    results = []
    for target in ("2024Q2", "2024Q4", "2025Q2"):
        prior = extended.before_embargo(periods, target)
        _, reference_history = signal.frames_and_history(sq, prior[-1])
        cutoffs = guard.training_thresholds(reference_history)
        all_training = pd.concat([frames[p] for p in prior], ignore_index=True)
        selected = []
        for period in prior:
            train_frame, history = signal.frames_and_history(sq, period)
            selected.append(train_frame.loc[select_eligible_seoul(train_frame, history, cutoffs)])
        filtered = pd.concat(selected, ignore_index=True)
        f, h = signal.frames_and_history(sq, target)
        pass_mask = select_eligible_seoul(f, h, cutoffs)
        all_industry = fit_seoul_only_industry(all_training)
        filtered_industry = fit_seoul_only_industry(filtered)
        base_all = unified.model().fit(
            all_training[list(candidate.UNIFIED_RESEARCH_FEATURES)], all_training.target
        )
        base_filtered = unified.model().fit(
            filtered[list(candidate.UNIFIED_RESEARCH_FEATURES)], filtered.target
        )
        models = {
            "unfiltered": (base_all, baseline_predict),
            "filtered": (base_filtered, baseline_predict),
            "unfiltered_industry": (all_industry, candidate.predict_model),
            "filtered_industry": (filtered_industry, candidate.predict_model),
        }
        metrics_by_model = {}
        for name, (model, predict) in models.items():
            prediction = predict(model, f)
            metrics_by_model[name] = {
                "passed": metrics(f, prediction, pass_mask),
                "withheld": metrics(f, prediction, ~pass_mask),
                "all": metrics(f, prediction, np.ones(len(f), dtype=bool)),
            }
        results.append(
            {
                "target": target,
                "threshold_training_target": prior[-1],
                "thresholds": list(cutoffs),
                "training_periods": prior,
                "embargo": str(pd.Period(target, freq="Q") - 1),
                "training_rows_all": len(all_training),
                "training_rows_passed": len(filtered),
                "models": metrics_by_model,
            }
        )
    return results


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
    code_files = {
        "trainer": Path(__file__),
        "model_features": Path(candidate.__file__),
        "unified_features": Path(shared.__file__),
        "features": Path(features.__file__),
        "gyeonggi_trainer": Path(base.__file__),
        "unified_trainer": Path(unified.__file__),
        "extended_trainer": Path(extended.__file__),
        "signal_audit": Path(signal.__file__),
        "guard_audit": Path(guard.__file__),
        "guard_policy": Path(seoul_guard_policy.__file__),
    }
    code_hashes = {name: base.sha(path) for name, path in code_files.items()}
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
    filtered_frames = {}
    filtering = {}
    for period in seoul_periods:
        frame, history = signal.frames_and_history(sq, period)
        mask = select_eligible_seoul(frame, history, cutoffs)
        filtered_frames[period] = frame.loc[mask].copy()
        filtering[period] = {
            "total": len(frame),
            "passed": int(mask.sum()),
            "withheld": int((~mask).sum()),
        }
    filtered_s = pd.concat([filtered_frames[p] for p in eligible], ignore_index=True)
    filtered_no_industry = extended.fit_joint(filtered_s, gg_train)
    filtered_industry = candidate.fit_model(pd.concat([filtered_s, gg_train], ignore_index=True))
    eval_models = {
        "baseline_joint": (extended.fit_joint(seoul_frames["2025Q2"], gg_train), baseline_predict),
        "extended_joint": (extended.fit_joint(long_seoul, gg_train), baseline_predict),
        "unfiltered_seoul_industry_joint": (
            candidate.fit_model(pd.concat([long_seoul, gg_train], ignore_index=True)),
            candidate.predict_model,
        ),
        "filtered_joint": (filtered_no_industry, baseline_predict),
        "filtered_seoul_industry_joint": (filtered_industry, candidate.predict_model),
    }
    comparisons = {}
    for period in ("2025Q4", "2026Q2"):
        frame, history = signal.frames_and_history(sq, period)
        comparisons[period] = evaluate(eval_models, frame, gg_frames[period], history, cutoffs)
    # This inspected validation is exploratory, not a claim of independent selection.
    selection_names = ("filtered_joint", "filtered_seoul_industry_joint")
    selected = min(
        selection_names,
        key=lambda name: comparisons["2025Q4"][name]["seoul_guard_pass"]["mae_pp"],
    )
    final_s = pd.concat([filtered_frames[p] for p in seoul_periods], ignore_index=True)
    final_g = pd.concat([gg_frames[p] for p in extended.FINAL_GYEONGGI_TARGETS], ignore_index=True)
    if selected == "filtered_joint":
        eval_bundle = unified.bundle(eval_models[selected][0])
        final = unified.bundle(extended.fit_joint(final_s, final_g))
        model_features = list(candidate.UNIFIED_RESEARCH_FEATURES)
    else:
        eval_bundle = eval_models[selected][0]
        final = candidate.fit_model(pd.concat([final_s, final_g], ignore_index=True))
        model_features = list(candidate.FEATURES)
    rolling = rolling_seoul_only(sq, seoul_periods, seoul_frames)
    report = {
        "artifact_kind": "seoul_guard_filtered_unified_research_v1",
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
        "code_sha256": code_hashes,
        "features": model_features,
        "selected_variant": selected,
        "selection_basis": (
            "lowest Seoul guard-pass MAE on previously inspected 2025Q4; exploratory only"
        ),
        "seoul_filtering_by_period": filtering,
        "evaluation_training_rows": {"seoul": len(filtered_s), "gyeonggi": len(gg_train)},
        "training_periods": {"seoul": eligible, "gyeonggi": ["2025Q2"]},
        "evaluation_guard_thresholds": list(cutoffs),
        "comparisons_previously_inspected": comparisons,
        "seoul_only_rolling_fold_specific_thresholds": rolling,
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
            "If industry selected, Seoul uses original source codes; Gyeonggi category is missing.",
            (
                "Seoul training keeps only rows passing exploratory fixed pre-target-history "
                "guard; no source data deleted."
            ),
            (
                "Fixed policy thresholds fitted on inputs with known 2025Q2 labels, selected "
                "post hoc and not representative of all requests; "
                "not historical real-time training."
            ),
            (
                "Seoul-only historical folds use separately calculated past-training-only "
                "thresholds and differ from fixed current policy and joint model."
            ),
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
    with tempfile.TemporaryDirectory(prefix=".seoul-guard-training-", dir=output.parent) as tmp:
        dest = Path(tmp) / "output"
        dest.mkdir()
        joblib.dump(eval_bundle, dest / "evaluation-model.joblib")
        joblib.dump(final, dest / "final-model.joblib")
        report["model_sha256"] = {p.name: base.sha(p) for p in dest.iterdir()}
        (dest / "metadata.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if hashes != {name: base.sha(path) for name, path in inputs.items()}:
            raise ValueError("Inputs changed during training")
        if code_hashes != {name: base.sha(path) for name, path in code_files.items()}:
            raise ValueError("Code changed during training")
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
