"""Retrospective Seoul withholding audit; no policy or data mutation.

Only locally trusted joblib artifacts may be loaded. Historical snapshots and
previously inspected evaluation targets cannot establish prospective performance.
"""

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

from pickplace_ai import features, seoul_guard_policy, unified_features
from scripts import audit_seoul_guard as guard
from scripts import audit_seoul_signals as signal
from scripts import train_gyeonggi_research as base
from scripts import train_unified_growth as unified

PERIODS = {"validation": "2025Q4", "reused_test": "2026Q2"}
ROLLING = ("2024Q2", "2024Q4", "2025Q2", "2025Q4", "2026Q2")


def summarize(error: np.ndarray, flagged: np.ndarray) -> dict:
    """Summarize scores and descriptive bands without defining an error label."""
    e = np.asarray(error, dtype=float)
    f = np.asarray(flagged, dtype=bool)
    if (
        e.ndim != 1
        or len(e) == 0
        or f.shape != e.shape
        or not np.isfinite(e).all()
        or (e < 0).any()
    ):
        raise ValueError("Expected finite nonnegative errors and a same-length flag")

    def group(mask: np.ndarray) -> dict:
        part = e[mask]
        return {
            "rows": int(len(part)),
            "fraction_pct": float(mask.mean() * 100),
            "mae_pp": float(part.mean() * 100) if len(part) else None,
            "median_error_pp": float(np.median(part) * 100) if len(part) else None,
            "p90_error_pp": float(np.quantile(part, 0.9) * 100) if len(part) else None,
            "error_le_10pp_rows": int((part <= 0.1).sum()),
            "error_gt_50pp_rows": int((part > 0.5).sum()),
            "error_gt_100pp_rows": int((part > 1).sum()),
        }

    return {"all": group(np.ones(len(e), dtype=bool)), "withheld": group(f), "retained": group(~f)}


def split_groups(frame: pd.DataFrame, code_col: str, name_col: str) -> pd.DataFrame:
    """Report per-code sample sizes and conditional error distributions."""
    rows = []
    for (code, name), block in frame.groupby([code_col, name_col], dropna=False, sort=True):
        result = summarize(block.absolute_error.to_numpy(), block.withheld.to_numpy())
        all_, withheld, retained = (result[k] for k in ("all", "withheld", "retained"))
        rows.append(
            {
                "code": code,
                "name_2025_reference": "" if pd.isna(name) else name,
                "rows": all_["rows"],
                "withheld_rows": withheld["rows"],
                "withheld_fraction_pct": withheld["fraction_pct"],
                "overall_mae_pp": all_["mae_pp"],
                "overall_median_error_pp": all_["median_error_pp"],
                "retained_rows": retained["rows"],
                "retained_mae_pp": retained["mae_pp"],
                "retained_median_error_pp": retained["median_error_pp"],
                "withheld_error_le_10pp_rows": withheld["error_le_10pp_rows"],
                "retained_error_gt_50pp_rows": retained["error_gt_50pp_rows"],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["rows", "code"], ascending=[False, True])
        .reset_index(drop=True)
    )


def names_from_2025(path: Path) -> tuple[dict, dict]:
    keys = ["행정동_코드", "행정동_코드_명", "서비스_업종_코드", "서비스_업종_코드_명"]
    refs = pd.read_csv(path, encoding="cp949", dtype=str, usecols=keys)

    def mapping(code: str, name: str) -> dict:
        return refs.groupby(code)[name].agg(lambda s: " / ".join(sorted(set(s.dropna())))).to_dict()

    return mapping(keys[0], keys[1]), mapping(keys[2], keys[3])


def ensure_api_flags_match(history: np.ndarray, flags: dict[str, np.ndarray]) -> None:
    """Require the study's fixed masks to match the actual input-only API policy."""
    names = {
        "low_reference": "low_reference_sales",
        "rapid_increase": "rapid_sales_increase",
        "rapid_decrease": "rapid_sales_decrease",
        "low_latest": "low_latest_sales",
    }
    if set(flags) != set(names) or any(len(mask) != len(history) for mask in flags.values()):
        raise ValueError("Study flag schema mismatch")
    for index, row in enumerate(history):
        expected = {names[key] for key, mask in flags.items() if mask[index]}
        actual = set(seoul_guard_policy.assess_sales_history(row.tolist()))
        if expected != actual:
            raise ValueError("Study flags do not match API policy")


def store_history_signal(history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eligibility and max absolute prior-quarter store-count change."""
    values = np.asarray(history, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Expected four store-count quarters")
    valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    valid &= (values[:, :3] > 0).all(axis=1)
    change = np.full(len(values), np.nan)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        change[valid] = np.max(np.abs(values[valid, 1:] / values[valid, :-1] - 1), axis=1)
    return valid, change


def store_quarters(path: Path) -> pd.DataFrame:
    """Read original store_count; do not fill missing counts or infer code links."""
    columns = ["region_code", "industry_code", "period", "store_count"]
    data = pd.read_csv(path, usecols=columns, dtype={"region_code": str, "industry_code": str})
    if data.duplicated(columns[:3]).any():
        raise ValueError("Duplicate store-count quarter")
    data["store_count"] = pd.to_numeric(data.store_count, errors="raise")
    data = data.rename(columns={"region_code": "행정동코드", "industry_code": "중분류업종코드"})
    data["period"] = pd.PeriodIndex(data.period, freq="Q")
    return data.pivot(index=base.SCOPE, columns="period", values="store_count")


def aligned_store_history(stores: pd.DataFrame, frame: pd.DataFrame, target: str) -> np.ndarray:
    """Use exactly target q-4 through q-1, never the target store count."""
    keys = pd.MultiIndex.from_frame(frame[base.SCOPE])
    quarter = pd.Period(target, freq="Q")
    periods = [quarter - offset for offset in (4, 3, 2, 1)]
    return stores.reindex(index=keys, columns=periods).to_numpy(dtype=float)


def subset_metrics(error: np.ndarray, mask: np.ndarray) -> dict:
    """Coverage denominator is all evaluable sales rows, including missing stores."""
    values = error[mask]
    return {
        "rows": int(mask.sum()),
        "coverage_pct_of_evaluable_sales": float(mask.mean() * 100),
        "mae_pp": float(values.mean() * 100) if len(values) else None,
        "median_error_pp": float(np.median(values) * 100) if len(values) else None,
        "error_le_10pp_rows": int((values <= 0.1).sum()),
        "error_gt_50pp_rows": int((values > 0.5).sum()),
    }


def store_group_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare coverage by industry; missing store histories remain explicit."""
    records = []
    for (code, name), block in frame.groupby(["중분류업종코드", "industry_name"], dropna=False):
        evaluable = block.store_valid.to_numpy(dtype=bool)
        existing = block.withheld.to_numpy(dtype=bool)
        store_flag = block.store_flag.to_numpy(dtype=bool)
        additional = evaluable & ~existing & store_flag
        error = block.absolute_error.to_numpy(dtype=float)
        records.append(
            {
                "industry_code": code,
                "industry_name_2025_reference": "" if pd.isna(name) else name,
                "rows": len(block),
                "store_valid_rows": int(evaluable.sum()),
                "store_unavailable_rows": int((~evaluable).sum()),
                "existing_guard_retained_rows_same_store_subset": int(
                    (evaluable & ~existing).sum()
                ),
                "combined_guard_retained_rows": int((evaluable & ~existing & ~store_flag).sum()),
                "additional_store_flagged_rows": int(additional.sum()),
                "additional_flag_error_le_10pp_rows": int((additional & (error <= 0.1)).sum()),
                "additional_flag_error_gt_50pp_rows": int((additional & (error > 0.5)).sum()),
            }
        )
    return pd.DataFrame(records).sort_values("rows", ascending=False).reset_index(drop=True)


def rolling_cutoff(target: str) -> tuple[str, str]:
    """Fix the label cutoff and skip one target quarter before evaluation."""
    quarter = pd.Period(target, freq="Q")
    return str(quarter - 2), str(quarter - 1)


def run(*, source: Path, names: Path, model_root: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh output directory")
    paths = {
        "source": source,
        "names_2025": names,
        "metadata": model_root / "metadata.json",
        "evaluation_model": model_root / "evaluation-pooled.joblib",
        "final_model": model_root / "final-model.joblib",
    }
    hashes = {k: base.sha(v) for k, v in paths.items()}
    meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if (
        meta.get("artifact_kind") != unified.KIND
        or meta.get("purpose") != "retrospective_research_only_not_service"
    ):
        raise ValueError("Unexpected artifact")
    if (
        meta["source_hashes"]["seoul"] != hashes["source"]
        or meta["model_sha256"]["evaluation-pooled.joblib"] != hashes["evaluation_model"]
    ):
        raise ValueError("Source or evaluation model mismatch")
    if not seoul_guard_policy.policy_compatible(meta, hashes["final_model"]):
        raise ValueError("Guard policy bound to a different model")
    codepaths = {
        "trainer": Path(unified.__file__),
        "feature_code": Path(features.__file__),
        "unified_feature_code": Path(unified_features.__file__),
        "base_trainer": Path(base.__file__),
        "guard_policy": Path(seoul_guard_policy.__file__),
        "guard_audit": Path(guard.__file__),
        "signal_audit": Path(signal.__file__),
        "this_script": Path(__file__),
    }
    code_hashes = {k: base.sha(v) for k, v in codepaths.items()}
    for key in ("trainer", "feature_code", "unified_feature_code", "base_trainer"):
        if meta["code_sha256"][key] != code_hashes[key]:
            raise ValueError("Model code mismatch")
    # A checksum detects corruption but cannot make an untrusted pickle safe.
    model = joblib.load(paths["evaluation_model"])
    quarters = unified.seoul_quarters(source)
    store_data = store_quarters(source)
    training_frame, training_history = signal.frames_and_history(quarters, "2025Q2")
    training_stores = aligned_store_history(store_data, training_frame, "2025Q2")
    store_train_valid, store_train_change = store_history_signal(training_stores)
    if not store_train_valid.any():
        raise ValueError("No valid training store histories")
    store_threshold = float(np.quantile(store_train_change[store_train_valid], 0.95))
    cutoffs = guard.training_thresholds(training_history)
    policy_thresholds = (
        seoul_guard_policy.LOW_REFERENCE_SALES_KRW,
        seoul_guard_policy.MAX_QOQ_INCREASE,
        seoul_guard_policy.MIN_QOQ_DECREASE,
        seoul_guard_policy.LOW_LATEST_SALES_KRW,
    )
    if not all(a == b for a, b in zip(cutoffs, policy_thresholds, strict=True)):
        raise ValueError("Training thresholds do not match current policy")
    region_names, industry_names = names_from_2025(names)
    summaries = {}
    store_study = {
        "purpose": "exploratory_alternative_not_applied_to_api",
        "training_target_period": "2025Q2",
        "training_evaluable_sales_rows": len(training_frame),
        "training_store_valid_rows": int(store_train_valid.sum()),
        "training_store_unavailable_rows": int((~store_train_valid).sum()),
        "store_change_quantile": 0.95,
        "maximum_absolute_prior_qoq_change_above": store_threshold,
        "periods": {},
        "limitations": [
            "Known-target, four-quarter sales histories only; request coverage may differ.",
            "Missing or zero prior store counts are not imputed; they are reported separately.",
            "Current revised snapshots do not establish actual historical availability.",
            "2026Q2 is a previously inspected reused test, not independent evidence.",
            "Store_count source meaning and historical reporting changes require validation.",
            "One threshold, two inspected periods, no prospective or causal validation.",
            "Gyeonggi store-count equivalence is unknown and is not evaluated.",
            "No source rows or current API policy are modified.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".seoul-guard-policy-review-", dir=output.parent
    ) as tmp:
        dest = Path(tmp) / "result"
        dest.mkdir()
        for phase, period in PERIODS.items():
            frame, history = signal.frames_and_history(quarters, period)
            flags = guard.guard_flags(history, *cutoffs)
            ensure_api_flags_match(history, flags)
            flagged = np.column_stack(list(flags.values())).any(axis=1)
            pred = unified.predict_unified_model(
                model, frame[list(unified_features.UNIFIED_RESEARCH_FEATURES)]
            )
            error = np.abs(frame.target.to_numpy(dtype=float) - pred)
            frame["withheld"] = flagged
            frame["absolute_error"] = error
            frame["region_name"] = frame["행정동코드"].map(region_names)
            frame["industry_name"] = frame["중분류업종코드"].map(industry_names)
            store_history = aligned_store_history(store_data, frame, period)
            store_valid, store_change = store_history_signal(store_history)
            store_flag = store_valid & (store_change > store_threshold)
            frame["store_valid"] = store_valid
            frame["store_flag"] = store_flag
            store_study["periods"][phase] = {
                "target_period": period,
                "all_evaluable_sales": subset_metrics(error, np.ones(len(error), dtype=bool)),
                "valid_store_histories": subset_metrics(error, store_valid),
                "unavailable_store_histories": subset_metrics(error, ~store_valid),
                "current_guard_retained_same_store_subset": subset_metrics(
                    error, store_valid & ~flagged
                ),
                "with_store_signal_retained_same_subset": subset_metrics(
                    error, store_valid & ~flagged & ~store_flag
                ),
                "newly_flagged_by_store_signal": subset_metrics(
                    error, store_valid & ~flagged & store_flag
                ),
            }
            store_group_stats(frame).to_csv(dest / f"{phase}_store_industries.csv", index=False)
            for group, code_col, name_col in (
                ("regions", "행정동코드", "region_name"),
                ("industries", "중분류업종코드", "industry_name"),
            ):
                split_groups(frame, code_col, name_col).to_csv(
                    dest / f"{phase}_{group}.csv", index=False
                )
            summaries[phase] = {
                "target_period": period,
                **summarize(error, flagged),
                "signal_counts": {key: int(mask.sum()) for key, mask in flags.items()},
                "overlap_two_or_more": int(
                    (np.column_stack(list(flags.values())).sum(axis=1) >= 2).sum()
                ),
                "missing_region_names": int(frame.region_name.isna().sum()),
                "missing_industry_names": int(frame.industry_name.isna().sum()),
            }
        rolling = []
        columns = list(unified_features.UNIFIED_RESEARCH_FEATURES)
        for target in ROLLING:
            train_target, embargo = rolling_cutoff(target)
            train_frame, train_history = signal.frames_and_history(quarters, train_target)
            test_frame, test_history = signal.frames_and_history(quarters, target)
            rolling_cutoffs = guard.training_thresholds(train_history)
            flags = guard.guard_flags(test_history, *rolling_cutoffs)
            flagged = np.column_stack(list(flags.values())).any(axis=1)
            # A different Seoul-only model is fit for each target. No later labels enter fit.
            fitted = unified.model().fit(train_frame[columns], train_frame.target)
            pred = unified.predict_unified_model(unified.bundle(fitted), test_frame[columns])
            error = np.abs(test_frame.target.to_numpy(dtype=float) - pred)
            rolling.append(
                {
                    "target_period": target,
                    "train_target_period": train_target,
                    "embargo_target_period": embargo,
                    "train_rows": len(train_frame),
                    "thresholds": [float(value) for value in rolling_cutoffs],
                    **summarize(error, flagged),
                }
            )
        if hashes != {k: base.sha(v) for k, v in paths.items()} or code_hashes != {
            k: base.sha(v) for k, v in codepaths.items()
        }:
            raise ValueError("Input or code changed during audit")
        summary = {
            "purpose": "retrospective_policy_audit_not_independent_not_production_validation",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "model_kind": unified.KIND,
            "policy_version": seoul_guard_policy.POLICY_VERSION,
            "inputs_sha256": hashes,
            "code_sha256": code_hashes,
            "fixed_policy_thresholds": list(policy_thresholds),
            "fixed_pooled_evaluation": summaries,
            "store_signal_study": store_study,
            "seoul_only_rolling": rolling,
            "descriptive_error_bands": (
                "<=10pp and >50pp are descriptive only, not labels "
                "for source errors or policy correctness"
            ),
            "limitations": [
                "All samples require known outcome and complete input history; "
                "request coverage may differ.",
                "2025Q4 has been used repeatedly in analysis and selection; "
                "2026Q2 is a previously examined reused test.",
                "Rolling uses Seoul-only re-fitted models, not the existing pooled API model.",
                "Current corrected snapshots do not verify historical availability or boundaries; "
                "no prospective inference is established.",
                "A withheld low-error sample is not evidence of a model error, "
                "and retained samples may still be inaccurate.",
                "Group names refer to 2025 source labels, not verified crosswalks.",
                "No source rows are deleted; no model or API policy is updated.",
            ],
        }
        paths_out = sorted(dest.glob("*.csv"))
        summary["output_sha256"] = {p.name: base.sha(p) for p in paths_out}
        (dest / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError("Use a fresh output directory")
        os.rename(dest, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        source=args.source, names=args.names, model_root=args.model_root, output=args.output
    )
    print(json.dumps(result["fixed_pooled_evaluation"], ensure_ascii=False))


if __name__ == "__main__":
    main()
