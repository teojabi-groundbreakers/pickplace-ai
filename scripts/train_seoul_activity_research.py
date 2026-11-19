"""Compare Seoul sales-growth models with observed activity and store covariates.

This is a current-snapshot retrospective experiment, not a serving model or a
validation of legal zoning. Historical publication dates remain unverified.
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

from pickplace_ai import seoul_activity_features as activity
from pickplace_ai.artifacts import package_versions
from scripts import audit_seoul_guard as guard
from scripts import train_gyeonggi_research as base
from scripts import train_unified_extended_history as extended

KIND = "seoul_activity_growth_research_v2"
EVALUATION_TARGETS = ("2025Q4", "2026Q2")


def make_target_frame(
    quarters: pd.DataFrame, source: pd.DataFrame, target_period: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Join only target-minus-one covariates to four pre-target sales quarters."""
    target = pd.Period(target_period, freq="Q")
    frame = base.make_frame(quarters, target_period)
    base_period = str(target - 1)
    scope = list(activity.SCOPE)
    lookup = source.loc[source.period.eq(base_period)].rename(
        columns={"region_code": scope[0], "industry_code": scope[1]}
    )
    lookup = lookup[[*scope, *activity.POPULATION_FEATURES, *activity.STORE_FEATURES]]
    if lookup.duplicated(scope).any():
        raise ValueError("Duplicate source features for target base period")
    frame = frame.merge(lookup, on=scope, how="left", validate="one_to_one", indicator=True)
    if (
        not frame._merge.eq("both").all()
        or frame[list(activity.FEATURE_SETS["sales_population_stores"])].isna().any().any()
    ):
        raise ValueError(f"Incomplete Seoul base-period features for {target_period}")
    frame = frame.drop(columns="_merge")
    frame["base_period"] = base_period
    keys = pd.MultiIndex.from_frame(frame[scope])
    history = quarters.loc[keys, [target - n for n in (4, 3, 2, 1)]].to_numpy(dtype=float)
    guard.minimum_growth(history)  # Validate all four pre-target values.
    return frame, history


def encoded_features(frame: pd.DataFrame, name: str, encoder: OrdinalEncoder) -> pd.DataFrame:
    """Use the fold's industry mapping for both learning and prediction."""
    columns = list(activity.FEATURE_SETS[name])
    result = frame[columns].copy()
    result["seoul_industry"] = encoder.transform(frame[[activity.SCOPE[1]]]).ravel()
    return result[[*columns, "seoul_industry"]]


def fit(frame: pd.DataFrame, name: str) -> dict:
    columns = list(activity.FEATURE_SETS[name])
    if frame.empty or frame[columns].isna().any().any() or frame[activity.SCOPE[1]].isna().any():
        raise ValueError("Empty or missing training features")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan).fit(
        frame[[activity.SCOPE[1]]]
    )
    if len(encoder.categories_[0]) > 255:
        raise ValueError("Too many industry categories")
    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1,
        early_stopping=False,
        random_state=42,
        categorical_features=[False] * len(columns) + [True],
    )
    model.fit(encoded_features(frame, name, encoder), frame.target)
    return {"model": model, "encoder": encoder}


def predict(bundle: dict, frame: pd.DataFrame, name: str) -> np.ndarray:
    result = np.asarray(
        bundle["model"].predict(encoded_features(frame, name, bundle["encoder"])), dtype=float
    )
    if result.shape != (len(frame),) or not np.isfinite(result).all():
        raise ValueError("Invalid Seoul research prediction")
    return np.maximum(result, -1.0)


def metrics(actual: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> dict:
    y = np.asarray(actual, dtype=float)[mask]
    p = np.asarray(prediction, dtype=float)[mask]
    if not len(y):
        return {"rows": 0, "mae_pp": None, "median_ae_pp": None, "p90_ae_pp": None}
    error = np.abs(y - p) * 100
    return {
        "rows": len(y),
        "mae_pp": float(error.mean()),
        "median_ae_pp": float(np.median(error)),
        "p90_ae_pp": float(np.quantile(error, 0.9)),
    }


def guard_pass(history: np.ndarray, thresholds: tuple[float, float, float, float]) -> np.ndarray:
    flags = guard.guard_flags(history, *thresholds)
    return ~np.logical_or.reduce(list(flags.values()))


def compare_fold(
    frames: dict[str, tuple[pd.DataFrame, np.ndarray]], target: str
) -> tuple[dict, dict[str, dict]]:
    periods = sorted(frames, key=lambda period: pd.Period(period, freq="Q"))
    earlier = extended.before_embargo(periods, target)
    if not earlier:
        raise ValueError("No earlier training targets")
    thresholds = guard.training_thresholds(frames[earlier[-1]][1])
    training = pd.concat(
        [frames[period][0].loc[guard_pass(frames[period][1], thresholds)] for period in earlier],
        ignore_index=True,
    )
    evaluation, history = frames[target]
    eligible = guard_pass(history, thresholds)
    if training.empty or not eligible.any():
        raise ValueError("No eligible Seoul training or evaluation rows")
    actual = evaluation.target.to_numpy(dtype=float)
    report = {
        "training_target_periods": earlier,
        "embargo_target_period": str(pd.Period(target, freq="Q") - 1),
        "training_rows_guard_pass": len(training),
        "evaluation_rows_all": len(evaluation),
        "evaluation_rows_guard_pass": int(eligible.sum()),
        "guard_thresholds_fitted_on_prior_inputs": list(thresholds),
        "models": {},
    }
    fitted = {}
    for name in activity.FEATURE_SETS:
        model = fit(training, name)
        prediction = predict(model, evaluation, name)
        report["models"][name] = {
            "guard_pass": metrics(actual, prediction, eligible),
            "all": metrics(actual, prediction, np.ones(len(actual), dtype=bool)),
        }
        fitted[name] = model
    zero = np.zeros(len(actual))
    report["models"]["zero_growth"] = {
        "guard_pass": metrics(actual, zero, eligible),
        "all": metrics(actual, zero, np.ones(len(actual), dtype=bool)),
    }
    # Describe comparative error by an input-only activity index. Cutoffs are
    # fitted to the training rows and do not encode legal zoning.
    day_column = "day_overnight_hourly_log_ratio"
    cutoffs = np.quantile(training[day_column], [1 / 3, 2 / 3]).tolist()
    values = evaluation[day_column].to_numpy(dtype=float)
    groups = {
        "lower_day_activity_index": values < cutoffs[0],
        "middle_day_activity_index": (values >= cutoffs[0]) & (values < cutoffs[1]),
        "higher_day_activity_index": values >= cutoffs[1],
    }
    report["activity_index_cutoffs_from_training"] = cutoffs
    report["activity_groups_guard_pass"] = {
        group: {
            name: metrics(actual, predict(fitted[name], evaluation, name), eligible & mask)
            for name in ("sales_only", "sales_population_stores")
        }
        for group, mask in groups.items()
    }
    return report, fitted


def train(snapshot: Path, floating: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Never overwrite a Seoul research artifact")
    input_paths = {
        "snapshot": snapshot / "snapshot.sqlite3",
        "snapshot_manifest": snapshot / "manifest.json",
        "floating_csv": floating,
    }
    # 읽기 전 지문을 고정해야 학습 도중 바뀐 파일을 학습 입력으로 기록하지 않는다.
    input_hashes = {name: activity.sha256(path) for name, path in input_paths.items()}
    quarters, source, manifest = activity.load_snapshot(snapshot, floating)
    covariates = activity.activity_covariates(
        source.resident_total.to_numpy(),
        source.floating_total.to_numpy(),
        source.store_count.to_numpy(),
        source[list(activity.TIME_COLUMNS)].to_numpy(),
    )
    source = pd.concat([source.reset_index(drop=True), covariates], axis=1)
    # One region-quarter is one reference observation. Repeating it once for
    # every industry would bias the activity reference toward busy industries.
    unique_activity = source.drop_duplicates(["period", "region_code"])
    profile_reference = {
        "kind": "relative_activity_pattern_reference_not_legal_zoning",
        "region_quarter_rows": len(unique_activity),
        "day_overnight_hourly_log_ratio_tertiles": np.quantile(
            unique_activity.day_overnight_hourly_log_ratio, [1 / 3, 2 / 3]
        ).tolist(),
        "evening_day_hourly_log_ratio_tertiles": np.quantile(
            unique_activity.evening_day_hourly_log_ratio, [1 / 3, 2 / 3]
        ).tolist(),
        "interpretation": (
            "Relative observed time-band pattern only; no residential, commercial, "
            "industrial, or green zoning label and no calibrated probability."
        ),
    }
    targets = [str(period) for period in pd.period_range("2022Q1", "2026Q2", freq="Q")]
    frames = {target: make_target_frame(quarters, source, target) for target in targets}
    comparisons = {target: compare_fold(frames, target)[0] for target in EVALUATION_TARGETS}
    # A final exploratory refit uses all known labels. No independent score is
    # attached to this model; the older folds above evaluate separate fits.
    final_thresholds = guard.training_thresholds(frames[targets[-1]][1])
    final_rows = pd.concat(
        [
            frames[period][0].loc[guard_pass(frames[period][1], final_thresholds)]
            for period in targets
        ],
        ignore_index=True,
    )
    final_model = fit(final_rows, "sales_population_stores")
    if input_hashes != {name: activity.sha256(path) for name, path in input_paths.items()}:
        raise ValueError("Source changed during training")
    code_paths = {
        "trainer": Path(__file__),
        "activity_features": Path(activity.__file__),
        "shared_sales_features": Path(activity.features.__file__),
        "base_frame": Path(base.__file__),
        "guard_diagnostic": Path(guard.__file__),
    }
    report = {
        "artifact_kind": KIND,
        "purpose": "retrospective_research_only_not_service",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_created_at_utc": manifest["created_at_utc"],
        "historical_publication_date_verified": False,
        "source_code_system": "seoul-adstrd-8",
        "industry_code_system": "seoul-CS",
        "sales_source": "Seoul estimated sales, not raw card sales",
        "target": "sales[target]/sales[target_minus_4]-1",
        "categorical_industry": "Seoul CS code, ordinal encoded on training rows only",
        "feature_sets": {name: list(columns) for name, columns in activity.FEATURE_SETS.items()},
        "source_hashes": input_hashes,
        "code_sha256": {name: activity.sha256(path) for name, path in code_paths.items()},
        "source_rows": len(source),
        "activity_profile_reference": profile_reference,
        "target_periods": targets,
        "comparisons_previously_inspected": comparisons,
        "final_refit": {
            "model": "sales_population_stores",
            "training_rows_guard_pass": len(final_rows),
            "industry_categories": len(final_model["encoder"].categories_[0]),
            "thresholds_fitted_on_final_inputs": list(final_thresholds),
            "independent_performance": None,
        },
        "versions": package_versions(),
        "limitations": [
            "No official residential/commercial/industrial/green zoning labels are present.",
            "Resident stock and floating aggregate have different units; no ratio is used.",
            "Time-band indices are relative aggregates, not daily people-per-hour or customers.",
            "Current revised snapshot does not prove past release or correction times.",
            "2025Q4 and 2026Q2 were previously inspected; neither is independent testing.",
            "Guard thresholds are exploratory input screens, not proof of invalid records.",
            "Final refit has no independent performance estimate.",
            "This Seoul-only model has no validated H10/CS bridge and no serving API.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seoul-activity-", dir=output.parent) as tmp:
        stage = Path(tmp) / "model"
        stage.mkdir()
        model_path = stage / "final-model.joblib"
        joblib.dump(
            {
                "artifact_kind": KIND,
                "feature_columns": list(activity.FEATURE_SETS["sales_population_stores"]),
                "categorical_feature": "seoul_industry",
                **final_model,
            },
            model_path,
        )
        report["model_sha256"] = activity.sha256(model_path)
        (stage / "metadata.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # 직렬화 중 변경도 잡아 최종 경로에 불일치 산출물이 발행되지 않게 한다.
        if input_hashes != {name: activity.sha256(path) for name, path in input_paths.items()}:
            raise ValueError("Source changed during training")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output already exists")
        os.rename(stage, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--floating", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = train(args.snapshot, args.floating, args.output)
    print(json.dumps(report["comparisons_previously_inspected"], ensure_ascii=False))


if __name__ == "__main__":
    main()
