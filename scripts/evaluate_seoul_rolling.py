"""수정 서울 스냅샷의 순차 분기 연구 평가. 과거 실제 공개 시점 검증이 아니다."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.data import read_observations
from pickplace_ai.periods import quarter_index, shift_quarter
from scripts import near_term_forecast as near

SERVICE_INDUSTRIES = ("CS100010", "CS100008", "CS100005")
DEFAULT_ORIGINS = tuple(shift_quarter("2024Q1", offset) for offset in range(8))
KIND = "seoul_rolling_revised_snapshot_evaluation_v1"


def candidates():
    models = near.candidates()
    # 업종 인코더도 각 분기의 훈련 자료에만 적합한다. 미관측 업종은 결측 범주로 둔다.
    transform = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                near.NUMERIC,
            ),
            (
                "industry",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan),
                ["industry_code"],
            ),
        ],
        sparse_threshold=0,
    )
    models["hist_absolute"] = Pipeline(
        [
            ("features", transform),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="absolute_error",
                    max_iter=80,
                    max_leaf_nodes=15,
                    learning_rate=0.05,
                    min_samples_leaf=20,
                    l2_regularization=1,
                    categorical_features=[len(near.NUMERIC)],
                    early_stopping=False,
                    random_state=42,
                ),
            ),
        ]
    )
    return models


def huber_target_candidates():
    huber = near.candidates()["huber_sgd"]
    return {
        "huber_sgd": huber,
        # 큰 성장률을 압축하고, 평가 전에는 원래 성장률 단위로 되돌린다.
        "huber_asinh": TransformedTargetRegressor(
            regressor=clone(huber),
            func=np.arcsinh,
            inverse_func=np.sinh,
        ),
    }


def split_fold(frame, origin, horizon, label_embargo_quarters=1):
    if label_embargo_quarters < 0:
        raise ValueError("Label embargo must be nonnegative")
    quarter_index(origin)
    if horizon not in (1, 2):
        raise ValueError("Only horizons 1 and 2 are supported")
    horizon_rows = frame.loc[frame.horizon == horizon]
    cutoff = shift_quarter(origin, -label_embargo_quarters)
    training = horizon_rows.loc[
        (horizon_rows.target_period <= cutoff) & horizon_rows.target.notna()
    ].copy()
    eligible = horizon_rows.loc[
        (horizon_rows.base_period == origin) & horizon_rows.industry_code.isin(SERVICE_INDUSTRIES)
    ].copy()
    evaluation = eligible.loc[eligible.target.notna()].copy()
    if training.empty or evaluation.empty:
        raise ValueError(f"Missing training or evaluation rows for {origin} horizon {horizon}")
    return training, evaluation.reset_index(drop=True), len(eligible), cutoff


def error_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if (
        actual.ndim != 1
        or len(actual) == 0
        or actual.shape != predicted.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError("Expected equally sized finite actuals and predictions")
    error = np.abs(actual - predicted)
    result = {
        "rows": len(actual),
        "mae_pp": float(np.mean(error) * 100),
        "median_absolute_error_pp": float(np.median(error) * 100),
        "p90_absolute_error_pp": float(np.quantile(error, 0.9) * 100),
        "absolute_error_over_100pp": int((error > 1).sum()),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite aggregate error metrics")
    return result


def evaluate(
    frame,
    origins=DEFAULT_ORIGINS,
    *,
    report_origins=2,
    label_embargo_quarters=1,
    base_counts=None,
    candidate_factory=None,
):
    if candidate_factory is None:
        candidate_factory = candidates
    origins = tuple(origins)
    if len(set(origins)) != len(origins) or list(origins) != sorted(origins, key=quarter_index):
        raise ValueError("Origins must be unique and chronological")
    if not 1 <= report_origins < len(origins):
        raise ValueError("Keep at least one selection and one reporting origin")
    if frame.empty or not set(SERVICE_INDUSTRIES).issubset(set(frame.industry_code)):
        raise ValueError("All three service industries are required")
    keys = ["region_code_system", "region_code", "industry_code", "base_period", "horizon"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate scope, origin and horizon")
    selection_origins = origins[:-report_origins]
    fold_rows, prediction_rows = [], []
    for horizon in (1, 2):
        for origin in origins:
            training, evaluation, eligible_count, cutoff = split_fold(
                frame, origin, horizon, label_embargo_quarters
            )
            phase = "selection" if origin in selection_origins else "reporting"
            for name, model in candidate_factory().items():
                model.fit(near.matrix(training), training.target.to_numpy(float))
                predicted = near.predict(model, evaluation)
                for industry in (*SERVICE_INDUSTRIES, "ALL_SERVICE3"):
                    mask = (
                        np.ones(len(evaluation), dtype=bool)
                        if industry == "ALL_SERVICE3"
                        else (evaluation.industry_code == industry).to_numpy()
                    )
                    subset = evaluation.loc[mask]
                    if subset.empty:
                        raise ValueError(f"Missing evaluation industry {industry} at {origin}")
                    eligible = frame.loc[(frame.horizon == horizon) & (frame.base_period == origin)]
                    eligible = eligible.loc[eligible.industry_code.isin(SERVICE_INDUSTRIES)]
                    eligible_n = (
                        eligible_count
                        if industry == "ALL_SERVICE3"
                        else int((eligible.industry_code == industry).sum())
                    )
                    base_n = (
                        sum(base_counts.get((origin, code), 0) for code in SERVICE_INDUSTRIES)
                        if base_counts is not None and industry == "ALL_SERVICE3"
                        else base_counts.get((origin, industry), 0)
                        if base_counts is not None
                        else None
                    )
                    fold_rows.append(
                        {
                            "phase": phase,
                            "base_period": origin,
                            "target_period": shift_quarter(origin, horizon),
                            "horizon": horizon,
                            "industry_code": industry,
                            "model": name,
                            "training_rows": len(training),
                            "training_label_cutoff": cutoff,
                            "latest_training_target": str(training.target_period.max()),
                            "base_scope_rows": base_n,
                            "eligible_feature_rows": eligible_n,
                            "missing_target_rows": eligible_n - len(subset),
                            "evaluation_coverage": len(subset) / base_n if base_n else None,
                            **error_metrics(subset.target, predicted[mask]),
                        }
                    )
                prediction_rows.extend(
                    {
                        "phase": phase,
                        "horizon": horizon,
                        "industry_code": industry,
                        "model": name,
                        "actual": float(actual),
                        "predicted": float(estimate),
                    }
                    for industry, actual, estimate in zip(
                        evaluation.industry_code, evaluation.target, predicted, strict=True
                    )
                )
    predictions = pd.DataFrame.from_records(prediction_rows)
    aggregate_rows = []
    for (phase, horizon, model), subset in predictions.groupby(["phase", "horizon", "model"]):
        for industry in (*SERVICE_INDUSTRIES, "ALL_SERVICE3"):
            part = (
                subset if industry == "ALL_SERVICE3" else subset[subset.industry_code == industry]
            )
            aggregate_rows.append(
                {
                    "phase": phase,
                    "horizon": int(horizon),
                    "model": model,
                    "industry_code": industry,
                    **error_metrics(part.actual, part.predicted),
                }
            )
    aggregate = pd.DataFrame.from_records(aggregate_rows)
    selected = {}
    for horizon in (1, 2):
        selection = aggregate.loc[
            (aggregate.phase == "selection")
            & (aggregate.horizon == horizon)
            & (aggregate.industry_code == "ALL_SERVICE3")
        ]
        selected[str(horizon)] = str(selection.sort_values(["mae_pp", "model"]).iloc[0]["model"])
    return pd.DataFrame.from_records(fold_rows), aggregate, selected


def run(
    input_path,
    output,
    as_of,
    *,
    origins=DEFAULT_ORIGINS,
    report_origins=2,
    label_embargo_quarters=1,
    training_scope="all",
    compare_huber_targets=False,
):
    input_path, output = Path(input_path), Path(output)
    if output.exists():
        raise FileExistsError("Output already exists")
    if training_scope not in {"all", "service3"}:
        raise ValueError("Training scope must be all or service3")
    input_hash = file_sha256(input_path)
    observations = read_observations(input_path)
    if any(row.available_at > as_of for row in observations):
        raise ValueError("Snapshot was not acquired by as_of")
    if any(row.source_kind != "official" for row in observations):
        raise ValueError("This runner requires official observations")
    source_count = len(observations)
    source_periods = sorted({row.period for row in observations}, key=quarter_index)
    availability = sorted({row.available_at.isoformat() for row in observations})
    if training_scope == "service3":
        observations = [row for row in observations if row.industry_code in SERVICE_INDUSTRIES]
    base_counts = {}
    for row in observations:
        if row.industry_code in SERVICE_INDUSTRIES:
            key = (row.period, row.industry_code)
            base_counts[key] = base_counts.get(key, 0) + 1
    earliest = shift_quarter(min((row.period for row in observations), key=quarter_index), 4)
    latest_origin = max(origins, key=quarter_index)
    all_origins = [
        shift_quarter(earliest, offset)
        for offset in range(quarter_index(latest_origin) - quarter_index(earliest) + 1)
    ]
    frame, omitted = near.build_frames(observations, as_of, all_origins)
    folds, aggregate, selected = evaluate(
        frame,
        origins,
        report_origins=report_origins,
        label_embargo_quarters=label_embargo_quarters,
        base_counts=base_counts,
        candidate_factory=huber_target_candidates if compare_huber_targets else None,
    )
    if file_sha256(input_path) != input_hash:
        raise ValueError("Input changed during evaluation")
    summary = {
        "artifact_kind": KIND,
        "release_status": "research_only_not_serving_artifact",
        "created_at": datetime.now(UTC).isoformat(),
        "as_of": as_of.isoformat(),
        "source_kind": "official",
        "input_sha256": input_hash,
        "source_observations": source_count,
        "source_observation_periods": source_periods,
        "source_available_at_range": [availability[0], availability[-1]],
        "training_scope": training_scope,
        "candidate_set": "huber_target_comparison" if compare_huber_targets else "default",
        "evaluation_industries": list(SERVICE_INDUSTRIES),
        "selection_origins": list(origins[:-report_origins]),
        "reporting_origins": list(origins[-report_origins:]),
        "label_embargo_quarters": label_embargo_quarters,
        "selected_by_selection_mae": selected,
        "omitted_feature_rows": omitted,
        "versions": package_versions(),
        "code_sha256": {
            "runner": file_sha256(Path(__file__)),
            "near_term": file_sha256(Path(near.__file__)),
            "features": file_sha256(Path(near.feature_record.__code__.co_filename)),
        },
        "limits": [
            "Revised snapshot retrospective simulation, not historical point-in-time validation.",
            "Acquisition dates are preserved; actual historical publication dates are unverified.",
            "Label embargo is an experimental assumption, not a verified publication schedule.",
            "Reporting quarters were exposed in earlier research; "
            "they are not an independent test.",
            "Missing actuals are not manufactured; no forecast interval, probability "
            "or production model is produced.",
            "Coverage means evaluation rows / observed base scopes, not a serving guard policy.",
            "Low sales and extreme targets are retained; -100% prediction floor matches near-term.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    folds.to_csv(output / "fold-metrics.csv", index=False)
    aggregate.to_csv(output / "aggregate-metrics.csv", index=False)
    summary["output_sha256"] = {
        name: file_sha256(output / name) for name in ("fold-metrics.csv", "aggregate-metrics.csv")
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--origins", nargs="+", default=list(DEFAULT_ORIGINS))
    parser.add_argument("--report-origins", type=int, default=2)
    parser.add_argument("--label-embargo-quarters", type=int, default=1)
    parser.add_argument("--training-scope", choices=("all", "service3"), default="all")
    parser.add_argument(
        "--compare-huber-targets",
        action="store_true",
        help="Compare Huber training on raw growth and asinh-transformed growth",
    )
    args = parser.parse_args()
    summary = run(
        args.input,
        args.output,
        args.as_of,
        origins=tuple(args.origins),
        report_origins=args.report_origins,
        label_embargo_quarters=args.label_embargo_quarters,
        training_scope=args.training_scope,
        compare_huber_targets=args.compare_huber_targets,
    )
    print(
        json.dumps({"output": str(args.output), "selected": summary["selected_by_selection_mae"]})
    )


if __name__ == "__main__":
    main()
