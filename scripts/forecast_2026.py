"""Train and produce research-only 2026 estimates from the 2025 snapshot."""

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.data import read_observations
from pickplace_ai.forecast_2026 import (
    ARTIFACT_KIND,
    FEATURES,
    candidates,
    fit_and_evaluate,
    metrics,
    model_matrix,
    predictions,
    year_rows,
)
from pickplace_ai.periods import quarter_end
from scripts.retrospective_all_industries import _source_manifest


def save_json(path, obj):
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def metrics_by_period(frame, estimates):
    result = {}
    for period, indexes in frame.groupby("target_period").groups.items():
        sub = frame.loc[indexes]
        result[period] = {
            name: metrics(sub, pred[indexes.to_numpy()]) for name, pred in estimates.items()
        }
    return result


def quarter_status(period, as_of):
    return (
        "completed_quarter_unobserved_estimate"
        if quarter_end(period) <= as_of
        else "in_progress_or_future_quarter_estimate"
    )


def run(input_path, output_path, snapshot_at, manifest_path, raw_root, as_of):
    input_path, output_path, manifest_path, raw_root = map(
        Path, (input_path, output_path, manifest_path, raw_root)
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    if as_of < snapshot_at:
        raise ValueError("Analysis date must not precede snapshot availability")
    manifest_bytes = manifest_path.read_bytes()
    _source_manifest(manifest_bytes, raw_root, snapshot_at)
    input_hash = file_sha256(input_path)
    observations = read_observations(input_path)
    if {row.source_kind for row in observations} != {"official"}:
        raise ValueError("Expected official observations")
    if any(row.period[:4] < "2021" or row.period[:4] > "2025" for row in observations):
        raise ValueError("This experiment requires observations from 2021 through 2025 only")
    train, train_omitted = year_rows(observations, 2022, snapshot_at, labels=True)
    validation, validation_omitted = year_rows(observations, 2023, snapshot_at, labels=True)
    test, test_omitted = year_rows(observations, 2024, snapshot_at, labels=True)
    forecast, forecast_omitted = year_rows(observations, 2025, snapshot_at, labels=False)
    if file_sha256(input_path) != input_hash:
        raise ValueError("Input changed during processing")

    (
        winner,
        validation_metrics,
        validation_predictions,
        test_metrics,
        test_predictions,
    ) = fit_and_evaluate(train, validation, test)
    test_by_industry = []
    for industry, indexes in test.groupby("industry_code").groups.items():
        sub = test.loc[indexes]
        values = {
            name: metrics(sub, pred[indexes.to_numpy()]) for name, pred in test_predictions.items()
        }
        test_by_industry.append(
            {
                "industry_code": industry,
                "candidates": values,
                "warnings": ["fewer_than_30_test_rows"] if len(sub) < 30 else [],
            }
        )

    all_labeled = pd.concat([train, validation, test], ignore_index=True)
    known = set(all_labeled.industry_code)
    unknown = ~forecast.industry_code.isin(known)
    forecast_omitted["industry_absent_from_training"] = int(unknown.sum())
    forecast = forecast.loc[~unknown].reset_index(drop=True)
    if forecast.empty:
        raise ValueError("No forecast scopes represented in model training")
    available_industries = {
        row.industry_code for row in observations if row.period.startswith("2025Q")
    }
    forecast_industries = set(forecast.industry_code)
    absent_industries = sorted(available_industries - forecast_industries)
    model = candidates()[winner]
    model.fit(model_matrix(all_labeled), all_labeled.target)
    estimate = predictions(model, forecast)
    learned_winner = min(
        ("ridge", "huber_sgd"), key=lambda name: validation_metrics[name]["mae_ratio"]
    )
    learned_model = candidates()[learned_winner]
    learned_model.fit(model_matrix(all_labeled), all_labeled.target)
    ridge_model = candidates()["ridge"]
    ridge_model.fit(model_matrix(all_labeled), all_labeled.target)
    output = forecast[
        [
            "region_code_system",
            "region_code",
            "industry_code",
            "base_period",
            "comparison_period",
            "target_period",
            "comparison_sales",
        ]
    ].copy()
    output["estimated_yoy_growth"] = estimate
    output["estimated_sales_krw"] = output.comparison_sales * (1 + estimate)
    if not np.isfinite(output.estimated_sales_krw).all():
        raise ValueError("Invalid estimated sales")
    output["estimate_kind"] = output.target_period.map(lambda p: quarter_status(p, as_of))
    output["as_of"] = as_of.isoformat()
    output["snapshot_available_at"] = snapshot_at.isoformat()
    industry_test_rows = test.groupby("industry_code").size().to_dict()
    output["industry_test_rows"] = (
        output.industry_code.map(industry_test_rows).fillna(0).astype(int)
    )
    output["warning"] = output.apply(
        lambda row: ";".join(
            ["research_only_not_point_in_time_validated"]
            + (["fewer_than_30_industry_test_rows"] if row.industry_test_rows < 30 else [])
            + (["comparison_sales_below_1m_krw"] if row.comparison_sales < 1_000_000 else [])
        ),
        axis=1,
    )

    output_path.mkdir(parents=True, exist_ok=False)
    output.to_csv(output_path / "forecast-2026.csv", index=False)
    joblib.dump(
        {
            "artifact_kind": ARTIFACT_KIND,
            "model": model,
            "model_name": winner,
            "feature_columns": FEATURES,
            "known_industries": sorted(known),
        },
        output_path / "model.joblib",
    )
    for filename, name, fitted in (
        ("learned-model.joblib", learned_winner, learned_model),
        ("ridge-model.joblib", "ridge", ridge_model),
    ):
        joblib.dump(
            {
                "artifact_kind": ARTIFACT_KIND,
                "model": fitted,
                "model_name": name,
                "feature_columns": FEATURES,
                "known_industries": sorted(known),
            },
            output_path / filename,
        )
    summary = {
        "artifact_kind": ARTIFACT_KIND,
        "release_status": "experimental_research_only",
        "created_at": datetime.now(UTC).isoformat(),
        "as_of": as_of.isoformat(),
        "snapshot_available_at": snapshot_at.isoformat(),
        "snapshot_basis": (
            "local acquisition; historical publication dates and vintages not verified"
        ),
        "input_sha256": input_hash,
        "source_manifest_sha256": file_sha256(manifest_path),
        "model_sha256": file_sha256(output_path / "model.joblib"),
        "learned_model_sha256": file_sha256(output_path / "learned-model.joblib"),
        "ridge_model_sha256": file_sha256(output_path / "ridge-model.joblib"),
        "forecast_sha256": file_sha256(output_path / "forecast-2026.csv"),
        "feature_code_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "src/pickplace_ai/features.py"
        ),
        "experiment_code_sha256": file_sha256(
            Path(__file__).resolve().parents[1] / "src/pickplace_ai/forecast_2026.py"
        ),
        "runner_sha256": file_sha256(Path(__file__)),
        "versions": package_versions(),
        "origin": "2025Q4 (data cutoff, not historical forecast creation date)",
        "target": "2026Qn aggregate estimated sales / 2025Qn aggregate estimated sales - 1",
        "prediction_floor": -1.0,
        "scope": (
            "pooled model across available Seoul administrative-dong and industries, "
            "not separate industry models"
        ),
        "selection": winner,
        "learned_selection": learned_winner,
        "learned_selection_rule": (
            "minimum 2024 retrospective validation MAE among learned candidates only; "
            "model.joblib remains the overall validation winner"
        ),
        "selection_rule": (
            "minimum 2024 retrospective validation MAE among zero_growth, pooled ridge, "
            "and pooled huber_sgd"
        ),
        "validation": validation_metrics,
        "validation_by_quarter": metrics_by_period(validation, validation_predictions),
        "test": test_metrics,
        "test_by_quarter": metrics_by_period(test, test_predictions),
        "test_by_industry": test_by_industry,
        "rows": {
            "train_2023": len(train),
            "validation_2024": len(validation),
            "test_2025": len(test),
            "forecast_2026": len(output),
        },
        "omitted": {
            "train_2023": train_omitted,
            "validation_2024": validation_omitted,
            "test_2025": test_omitted,
            "forecast_2026": forecast_omitted,
        },
        "final_fit": (
            "2023, 2024, 2025 target labels; 2024 validation selected the method; "
            "refit after untouched 2025 assessment"
        ),
        "test_fit": "2023 and 2024 target labels; 2025 targets held out",
        "features": FEATURES,
        "prediction_method": (
            "Only for trusted artifacts: construct year_rows with the same code, then call "
            "predictions(bundle['model'], frame); it applies model_matrix and the -100% floor. "
            "Calling model.predict on raw inputs is incorrect."
        ),
        "forecast_periods": output.target_period.value_counts().sort_index().to_dict(),
        "forecast_industry_count": len(forecast_industries),
        "source_2025_industry_count": len(available_industries),
        "forecast_absent_industries": [
            {"industry_code": code, "reason": "no_eligible_forecast_rows"}
            for code in absent_industries
        ],
        "limits": [
            "Retrospective assessment uses currently revised data; original release timing "
            "and vintages are unverified.",
            "Completed 2026 quarters are retrospective estimates, not forecasts made before "
            "those quarters; remaining quarters are in-progress or future estimates.",
            "2026 observations are not input, even if a quarter has already finished.",
            "No verified forward performance, prediction intervals, probability of success, "
            "or production approval.",
            "Extreme ratios due to very small comparison sales remain in training and "
            "evaluation; no target trimming.",
            "The -100% prediction floor enforces non-negative sales; no upper prediction cap.",
            "Excluded records reflect missing five-quarter histories, denominators or labels; "
            "absence is not zero.",
            "Population variables are unavailable in the processed snapshot; "
            "no inferred population used.",
        ],
    }
    save_json(output_path / "summary.json", summary)
    (output_path / "source-manifest.json").write_bytes(manifest_bytes)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-at", type=date.fromisoformat, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.input,
        args.output,
        args.snapshot_at,
        args.manifest,
        args.raw_root,
        args.as_of,
    )
    print(
        json.dumps(
            {
                "selection": result["selection"],
                "rows": result["rows"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
