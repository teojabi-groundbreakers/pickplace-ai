"""서울 다음 분기 예측의 평균·큰 오차를 동일한 수정 스냅샷에서 비교한다."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor

from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.data import read_observations
from pickplace_ai.periods import quarter_index, shift_quarter
from scripts import evaluate_seoul_rolling as rolling
from scripts import near_term_forecast as near
from scripts.analyze_unified_seoul import denominator_bins, read_name_maps
from scripts.build_seoul_fit_reference import _read_mapping

SELECTION_ORIGINS = rolling.DEFAULT_ORIGINS[:-2]
REPORT_ORIGINS = rolling.DEFAULT_ORIGINS[-2:]
ADDITIONAL_ORIGIN = "2026Q1"
REGION_REFERENCE_DATE = date(2025, 6, 30)
RAW = "huber_sgd"
ASINH = "huber_asinh"
BLEND = "huber_equal_blend"
GENTLE = "huber_gentle_asinh"
TAIL_ONLY = "huber_tail_only_asinh"
LOW_SALES_CUTOFFS = {"huber_asinh_under_10m": 10_000_000, "huber_asinh_under_100m": 100_000_000}
MODEL_NAMES = (RAW, ASINH, BLEND, GENTLE, TAIL_ONLY, *LOW_SALES_CUTOFFS)
SCOPE = ["region_code_system", "region_code", "industry_code", "base_period"]
AUDIT_FIELDS = [*SCOPE, "target_period", "target", "comparison_sales", "sales_yoy", "sales_qoq"]


def gentle_transform(values):
    # 0 근처 기울기를 유지하면서 큰 성장률의 압축 강도를 낮춘다.
    return 2 * np.arcsinh(values / 2)


def gentle_inverse(values):
    return 2 * np.sinh(values / 2)


def tail_transform(values):
    # 성장률 100%까지는 원래 단위를 유지하고 그 위 구간만 압축한다.
    return np.minimum(values, 1) + np.arcsinh(np.maximum(values - 1, 0))


def tail_inverse(values):
    return np.minimum(values, 1) + np.sinh(np.maximum(values - 1, 0))


def fold_predictions(training, evaluation):
    models = rolling.huber_target_candidates()
    models[GENTLE] = TransformedTargetRegressor(
        regressor=clone(models[RAW]), func=gentle_transform, inverse_func=gentle_inverse
    )
    models[TAIL_ONLY] = TransformedTargetRegressor(
        regressor=clone(models[RAW]), func=tail_transform, inverse_func=tail_inverse
    )
    prediction = evaluation[AUDIT_FIELDS].copy()
    train_features = near.matrix(training)
    for name, model in models.items():
        model.fit(train_features, training.target.to_numpy(float))
        prediction[name] = near.predict(model, evaluation)
    prediction[BLEND] = (prediction[RAW] + prediction[ASINH]) / 2
    for name, cutoff in LOW_SALES_CUTOFFS.items():
        prediction[name] = prediction[ASINH].where(
            prediction.comparison_sales < cutoff, prediction[RAW]
        )
    return prediction


def choose_candidate(selection):
    metrics = {
        name: rolling.error_metrics(selection.target, selection[name]) for name in MODEL_NAMES
    }
    reference = metrics[RAW]
    eligible = [
        name
        for name in MODEL_NAMES
        if metrics[name]["p90_absolute_error_pp"] <= reference["p90_absolute_error_pp"]
        and metrics[name]["absolute_error_over_100pp"] <= reference["absolute_error_over_100pp"]
    ]
    return min(eligible, key=lambda name: (metrics[name]["mae_pp"], MODEL_NAMES.index(name)))


def summarize(pairs, fields):
    records = []
    for key, part in pairs.groupby(fields, dropna=False, observed=True):
        group = dict(zip(fields, key, strict=True))
        for name in MODEL_NAMES:
            records.append(
                {**group, "model": name, **rolling.error_metrics(part.target, part[name])}
            )
    return pd.DataFrame.from_records(records)


def fixed_p90_transitions(reporting):
    cutoff = reporting[f"{RAW}_error_pp"].quantile(0.9)
    raw_tail = reporting[f"{RAW}_error_pp"] > cutoff
    asinh_tail = reporting[f"{ASINH}_error_pp"] > cutoff
    changes = reporting.assign(entered=~raw_tail & asinh_tail, exited=raw_tail & ~asinh_tail)
    return pd.concat(
        [
            changes.groupby(field, dropna=False)
            .agg(rows=("entered", "size"), entered=("entered", "sum"), exited=("exited", "sum"))
            .rename_axis("group")
            .reset_index()
            .assign(dimension=field, raw_p90_pp=cutoff)
            for field in ("industry_code", "district_name_2025q2")
        ],
        ignore_index=True,
    )


def read_districts(mapping_file, sgis_file):
    mapping = _read_mapping(mapping_file, REGION_REFERENCE_DATE)
    areas = pd.read_csv(
        sgis_file, dtype=str, usecols=["reference_date", "sgis_code", "district_name"]
    )
    if (
        set(areas.reference_date) != {REGION_REFERENCE_DATE.isoformat()}
        or areas.sgis_code.duplicated().any()
        or areas[["sgis_code", "district_name"]].isna().any().any()
    ):
        raise ValueError("Expected unique SGIS district labels for 2025-06-30")
    districts = areas.set_index("sgis_code").district_name
    return {seoul: districts.loc[sgis] for seoul, (sgis, _h10) in mapping.items()}


def evaluate(frame):
    if frame.duplicated([*SCOPE, "horizon"]).any():
        raise ValueError("Duplicate scope, origin and horizon")
    periods = {
        "selection": SELECTION_ORIGINS,
        "reporting": REPORT_ORIGINS,
        "additional_check": (ADDITIONAL_ORIGIN,),
    }
    predictions, fold_details = [], []
    for phase, origins in periods.items():
        phase_predictions = []
        for origin in origins:
            train, test, feature_rows, cutoff = rolling.split_fold(frame, origin, horizon=1)
            prediction = fold_predictions(train, test)
            phase_predictions.append(prediction.assign(phase=phase))
            fold_details.append(
                {
                    "phase": phase,
                    "base_period": origin,
                    "training_label_cutoff": cutoff,
                    "eligible_feature_rows": feature_rows,
                    "evaluation_rows": len(test),
                }
            )
        predictions.extend(phase_predictions)
        if phase == "selection":
            selected = choose_candidate(pd.concat(phase_predictions, ignore_index=True))
    return pd.concat(predictions, ignore_index=True), selected, fold_details


def run(input_path, name_reference, region_mapping, sgis_reference, output, as_of):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output already exists")
    inputs = {
        "snapshot": input_path,
        "names_2025": name_reference,
        "region_mapping_2025q2": region_mapping,
        "sgis_2025q2": sgis_reference,
    }
    hashes = {name: file_sha256(path) for name, path in inputs.items()}
    observations = read_observations(input_path)
    if any(
        row.available_at > as_of
        or row.source_kind != "official"
        or row.region_code_system != "seoul-adstrd"
        for row in observations
    ):
        raise ValueError("Use official Seoul observations acquired by the analysis date")
    earliest = shift_quarter(min(row.period for row in observations), 4)
    origins = tuple(
        shift_quarter(earliest, offset)
        for offset in range(quarter_index(ADDITIONAL_ORIGIN) - quarter_index(earliest) + 1)
    )
    frame, _ = near.build_frames(observations, as_of, origins)
    pairs, selected, folds = evaluate(frame)
    regions, industries = read_name_maps(name_reference)
    pairs["region_name_2025"] = pairs.region_code.map(regions)
    pairs["industry_name"] = pairs.industry_code.map(industries)
    pairs["district_name_2025q2"] = pairs.region_code.map(
        read_districts(region_mapping, sgis_reference)
    )
    pairs["comparison_sales_band"] = denominator_bins(pairs.comparison_sales)
    pairs["actual_growth_band"] = pd.cut(
        pairs.target,
        [-np.inf, -0.25, 0, 0.25, 1, np.inf],
        labels=["<-25%", "-25% to 0%", "0% to 25%", "25% to 100%", ">=100%"],
        right=False,
    )
    for model in MODEL_NAMES:
        pairs[f"{model}_error_pp"] = (pairs[model] - pairs.target).abs() * 100
    pairs["asinh_minus_raw_error_pp"] = pairs[f"{ASINH}_error_pp"] - pairs[f"{RAW}_error_pp"]
    if any(file_sha256(path) != hashes[name] for name, path in inputs.items()):
        raise ValueError("Input changed during evaluation")
    outputs = {
        "overall-metrics.csv": summarize(pairs.assign(scope="ALL_SERVICE3"), ["phase", "scope"]),
        "industry-metrics.csv": summarize(pairs, ["phase", "industry_code"]),
        "district-metrics.csv": summarize(pairs, ["phase", "district_name_2025q2"]),
        "district-industry-metrics.csv": summarize(
            pairs, ["phase", "district_name_2025q2", "industry_code"]
        ),
        "sales-band-metrics.csv": summarize(pairs, ["phase", "comparison_sales_band"]),
        "actual-growth-metrics.csv": summarize(pairs, ["phase", "actual_growth_band"]),
        "tail-transitions.csv": fixed_p90_transitions(pairs.loc[pairs.phase == "reporting"]),
        "private-paired-errors.csv": pairs,
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, table in outputs.items():
        table.to_csv(output / name, index=False)
    summary = {
        "artifact_kind": "seoul_next_quarter_tail_research_v1",
        "as_of": as_of.isoformat(),
        "selected_by_earlier_folds": selected,
        "selection_rule": "min MAE with P90 and count(error > 100pp) no higher than raw Huber",
        "region_labels_reference_date": REGION_REFERENCE_DATE.isoformat(),
        "source_sha256": hashes,
        "code_sha256": {
            "runner": file_sha256(Path(__file__)),
            "rolling": file_sha256(Path(rolling.__file__)),
            "near_term": file_sha256(Path(near.__file__)),
            "features": file_sha256(Path(near.feature_record.__code__.co_filename)),
        },
        "output_sha256": {name: file_sha256(output / name) for name in outputs},
        "versions": package_versions(),
        "folds": folds,
        "limits": [
            "Revised-snapshot research. Reporting and additional targets were previously exposed.",
            "The tail-only and sales-band candidates follow inspection of reporting residuals.",
            "One-quarter label embargo is an assumption; historical availability is unverified.",
            "Region labels apply to the reference date; no unmatched region is inferred.",
            "Actual-growth strata are post-hoc diagnostics; they are not model inputs.",
            "Keep paired data-derived values under the ignored report path.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--name-reference", type=Path, required=True)
    parser.add_argument("--region-mapping", type=Path, required=True)
    parser.add_argument("--sgis-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    result = run(
        args.input,
        args.name_reference,
        args.region_mapping,
        args.sgis_reference,
        args.output,
        args.as_of,
    )
    print(json.dumps({"output": str(args.output), "selected": result["selected_by_earlier_folds"]}))


if __name__ == "__main__":
    main()
