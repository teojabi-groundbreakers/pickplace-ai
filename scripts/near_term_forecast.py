"""Research-only direct 1–2 quarter forecasts using the latest acquired snapshot.

Past observations retain their actual acquisition dates. All backtests are
retrospective simulations on revised data, not historical point-in-time tests.
"""

import argparse
import inspect
import json
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.data import read_observations, validate_observations
from pickplace_ai.features import FEATURE_COLUMNS, FeatureUnavailable, feature_record
from pickplace_ai.forecast_2026 import (
    LOG_FIELDS,
    metrics,
    year_rows,
)
from pickplace_ai.forecast_2026 import (
    fit_and_evaluate as old_evaluate,
)
from pickplace_ai.ingest import import_seoul
from pickplace_ai.periods import quarter_end, shift_quarter
from pickplace_ai.training import RecentGrowthRegressor
from scripts.retrospective_all_industries import _source_manifest

FEATURES = FEATURE_COLUMNS + ["comparison_sales", "target_quarter", "industry_code"]
NUMERIC = [field for field in FEATURES if field != "industry_code"]
KIND = "seoul_near_term_snapshot_forecast_v1"


def build_frames(observations, as_of, origins):
    """Construct both horizons independently, using no inputs after each origin."""
    validate_observations(observations)
    groups = defaultdict(dict)
    for obs in observations:
        if obs.available_at > as_of or quarter_end(obs.period) > as_of:
            raise ValueError("Observation is unavailable or its period is unfinished")
        if obs.region_code_system != "seoul-adstrd" or not obs.region_code.startswith("11"):
            raise ValueError("Expected Seoul administrative-dong data")
        groups[(obs.region_code_system, obs.region_code, obs.industry_code)][obs.period] = obs
    rows, omitted = [], Counter()
    for (system, region, industry), indexed in groups.items():
        for origin in origins:
            if origin not in indexed:
                continue
            history = [
                indexed[p] for p in (shift_quarter(origin, i) for i in range(-4, 1)) if p in indexed
            ]
            try:
                features = feature_record(history, origin, as_of)
            except FeatureUnavailable as exc:
                omitted[f"{origin}:{exc.reason}"] += 2
                continue
            for horizon in (1, 2):
                target = shift_quarter(origin, horizon)
                comparison = indexed.get(shift_quarter(target, -4))
                if (
                    comparison is None
                    or comparison.sales_amount is None
                    or comparison.sales_amount <= 0
                ):
                    omitted[f"{origin}:missing_or_nonpositive_comparison"] += 1
                    continue
                target_obs = indexed.get(target)
                y = (
                    None
                    if target_obs is None or target_obs.sales_amount is None
                    else (target_obs.sales_amount / comparison.sales_amount - 1)
                )
                if y is not None and not np.isfinite(y):
                    raise ValueError("Invalid target")
                rows.append(
                    {
                        **features,
                        "region_code_system": system,
                        "region_code": region,
                        "industry_code": industry,
                        "base_period": origin,
                        "horizon": horizon,
                        "comparison_period": comparison.period,
                        "comparison_sales": comparison.sales_amount,
                        "target_quarter": int(target[-1]),
                        "target_period": target,
                        "target": y,
                    }
                )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("No eligible forecast rows")
    return frame, dict(omitted)


def matrix(frame):
    result = frame[FEATURES].copy()
    for field in LOG_FIELDS:
        if (result[field].dropna() < 0).any():
            raise ValueError("Negative sales input")
        result[field] = np.log1p(result[field])
    for field in ("sales_yoy", "sales_qoq"):
        result[field] = np.sign(result[field]) * np.log1p(np.abs(result[field]))
    if np.isinf(result[NUMERIC].to_numpy(float)).any():
        raise ValueError("Infinite input")
    return result


def candidates():
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            ("industry", OneHotEncoder(handle_unknown="ignore"), ["industry_code"]),
        ]
    )
    return {
        "zero_growth": DummyRegressor(strategy="constant", constant=0),
        "recent_growth": RecentGrowthRegressor(),
        "huber_sgd": Pipeline(
            [
                ("features", clone(preprocessing)),
                (
                    "model",
                    SGDRegressor(
                        loss="huber",
                        epsilon=1,
                        alpha=0.01,
                        max_iter=1000,
                        tol=1e-4,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def predict(model, frame):
    # RecentGrowthRegressor is a baseline in raw ratio units, not log units.
    inputs = frame[FEATURES] if isinstance(model, RecentGrowthRegressor) else matrix(frame)
    values = np.asarray(model.predict(inputs), dtype=float)
    if len(values) != len(frame) or not np.isfinite(values).all():
        raise ValueError("Invalid prediction")
    return np.maximum(values, -1)


def scope_training_flags(training, forecast):
    """Mark scope membership without claiming verified geographic generalization."""
    columns = ["region_code_system", "region_code", "industry_code"]
    seen = set(training[columns].itertuples(index=False, name=None))
    return [row in seen for row in forecast[columns].itertuples(index=False, name=None)]


def select_and_test(frame, horizon):
    h = frame.loc[frame.horizon == horizon]
    val = h.loc[(h.base_period == "2024Q2") & h.target.notna()].copy()
    test = h.loc[(h.base_period == "2025Q2") & h.target.notna()].copy()
    train_val = h.loc[(h.target_period <= "2024Q2") & h.target.notna()].copy()
    train_test = h.loc[(h.target_period <= "2025Q2") & h.target.notna()].copy()
    if any(part.empty for part in (val, test, train_val, train_test)):
        raise ValueError("Missing time partition")
    vm, tm, test_pred = {}, {}, {}
    for name, model in candidates().items():
        model.fit(matrix(train_val), train_val.target)
        vm[name] = metrics(val, predict(model, val))
    winner = min(vm, key=lambda name: vm[name]["mae_ratio"])
    for name, model in candidates().items():
        model.fit(matrix(train_test), train_test.target)
        test_pred[name] = predict(model, test)
        tm[name] = metrics(test, test_pred[name])
    return winner, vm, tm, test.reset_index(drop=True), test_pred


def old_test_predictions(historical, as_of):
    train, _ = year_rows(historical, 2022, as_of, labels=True)
    val, _ = year_rows(historical, 2023, as_of, labels=True)
    test, _ = year_rows(historical, 2024, as_of, labels=True)
    winner, _, _, _, preds = old_evaluate(train, val, test)
    test = test.copy()
    test["old_prediction"] = preds[winner]
    return winner, test


def run(historical_path, raw_root, old_manifest, output, as_of):
    historical_path, raw_root, old_manifest, output = map(
        Path, (historical_path, raw_root, old_manifest, output)
    )
    if output.exists():
        raise FileExistsError("Output already exists")
    old_date = date(2026, 11, 3)
    old_bytes = old_manifest.read_bytes()
    _source_manifest(old_bytes, raw_root, old_date)
    hist_hash = file_sha256(historical_path)
    old_summary_path = old_manifest.parent / "summary.json"
    old_summary_hash = file_sha256(old_summary_path)
    old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
    if old_summary.get("input_sha256") != hist_hash or old_summary.get(
        "raw_source_manifest_sha256"
    ) != file_sha256(old_manifest):
        raise ValueError("Historical processed data does not match the source experiment")
    historical = read_observations(historical_path)
    if any(
        row.available_at != old_date or row.source_kind != "official" or row.period > "2025Q4"
        for row in historical
    ):
        raise ValueError("Unexpected historical snapshot")
    source_info, new_rows = [], []
    for q in (1, 2):
        files = {
            kind: raw_root / kind / f"seoul-api-{kind}-2026Q{q}.csv" for kind in ("sales", "stores")
        }
        acquired = []
        for kind, path in files.items():
            meta_path = path.with_suffix(".metadata.json")
            meta_hash = file_sha256(meta_path)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                meta["kind"] != kind
                or meta["period"] != f"2026Q{q}"
                or file_sha256(path) != meta["sha256"]
            ):
                raise ValueError("API input does not match metadata")
            acquired.append(
                datetime.fromisoformat(meta["acquired_at_utc"])
                .astimezone(ZoneInfo("Asia/Seoul"))
                .date()
            )
            source_info.append(
                {
                    "kind": kind,
                    "period": meta["period"],
                    "sha256": meta["sha256"],
                    "metadata_sha256": meta_hash,
                    "acquired_at_utc": meta["acquired_at_utc"],
                    "catalog": meta["catalog"],
                }
            )
        available = max(acquired)
        if available > as_of:
            raise ValueError("API data not yet acquired")
        quarter_rows = import_seoul([files["sales"]], [files["stores"]], available)
        if any(row.period != f"2026Q{q}" for row in quarter_rows):
            raise ValueError("Unexpected quarter in API data")
        new_rows.extend(quarter_rows)
    observations = historical + new_rows
    origins = [shift_quarter("2022Q1", i) for i in range(18)]  # 2022Q1 through 2026Q2
    frame, omitted = build_frames(observations, as_of, origins)
    all_scopes = {(r.region_code_system, r.region_code, r.industry_code) for r in observations}
    current_scopes = {
        (r.region_code_system, r.region_code, r.industry_code)
        for r in observations
        if r.period == "2026Q2"
    }
    omitted["2026Q2:scope_absent_from_base_quarter"] = len(all_scopes - current_scopes)
    old_winner, old_test = old_test_predictions(historical, old_date)
    all_results, outputs, bundles = {}, [], {}
    for horizon in (1, 2):
        winner, val_metrics, test_metrics, test, predictions = select_and_test(frame, horizon)
        selected = test.copy()
        selected["new_prediction"] = predictions[winner]
        keys = ["region_code_system", "region_code", "industry_code", "target_period"]
        common = selected.merge(
            old_test[keys + ["old_prediction", "target"]],
            on=keys,
            suffixes=("", "_old"),
            validate="one_to_one",
        )
        if common.empty or not np.allclose(common.target, common.target_old):
            raise ValueError("No comparable old and new historical test samples")
        fit = frame.loc[
            (frame.horizon == horizon) & (frame.target_period <= "2026Q2") & frame.target.notna()
        ].copy()
        out = frame.loc[(frame.horizon == horizon) & (frame.base_period == "2026Q2")].copy()
        known = set(fit.industry_code)
        omitted[f"2026Q2:h{horizon}:industry_absent_from_training"] = int(
            (~out.industry_code.isin(known)).sum()
        )
        out = out.loc[out.industry_code.isin(known)].copy()
        if out.empty:
            raise ValueError("No forecast output")
        out["scope_seen_in_training"] = scope_training_flags(fit, out)
        model = candidates()[winner]
        model.fit(matrix(fit), fit.target)
        out["estimated_yoy_growth"] = predict(model, out)
        out["estimated_sales_krw"] = out.comparison_sales * (1 + out.estimated_yoy_growth)
        test_counts = test.groupby("industry_code").size().to_dict()
        test_by_industry = {}
        for industry, indexes in test.groupby("industry_code").groups.items():
            positions = indexes.to_numpy()
            subset = test.loc[indexes]
            test_by_industry[industry] = {
                "selected": metrics(subset, predictions[winner][positions]),
                "warnings": ["fewer_than_30_test_rows"] if len(subset) < 30 else [],
            }
        out["industry_test_rows"] = out.industry_code.map(test_counts).fillna(0).astype(int)
        out["warning"] = out.apply(
            lambda r: ";".join(
                ["research_only_not_point_in_time_validated"]
                + ([] if r.scope_seen_in_training else ["scope_absent_from_training"])
                + (["low_industry_test_sample"] if r.industry_test_rows < 30 else [])
                + (["comparison_sales_below_1m_krw"] if r.comparison_sales < 1_000_000 else [])
            ),
            axis=1,
        )
        outputs.append(out)
        bundles[str(horizon)] = {"model_name": winner, "model": model, "features": FEATURES}
        all_results[str(horizon)] = {
            "target_period": shift_quarter("2026Q2", horizon),
            "selection": winner,
            "validation": val_metrics,
            "test": test_metrics,
            "test_by_industry": test_by_industry,
            "common_2025_test": {
                "rows": len(common),
                "old_model": old_winner,
                "old": metrics(common, common.old_prediction),
                "new": metrics(common, common.new_prediction),
            },
            "forecast_rows": len(out),
            "forecast_industries": len(set(out.industry_code)),
            "forecast_absent_industries": sorted(
                {scope[2] for scope in current_scopes} - set(out.industry_code)
            ),
            "training_rows_final": len(fit),
            "forecast_scopes_absent_from_training": int((~out.scope_seen_in_training).sum()),
        }
    result = pd.concat(outputs, ignore_index=True)
    if file_sha256(historical_path) != hist_hash:
        raise ValueError("Historical input changed during training")
    for item in source_info:
        path = raw_root / item["kind"] / f"seoul-api-{item['kind']}-{item['period']}.csv"
        if (
            file_sha256(path) != item["sha256"]
            or file_sha256(path.with_suffix(".metadata.json")) != item["metadata_sha256"]
        ):
            raise ValueError("API input changed during training")
    if file_sha256(old_summary_path) != old_summary_hash or old_manifest.read_bytes() != old_bytes:
        raise ValueError("Historical manifest or summary changed during training")
    _source_manifest(old_bytes, raw_root, old_date)
    result["estimate_kind"] = result.target_period.map(
        lambda p: (
            "completed_quarter_unobserved_estimate"
            if quarter_end(p) <= as_of
            else "in_progress_or_future_quarter_estimate"
        )
    )
    result["as_of"] = as_of.isoformat()
    result = result[
        [
            "region_code_system",
            "region_code",
            "industry_code",
            "base_period",
            "target_period",
            "horizon",
            "comparison_period",
            "comparison_sales",
            "estimated_yoy_growth",
            "estimated_sales_krw",
            "scope_seen_in_training",
            "industry_test_rows",
            "warning",
            "estimate_kind",
            "as_of",
        ]
    ]
    output.mkdir(parents=True, exist_ok=False)
    result.to_csv(output / "forecast-2026-q3-q4.csv", index=False)
    joblib.dump({"artifact_kind": KIND, "models": bundles}, output / "model.joblib")
    summary = {
        "artifact_kind": KIND,
        "release_status": "experimental_research_only",
        "created_at": datetime.now(UTC).isoformat(),
        "as_of": as_of.isoformat(),
        "historical_acquired_at": old_date.isoformat(),
        "historical_input_sha256": hist_hash,
        "historical_manifest_sha256": file_sha256(old_manifest),
        "historical_summary_sha256": old_summary_hash,
        "new_sources": source_info,
        "model_sha256": file_sha256(output / "model.joblib"),
        "forecast_sha256": file_sha256(output / "forecast-2026-q3-q4.csv"),
        "runner_sha256": file_sha256(Path(__file__)),
        "feature_code_sha256": file_sha256(Path(inspect.getfile(feature_record))),
        "dependency_code_sha256": {
            "forecast_2026.py": file_sha256(Path(inspect.getfile(year_rows))),
            "training.py": file_sha256(Path(inspect.getfile(RecentGrowthRegressor))),
            "ingest.py": file_sha256(Path(inspect.getfile(import_seoul))),
            "data.py": file_sha256(Path(inspect.getfile(read_observations))),
            "periods.py": file_sha256(Path(inspect.getfile(quarter_end))),
            "retrospective_all_industries.py": file_sha256(Path(inspect.getfile(_source_manifest))),
        },
        "versions": package_versions(),
        "origin": "2026Q2",
        "results": all_results,
        "omitted": omitted,
        "selection_rule": (
            "each horizon: minimum 2024 Q2-origin validation MAE; 2025 Q2-origin test held out"
        ),
        "prediction_method": (
            "Trusted artifacts only: use build_frames on observations with their "
            "actual available_at, "
            "then predict(bundle['models'][horizon]['model'], frame). The helper applies the "
            "required transform and -100% floor; do not call model.predict on raw values."
        ),
        "limits": [
            "Revised snapshots acquired in Nov 2026: historical release dates and vintages "
            "unverified; not point-in-time backtests.",
            "Quarter status is computed from as_of. Completed quarters with missing actuals "
            "are retrospectively estimated, not forecasts made before those quarters.",
            "2026Q3/Q4 actuals are unavailable; no forward accuracy or interval is established.",
            "Small denominators and extreme targets are not removed. "
            "Predictions are floored at -100%.",
            "No population fields are present; selected models are pooled across industries.",
            "Only known complete-history combinations are forecast; missing results are not zero.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    summary = run(args.historical, args.raw_root, args.old_manifest, args.output, args.as_of)
    print(
        json.dumps({"output": str(args.output), "results": summary["results"]}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
