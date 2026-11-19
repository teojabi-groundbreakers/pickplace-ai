"""Pooled Seoul and Gyeonggi quarterly growth on current snapshots; research only."""

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

import pickplace_ai.features as features_module
import pickplace_ai.unified_features as unified_features_module
from pickplace_ai.artifacts import package_versions
from pickplace_ai.features import GYEONGGI_RESEARCH_FEATURES
from pickplace_ai.unified_features import UNIFIED_RESEARCH_FEATURES, unified_research_feature_frame
from scripts import train_gyeonggi_research as base

KIND = "seoul_gyeonggi_growth_pooled_research_v1"
PERIODS = {"train": "2025Q2", "validation": "2025Q4", "reused_test": "2026Q2"}
FINAL_PERIODS = ("2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2")
SOURCE_CODE_SYSTEMS = {"seoul": "seoul-adstrd", "gyeonggi": "gyeonggi-admdong-10-unverified"}


def acquisition_day(value: str) -> date:
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Acquisition timestamps must have a timezone")
    return timestamp.astimezone(ZoneInfo("Asia/Seoul")).date()


def seoul_quarters(path: Path) -> pd.DataFrame:
    names = [
        "region_code",
        "region_code_system",
        "industry_code",
        "period",
        "sales_unit",
        "sales_amount",
    ]
    rows = pd.read_csv(
        path, dtype={"region_code": str, "industry_code": str, "period": str}, usecols=names
    )
    if rows.empty or rows[names].isna().any().any():
        raise ValueError("Missing Seoul observation")
    if not rows.region_code_system.eq("seoul-adstrd").all() or not rows.sales_unit.eq("KRW").all():
        raise ValueError("Unexpected Seoul source or unit")
    if rows.duplicated(["region_code", "industry_code", "period"]).any():
        raise ValueError("Duplicate Seoul quarterly key")
    if not rows.period.str.fullmatch(r"\d{4}Q[1-4]").all():
        raise ValueError("Invalid Seoul period")
    rows.sales_amount = pd.to_numeric(rows.sales_amount, errors="raise")
    if not np.isfinite(rows.sales_amount).all() or (rows.sales_amount < 0).any():
        raise ValueError("Invalid Seoul sales")
    rows = rows.rename(columns={"region_code": "행정동코드", "industry_code": "중분류업종코드"})
    rows["period"] = pd.PeriodIndex(rows.period, freq="Q")
    return rows.pivot(index=base.SCOPE, columns="period", values="sales_amount")


def make_frame(quarters: pd.DataFrame, period: str, source: str) -> pd.DataFrame:
    frame = base.make_frame(quarters, period)
    features = unified_research_feature_frame(frame[list(GYEONGGI_RESEARCH_FEATURES)], source)
    frame["source_gyeonggi"] = features["source_gyeonggi"]
    frame["source"] = source
    return frame


def predict_unified_model(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    if bundle.get("artifact_kind") != KIND:
        raise ValueError("Unexpected research model kind")
    if list(bundle.get("feature_columns", ())) != list(UNIFIED_RESEARCH_FEATURES):
        raise ValueError("Unexpected model feature schema")
    if list(features.columns) != list(UNIFIED_RESEARCH_FEATURES):
        raise ValueError("Features must use expected order")
    if features.empty or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Features must be nonempty and finite")
    predicted = np.asarray(bundle["model"].predict(features), dtype=float)
    if predicted.ndim != 1 or len(predicted) != len(features) or not np.isfinite(predicted).all():
        raise ValueError("Invalid model prediction")
    return np.maximum(predicted, -1.0)


def model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.05,
        l2_regularization=1,
        early_stopping=False,
        random_state=42,
    )


def bundle(model_instance) -> dict:
    return {
        "artifact_kind": KIND,
        "model": model_instance,
        "feature_columns": list(UNIFIED_RESEARCH_FEATURES),
    }


def metric(y: np.ndarray, prediction: np.ndarray) -> dict:
    return base.metrics(y, prediction)


def train(
    seoul: Path,
    seoul_manifest: Path,
    gyeonggi: Path,
    gyeonggi_manifest: Path,
    output: Path,
    snapshot: date,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Never overwrite a research artifact")
    sources = {
        "seoul": seoul,
        "seoul_manifest": seoul_manifest,
        "gyeonggi": gyeonggi,
        "gyeonggi_manifest": gyeonggi_manifest,
    }
    hashes = {k: base.sha(p) for k, p in sources.items()}
    sm = json.loads(seoul_manifest.read_text())
    gm = json.loads(gyeonggi_manifest.read_text())
    if sm.get("output_sha256") != hashes["seoul"]:
        raise ValueError("Seoul source checksum mismatch")
    if (
        gm.get("purpose") != "review_only_not_training_or_prediction"
        or gm.get("candidate_sha256") != hashes["gyeonggi"]
    ):
        raise ValueError("Gyeonggi source checksum mismatch")
    try:
        api_sources = sm["api_sources"]
        if not api_sources:
            raise ValueError("Empty Seoul API provenance")
        seoul_date = max(
            date.fromisoformat(sm["historical_available_at"]),
            *(acquisition_day(item["acquired_at_utc"]) for item in api_sources),
        )
        gyeonggi_date = acquisition_day(gm["generated_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Missing or malformed acquisition provenance") from exc
    source_dates = {"seoul": seoul_date.isoformat(), "gyeonggi": gyeonggi_date.isoformat()}
    if snapshot < max(seoul_date, gyeonggi_date):
        raise ValueError("Snapshot precedes latest acquisition or review")
    if snapshot > datetime.now(ZoneInfo("Asia/Seoul")).date():
        raise ValueError("Snapshot cannot be in the future")
    g = pd.read_csv(gyeonggi, dtype={"기준년월": str, "행정동코드": str, "중분류업종코드": str})
    quarters = {"seoul": seoul_quarters(seoul), "gyeonggi": base.make_quarters(g)}
    frames = {
        period: {source: make_frame(q, period, source) for source, q in quarters.items()}
        for period in FINAL_PERIODS
    }
    columns = list(UNIFIED_RESEARCH_FEATURES)
    fit = pd.concat(list(frames[PERIODS["train"]].values()), ignore_index=True)
    weights = np.array([1.0 / len(frames[PERIODS["train"]][source]) for source in fit.source])
    weights *= len(weights) / weights.sum()
    pooled = model().fit(fit[columns], fit.target, sample_weight=weights)
    source_models = {
        source: model().fit(f[columns], f.target) for source, f in frames[PERIODS["train"]].items()
    }
    results = {}
    for phase, period in PERIODS.items():
        if phase == "train":
            continue
        by_source = {}
        for source, f in frames[period].items():
            target = f.target.to_numpy()
            pooled_pred = predict_unified_model(bundle(pooled), f[columns])
            source_pred = predict_unified_model(bundle(source_models[source]), f[columns])
            by_source[source] = {
                "pooled": metric(target, pooled_pred),
                "source_only": metric(target, source_pred),
                "zero_growth": metric(target, np.zeros(len(f))),
                "scope_count": len(f),
                "industries": int(f["중분류업종코드"].nunique()),
            }
        results[phase] = by_source
    final = pd.concat(
        [frame for p in FINAL_PERIODS for frame in frames[p].values()], ignore_index=True
    )
    totals = final.source.value_counts().to_dict()
    weights = np.array([1.0 / totals[source] for source in final.source])
    weights *= len(weights) / weights.sum()
    final_model = model().fit(final[columns], final.target, sample_weight=weights)
    scopes = {
        source: sorted(
            {
                (str(r), str(i))
                for p in FINAL_PERIODS
                for r, i in frames[p][source][base.SCOPE].itertuples(index=False, name=None)
            }
        )
        for source in quarters
    }
    manifest = {
        "artifact_kind": KIND,
        "purpose": "retrospective_research_only_not_service",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_review_date": snapshot.isoformat(),
        "source_known_by": source_dates,
        "source_date_basis": {
            "seoul": "latest historical or API acquisition date, Asia/Seoul",
            "gyeonggi": "candidate review generation date; prior source acquisition time unknown",
        },
        "source_code_systems": SOURCE_CODE_SYSTEMS,
        "source_scopes": scopes,
        "source_hashes": hashes,
        "code_sha256": {
            "trainer": base.sha(Path(__file__)),
            "feature_code": base.sha(Path(features_module.__file__)),
            "unified_feature_code": base.sha(Path(unified_features_module.__file__)),
            "base_trainer": base.sha(Path(base.__file__)),
        },
        "features": columns,
        "target": "quarter_sales[t+1]/quarter_sales[t-3]-1",
        "prediction_method": "scripts.train_unified_growth.predict_unified_model",
        "prediction_floor": -1.0,
        "base_period": "2026Q2",
        "target_period": "2026Q3",
        "periods": PERIODS,
        "embargo_periods": ["2025Q3", "2026Q1"],
        "test_reused": True,
        "final_training_target_periods": FINAL_PERIODS,
        "final_training_rows": len(final),
        "final_rows_by_source": totals,
        "evaluation": results,
        "versions": package_versions(),
        "limitations": [
            "Seoul estimated sales KRW and Gyeonggi card sales unknown unit "
            "are distinct targets and not equated.",
            "Gyeonggi monthly summation, source definitions, H10 and historical "
            "boundaries are unverified.",
            "No inferred cross-region industry or geographical mappings.",
            "Current snapshots have unknown historical publication dates; "
            "not a point-in-time backtest.",
            "Previously inspected 2026Q2 labels are reused, not an independent test.",
            "Final model is refit with 2026Q2 labels; metrics do not independently measure it.",
            "Not production approved, not probability or commercial success.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".unified-research-", dir=output.parent) as temp:
        path = Path(temp) / "out"
        path.mkdir()
        joblib.dump(bundle(final_model), path / "final-model.joblib")
        joblib.dump(bundle(pooled), path / "evaluation-pooled.joblib")
        manifest["model_sha256"] = {p.name: base.sha(p) for p in path.iterdir()}
        (path / "metadata.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if hashes != {k: base.sha(p) for k, p in sources.items()}:
            raise ValueError("Inputs changed while training")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Never overwrite a research artifact")
        os.rename(path, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seoul", type=Path, required=True)
    parser.add_argument("--seoul-manifest", type=Path, required=True)
    parser.add_argument("--gyeonggi", type=Path, required=True)
    parser.add_argument("--gyeonggi-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    result = train(
        args.seoul,
        args.seoul_manifest,
        args.gyeonggi,
        args.gyeonggi_manifest,
        args.output,
        args.snapshot,
    )
    print(
        json.dumps(
            {k: result[k] for k in ("final_rows_by_source", "evaluation")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
