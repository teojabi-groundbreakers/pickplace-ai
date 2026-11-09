"""Compare optional Seoul store-history features without changing prediction scope.

Only the already-inspected 2025Q2 training and 2025Q4 validation are used.
This is an offline retrospective research comparison, not an inference path.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import pickplace_ai.features as common_features
import pickplace_ai.unified_features as unified_features
from pickplace_ai.artifacts import package_versions
from scripts import train_gyeonggi_research as base
from scripts import train_unified_growth as unified

PERIODS = {"train": "2025Q2", "validation": "2025Q4"}
STORE_FIELDS = ("store_count", "opening_count", "closing_count")
STORE_FEATURES = (
    "store_current",
    "store_previous",
    "store_change",
    "opening_share",
    "closing_share",
)
OPTIONS = {
    "baseline": (),
    "stores_current_previous": STORE_FEATURES[:2],
    "stores_and_turnover": STORE_FEATURES,
}


def store_lookup(rows: pd.DataFrame) -> pd.DataFrame:
    """Build a distinct quarterly Seoul lookup, preserving missing values."""
    keys = ["region_code", "region_code_system", "industry_code", "period"]
    if not set(keys + list(STORE_FIELDS)).issubset(rows.columns):
        raise ValueError("Missing store source columns")
    data = rows[keys + list(STORE_FIELDS)].copy()
    if data.empty or data[keys].isna().any().any():
        raise ValueError("Missing store source key")
    if not data.region_code_system.eq("seoul-adstrd").all():
        raise ValueError("Unexpected Seoul code system")
    if data.duplicated(["region_code", "industry_code", "period"]).any():
        raise ValueError("Duplicate store source quarter")
    if not data.period.astype(str).str.fullmatch(r"\d{4}Q[1-4]").all():
        raise ValueError("Invalid store source period")
    data["period"] = pd.PeriodIndex(data.period, freq="Q")
    for name in STORE_FIELDS:
        data[name] = pd.to_numeric(data[name], errors="raise")
        valid = data[name].dropna().to_numpy(dtype=float)
        if not np.isfinite(valid).all() or (valid < 0).any():
            raise ValueError("Invalid store source amount")
    return data.set_index(["region_code", "industry_code", "period"])[list(STORE_FIELDS)]


def add_store_features(frame: pd.DataFrame, lookup: pd.DataFrame, target: str) -> pd.DataFrame:
    """Attach Seoul t and t-1 stores only; never inspect target-quarter t+1 stores."""
    if not set(base.SCOPE + ["source"]).issubset(frame.columns):
        raise ValueError("Missing frame scope")
    if not set(frame.source).issubset({"seoul", "gyeonggi"}):
        raise ValueError("Unexpected frame source")
    if frame[base.SCOPE + ["source"]].isna().any().any():
        raise ValueError("Missing frame scope value")
    target_period = pd.Period(target, freq="Q")
    out = frame.copy()
    for name in STORE_FEATURES:
        out[name] = np.nan
    positions = np.flatnonzero(out.source.to_numpy() == "seoul")
    if not len(positions):
        return out
    seoul = out.iloc[positions]
    keys = list(seoul[base.SCOPE].itertuples(index=False, name=None))
    current_index = pd.MultiIndex.from_tuples(
        [(region, industry, target_period - 1) for region, industry in keys],
        names=lookup.index.names,
    )
    previous_index = pd.MultiIndex.from_tuples(
        [(region, industry, target_period - 2) for region, industry in keys],
        names=lookup.index.names,
    )
    current = lookup.reindex(current_index)
    previous = lookup.reindex(previous_index)
    count = current.store_count.to_numpy(dtype=float)
    prior = previous.store_count.to_numpy(dtype=float)
    computed = {"store_current": count, "store_previous": prior}
    ratio = np.full(len(seoul), np.nan)
    np.divide(count, prior, out=ratio, where=np.isfinite(prior) & (prior > 0))
    computed["store_change"] = ratio - 1
    for name, source in (("opening_share", "opening_count"), ("closing_share", "closing_count")):
        ratio = np.full(len(seoul), np.nan)
        np.divide(
            current[source].to_numpy(dtype=float),
            count,
            out=ratio,
            where=np.isfinite(count) & (count > 0),
        )
        computed[name] = ratio
    for name, values in computed.items():
        out.iloc[positions, out.columns.get_loc(name)] = values
    if np.isinf(out[list(STORE_FEATURES)].to_numpy(dtype=float)).any():
        raise ValueError("Infinite store features")
    return out


def _predict(model, frame: pd.DataFrame) -> np.ndarray:
    result = np.asarray(model.predict(frame), dtype=float)
    if result.ndim != 1 or len(result) != len(frame) or not np.isfinite(result).all():
        raise ValueError("Invalid predictions")
    return np.maximum(result, -1.0)


def run(seoul: Path, gyeonggi: Path, model_root: Path, output: Path) -> dict:
    """Compare validation predictions with identical targets, scopes and weights."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output already exists")
    meta_path = model_root / "metadata.json"
    paths = {"seoul": seoul, "gyeonggi": gyeonggi, "reference_metadata": meta_path}
    fingerprints = {name: base.sha(path) for name, path in paths.items()}
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_kind") != unified.KIND or metadata.get("purpose") != (
        "retrospective_research_only_not_service"
    ):
        raise ValueError("Unexpected pooled reference artifact")
    for key in ("seoul", "gyeonggi"):
        if metadata.get("source_hashes", {}).get(key) != fingerprints[key]:
            raise ValueError(f"{key} input differs from the reference artifact")
    code_paths = {
        "trainer": Path(unified.__file__),
        "feature_code": Path(common_features.__file__),
        "unified_feature_code": Path(unified_features.__file__),
        "base_trainer": Path(base.__file__),
        "store_study": Path(__file__),
    }
    code_hashes = {name: base.sha(path) for name, path in code_paths.items()}
    for name in ("trainer", "feature_code", "unified_feature_code", "base_trainer"):
        if metadata.get("code_sha256", {}).get(name) != code_hashes[name]:
            raise ValueError(f"Reference code mismatch: {name}")
    sq = unified.seoul_quarters(seoul)
    gg_monthly = pd.read_csv(
        gyeonggi, dtype={"기준년월": str, "행정동코드": str, "중분류업종코드": str}
    )
    gq = base.make_quarters(gg_monthly)
    rows = pd.read_csv(
        seoul,
        dtype={"region_code": str, "industry_code": str, "period": str},
        usecols=["region_code", "region_code_system", "industry_code", "period", *STORE_FIELDS],
    )
    lookup = store_lookup(rows)
    frames = {}
    for phase, period in PERIODS.items():
        frames[phase] = {
            source: add_store_features(unified.make_frame(quarters, period, source), lookup, period)
            for source, quarters in (("seoul", sq), ("gyeonggi", gq))
        }
    train = pd.concat(list(frames["train"].values()), ignore_index=True)
    counts = {source: len(frame) for source, frame in frames["train"].items()}
    if not counts["seoul"] or not counts["gyeonggi"]:
        raise ValueError("Training source cannot be empty")
    weights = np.array([1.0 / counts[source] for source in train.source])
    weights *= len(weights) / weights.sum()
    main = list(unified_features.UNIFIED_RESEARCH_FEATURES)
    results = {}
    for name, extra in OPTIONS.items():
        columns = main + list(extra)
        model = unified.model().fit(train[columns], train.target, sample_weight=weights)
        results[name] = {}
        for source, frame in frames["validation"].items():
            result = unified.metric(frame.target.to_numpy(), _predict(model, frame[columns]))
            result["industries"] = int(frame["중분류업종코드"].nunique())
            results[name][source] = result
    missing = {
        phase: {
            source: {name: int(frame[name].isna().sum()) for name in STORE_FEATURES}
            for source, frame in partition.items()
        }
        for phase, partition in frames.items()
    }
    summary = {
        "purpose": "exploratory_retrospective_store_feature_comparison_not_service",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reference_artifact_kind": unified.KIND,
        "source_hashes": fingerprints,
        "code_sha256": code_hashes,
        "versions": package_versions(),
        "target": "quarter_sales[t+1]/quarter_sales[t-3]-1",
        "periods": PERIODS,
        "embargo_target_period": "2025Q3",
        "test_2026Q2_used_for_model_training_or_evaluation": False,
        "sample_counts": {
            phase: {source: len(frame) for source, frame in partition.items()}
            for phase, partition in frames.items()
        },
        "store_missing_counts": missing,
        "features": {name: list(cols) for name, cols in OPTIONS.items()},
        "store_feature_periods": "target-1 and target-2 only",
        "source_weights": "Equal total model training weight for Seoul and Gyeonggi",
        "results": results,
        "limitations": [
            "2025Q4 was already examined and is an exploratory validation, not a holdout.",
            "Revised snapshot and unknown publication dates prevent real-time backtesting.",
            "Gyeonggi store inputs are all missing; no zero or imputed values are fabricated.",
            "Store signals reflect aggregate observations and do not identify individual stores.",
            "Seoul source labels sales as 당월; the temporal aggregation is unverified.",
            "Future-quarter store inputs are not used; unsupported code mappings are not inferred.",
            "This study trains evaluation models only; no service or trusted model is replaced.",
        ],
    }
    if fingerprints != {name: base.sha(path) for name, path in paths.items()}:
        raise ValueError("Source changed while running")
    if code_hashes != {name: base.sha(path) for name, path in code_paths.items()}:
        raise ValueError("Code changed while running")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seoul-store-study-", dir=output.parent) as temp:
        stage = Path(temp) / "output"
        stage.mkdir()
        summary_path = stage / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (stage / "manifest.json").write_text(
            json.dumps({"summary_sha256": base.sha(summary_path)}, indent=2) + "\n",
            encoding="utf-8",
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output already exists")
        os.rename(stage, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seoul", type=Path, required=True)
    parser.add_argument("--gyeonggi", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seoul, args.gyeonggi, args.model_root, args.output)
    print(
        json.dumps({"periods": result["periods"], "results": result["results"]}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
