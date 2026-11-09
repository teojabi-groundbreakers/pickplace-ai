"""Reproduce the Seoul error breakdown of a trusted pooled evaluation model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pickplace_ai import features, unified_features
from scripts import train_gyeonggi_research, train_unified_growth

PERIODS = {"validation": "2025Q4", "reused_test": "2026Q2"}
BINS = [0.0, 1_000_000.0, 10_000_000.0, 100_000_000.0, 1_000_000_000.0, np.inf]
BIN_LABELS = ["(0,1M)", "[1M,10M)", "[10M,100M)", "[100M,1B)", "[1B,infinity)"]


def denominator_bins(values: pd.Series) -> pd.Series:
    """Use disjoint intervals; zero or negative comparisons are invalid."""
    numeric = pd.to_numeric(values, errors="raise")
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Comparison denominator must be finite and positive")
    bins = pd.cut(numeric, bins=BINS, labels=BIN_LABELS, right=False)
    if bins.isna().any():
        raise ValueError("Comparison denominator outside recorded bins")
    return bins


def read_name_maps(path: Path) -> tuple[pd.Series, pd.Series]:
    """Preserve every observed name, including ambiguous code names."""
    refs = pd.read_csv(
        path,
        encoding="cp949",
        dtype=str,
        usecols=["행정동_코드", "행정동_코드_명", "서비스_업종_코드", "서비스_업종_코드_명"],
    )

    def all_names(values: pd.Series) -> str | None:
        names = sorted(set(values.dropna()))
        return " / ".join(names) if names else None

    regions = refs.groupby("행정동_코드")["행정동_코드_명"].agg(all_names)
    industries = refs.groupby("서비스_업종_코드")["서비스_업종_코드_명"].agg(all_names)
    return regions, industries


def eligibility_counts(quarters: pd.DataFrame, target: pd.Period) -> tuple[int, int, int]:
    """Separate missing histories from nonpositive input denominators."""
    periods = [target - i for i in (4, 3, 2, 1, 0)]
    observed = quarters.loc[quarters[target].notna(), periods]
    missing = observed.isna().any(axis=1)
    nonpositive = (observed.iloc[:, :3] <= 0).any(axis=1) & ~missing
    return len(observed), int(missing.sum()), int(nonpositive.sum())


def exploratory_threshold_stats(frame: pd.DataFrame, threshold: float) -> dict:
    """Compare above/below threshold without excluding or changing predictions."""
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Exploratory threshold must be finite and positive")
    result = {}
    for label, subset in (
        ("below", frame.loc[frame.comparison_sales_krw < threshold]),
        ("at_or_above", frame.loc[frame.comparison_sales_krw >= threshold]),
    ):
        result[label] = {
            "rows": len(subset),
            "coverage_pct": float(len(subset) / len(frame) * 100) if len(frame) else 0.0,
            "model_mae_pp": float(subset.absolute_error.mean() * 100) if len(subset) else None,
            "zero_growth_mae_pp": float(subset.target.abs().mean() * 100) if len(subset) else None,
        }
    return result


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_errors(frame: pd.DataFrame, code: str, name: str) -> pd.DataFrame:
    grouped = (
        frame.groupby([code, name], dropna=False)
        .agg(
            n=("absolute_error", "size"),
            mean_absolute_error_pp=("absolute_error", lambda x: x.mean() * 100),
            median_absolute_error_pp=("absolute_error", lambda x: x.median() * 100),
            p90_absolute_error_pp=("absolute_error", lambda x: x.quantile(0.9) * 100),
            total_absolute_error=("absolute_error", "sum"),
        )
        .reset_index()
    )
    total = frame.absolute_error.sum()
    grouped["share_of_absolute_error_pct"] = (
        100 * grouped.total_absolute_error / total if total > 0 else 0.0
    )
    return grouped.sort_values("total_absolute_error", ascending=False).reset_index(drop=True)


def run(source: Path, names: Path, model_root: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refusing to overwrite analysis output")
    metadata_path = model_root / "metadata.json"
    model_path = model_root / "evaluation-pooled.joblib"
    inputs = {
        "source": source,
        "names_2025": names,
        "metadata": metadata_path,
        "model": model_path,
    }
    hashes = {key: sha(path) for key, path in inputs.items()}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_kind") != train_unified_growth.KIND:
        raise ValueError("Unexpected model type")
    if metadata.get("source_hashes", {}).get("seoul") != hashes["source"]:
        raise ValueError("Source hash differs from evaluated model")
    if metadata.get("model_sha256", {}).get(model_path.name) != hashes["model"]:
        raise ValueError("Model hash differs from metadata")
    modules = {
        "trainer": train_unified_growth.__file__,
        "feature_code": features.__file__,
        "unified_feature_code": unified_features.__file__,
        "base_trainer": train_gyeonggi_research.__file__,
    }
    for key, filename in modules.items():
        if metadata.get("code_sha256", {}).get(key) != sha(Path(filename)):
            raise ValueError(f"Code hash differs: {key}")
    # Load only locally trusted joblib files; hashes alone do not guarantee safety.
    bundle = joblib.load(model_path)
    quarters = train_unified_growth.seoul_quarters(source)
    region, industry = read_name_maps(names)
    training_target = pd.Period("2025Q2", freq="Q")
    training = train_unified_growth.make_frame(quarters, str(training_target), "seoul")
    training_scopes = pd.MultiIndex.from_frame(training[train_gyeonggi_research.SCOPE])
    training_denominator = quarters.loc[training_scopes, training_target - 4].to_numpy()
    threshold = float(np.quantile(training_denominator, 0.05))
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with tempfile.TemporaryDirectory(prefix=".seoul-error-", dir=output.parent) as temp:
        staged = Path(temp) / "output"
        staged.mkdir()
        for partition, period in PERIODS.items():
            target = pd.Period(period, freq="Q")
            observed, missing, nonpositive = eligibility_counts(quarters, target)
            frame = train_unified_growth.make_frame(quarters, period, "seoul")
            prediction = train_unified_growth.predict_unified_model(
                bundle, frame[list(unified_features.UNIFIED_RESEARCH_FEATURES)]
            )
            frame["prediction"] = prediction
            frame["absolute_error"] = np.abs(frame.target - prediction)
            scopes = pd.MultiIndex.from_frame(frame[train_gyeonggi_research.SCOPE])
            frame["comparison_sales_krw"] = quarters.loc[scopes, target - 4].to_numpy()
            frame["target_sales_krw"] = quarters.loc[scopes, target].to_numpy()
            frame["region_name_2025_reference"] = frame["행정동코드"].map(region)
            frame["industry_name_2025_reference"] = frame["중분류업종코드"].map(industry)
            frame["denominator_bin_krw"] = denominator_bins(frame.comparison_sales_krw)
            total = frame.absolute_error.sum()
            bins = (
                frame.groupby("denominator_bin_krw", observed=False)
                .agg(
                    rows=("absolute_error", "size"),
                    mean_absolute_error_pp=("absolute_error", lambda x: x.mean() * 100),
                    median_absolute_error_pp=(
                        "absolute_error",
                        lambda x: x.median() * 100,
                    ),
                    total_absolute_error=("absolute_error", "sum"),
                )
                .reset_index()
            )
            bins["share_of_absolute_error_pct"] = (
                100 * bins.total_absolute_error / total if total > 0 else 0.0
            )
            regions = group_errors(frame, "행정동코드", "region_name_2025_reference")
            industries = group_errors(frame, "중분류업종코드", "industry_name_2025_reference")
            frame.to_csv(staged / f"{partition}_rows.csv", index=False)
            regions.to_csv(staged / f"{partition}_regions.csv", index=False)
            industries.to_csv(staged / f"{partition}_industries.csv", index=False)
            bins.to_csv(staged / f"{partition}_denominator_bins.csv", index=False)
            k = max(1, int(np.ceil(len(frame) * 0.01)))
            summary[partition] = {
                "target_period": period,
                "rows_with_target": observed,
                "history_missing": missing,
                "invalid_denominators_with_complete_history": nonpositive,
                "evaluated": len(frame),
                "mae_pp": float(frame.absolute_error.mean() * 100),
                "median_absolute_error_pp": float(frame.absolute_error.median() * 100),
                "p90_absolute_error_pp": float(frame.absolute_error.quantile(0.9) * 100),
                "top_1pct_rows": k,
                "top_1pct_error_share_pct": float(
                    frame.absolute_error.nlargest(k).sum() / total * 100
                )
                if total > 0
                else 0.0,
                "missing_region_names": int(frame.region_name_2025_reference.isna().sum()),
                "missing_industry_names": int(frame.industry_name_2025_reference.isna().sum()),
                "posthoc_threshold_slices": exploratory_threshold_stats(frame, threshold),
            }
        if hashes != {key: sha(path) for key, path in inputs.items()}:
            raise ValueError("Input changed while reading")
        report = {
            "purpose": "retrospective_diagnostic_not_independent_test",
            "source_sha256": hashes,
            "code_sha256": {key: sha(Path(path)) for key, path in modules.items()},
            "name_reference": "Seoul original 2025 sales code names; not a crosswalk",
            "denominator_bin_boundaries": "Half-open positive KRW intervals; first label (0,1M)",
            "output_sha256": {path.name: sha(path) for path in sorted(staged.glob("*.csv"))},
            "posthoc_threshold": {
                "amount_krw": threshold,
                "quantile": 0.05,
                "training_target_period": str(training_target),
                "training_rows": len(training),
                "note": "Exploratory choice made after previous evaluation; not a model metric.",
            },
            "results": summary,
            "warnings": [
                "2026Q2 test was previously inspected and is reused; not an independent holdout.",
                "Name references and code matches do not verify geographic boundaries.",
                "Small denominators and abrupt sales changes are not proven source errors.",
                "Rows with incomplete history are excluded and may create selection bias.",
                "Posthoc subgroup metrics are exploratory; no automatic exclusion or API changes.",
                "Generated rows contain aggregate real data; do not publish automatically.",
            ],
        }
        (staged / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError("Refusing to overwrite analysis output")
        os.rename(staged, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source, args.names, args.model_root, args.output)
    print(json.dumps(result["results"], ensure_ascii=False))


if __name__ == "__main__":
    main()
