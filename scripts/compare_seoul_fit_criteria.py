"""서울 적합도 기준의 표본 내 표현력 비교이며 수익·사업성 검증이 아니다."""

import argparse
import json
import math
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pickplace_ai import data, periods, scoring
from pickplace_ai.artifacts import file_sha256
from scripts import audit_seoul_scores as audit

KIND = "seoul_fit_criteria_cross_section_research_v1"
CRITERIA = ("fixed_500_50", "industry_q75_q75", "industry_q90_q90")
REQUIRED = (
    "period",
    "region_code",
    "industry_code",
    "fit_score",
    "residents_per_store",
    "stores_per_km2",
)


def _common_scopes(frame: pd.DataFrame) -> pd.DataFrame:
    if not set(REQUIRED).issubset(frame):
        raise ValueError("Missing required score audit columns")
    if set(frame.period) != {audit.SNAPSHOT_PERIOD}:
        raise ValueError("Compare the 2026Q2 cross-section only")
    selected = frame.loc[frame.industry_code.isin(audit.SERVICE_CODES)].copy()
    if selected.duplicated(["industry_code", "region_code"]).any():
        raise ValueError("Duplicate industry and original region code")
    available = selected.loc[selected.fit_score.notna()].copy()
    values = available[["fit_score", "residents_per_store", "stores_per_km2"]].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values[:, 0] < 0).any()
        or (values[:, 0] > 100).any()
        or (values[:, 1] < 0).any()
        or (values[:, 2] <= 0).any()
    ):
        raise ValueError("Expected finite scores, nonnegative demand and positive density")
    counts = available.groupby("region_code").industry_code.nunique()
    regions = counts.index[counts == len(audit.SERVICE_CODES)]
    if len(regions) < 2:
        raise ValueError("At least two common valid regions are needed")
    return available.loc[available.region_code.isin(regions)].sort_values(
        ["industry_code", "region_code"]
    )


def _score(group, demand_threshold, density_threshold, population_scale=1.0):
    if not all(math.isfinite(x) and x > 0 for x in (demand_threshold, density_threshold)):
        raise ValueError("Reference quantiles must be finite and positive")
    demand = (group.residents_per_store * population_scale / demand_threshold * 100).clip(0, 100)
    competition = ((1 - group.stores_per_km2 / density_threshold) * 100).clip(0, 100)
    score = ((demand + competition) / 2).map(lambda value: round(float(value), 2))
    return demand / 2, competition / 2, score


def _top_regions(group: pd.DataFrame, score: pd.Series, count: int) -> set[str]:
    return set(
        group.assign(comparison_score=score)
        .sort_values(["comparison_score", "region_code"], ascending=[False, True])
        .head(count)
        .region_code
    )


def _spearman(first: pd.Series, second: pd.Series) -> float | None:
    if first.nunique() < 2 or second.nunique() < 2:
        return None
    value = first.rank(method="average").corr(second.rank(method="average"))
    return float(value) if math.isfinite(value) else None


def compare(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """동일 유효 지역의 임계값 시나리오와 분리된 두 분위 지표를 집계한다."""
    common = _common_scopes(frame)
    rows, component_rows = [], []
    for industry, group in common.groupby("industry_code", sort=True):
        count = len(group)
        observed = group.fit_score
        _, _, reconstructed = _score(group, 500.0, 50.0)
        if not np.array_equal(observed.to_numpy(), reconstructed.to_numpy()):
            raise ValueError("The current scoring output no longer matches draft-v0.1 criteria")
        thresholds = {"fixed_500_50": (500.0, 50.0)}
        for quantile, name in ((0.75, "industry_q75_q75"), (0.90, "industry_q90_q90")):
            thresholds[name] = (
                float(np.quantile(group.residents_per_store, quantile, method="linear")),
                float(np.quantile(group.stores_per_km2, quantile, method="linear")),
            )
        top_count = math.ceil(count * 0.1)
        top_before = _top_regions(group, observed, top_count)
        for name, (demand_threshold, density_threshold) in thresholds.items():
            demand, competition, score = _score(group, demand_threshold, density_threshold)
            # 인구 변화 실험에서 비교 기준을 다시 적합하면 공통 증가가 상쇄되므로 고정한다.
            _, _, changed = _score(group, demand_threshold, density_threshold, population_scale=1.1)
            delta = changed - score
            rows.append(
                {
                    "period": audit.SNAPSHOT_PERIOD,
                    "industry_code": industry,
                    "criteria": name,
                    "matched_rows": count,
                    "residents_per_store_threshold": demand_threshold,
                    "density_threshold": density_threshold,
                    "demand_saturated_rows": int((demand == 50).sum()),
                    "demand_saturated_fraction": float((demand == 50).mean()),
                    "competition_zero_rows": int((competition == 0).sum()),
                    "competition_zero_fraction": float((competition == 0).mean()),
                    "both_plateau_50_rows": int(((demand == 50) & (competition == 0)).sum()),
                    "demand_mean_points": float(demand.mean()),
                    "competition_mean_points": float(competition.mean()),
                    "score_mean": float(score.mean()),
                    "score_median": float(score.median()),
                    "score_p10": float(score.quantile(0.1, interpolation="linear")),
                    "score_p90": float(score.quantile(0.9, interpolation="linear")),
                    "score_min": float(score.min()),
                    "score_max": float(score.max()),
                    "unique_scores": int(score.nunique()),
                    "residents_x1_1_mean_shift": float(delta.mean()),
                    "residents_x1_1_median_shift": float(delta.median()),
                    "residents_x1_1_unchanged_rows": int((delta == 0).sum()),
                    "score_spearman_vs_fixed": _spearman(observed, score),
                    "top_decile_rows": top_count,
                    "top_decile_overlap_vs_fixed": len(
                        top_before & _top_regions(group, score, top_count)
                    ),
                }
            )
        # 상대 순위는 현황 설명으로만 남긴다. 두 항목을 적합도 점수로 합치지 않는다.
        for column in ("residents_per_store", "stores_per_km2"):
            values = group[column]
            percentile = (values.rank(method="average") - 1) / (count - 1) * 100
            component_rows.append(
                {
                    "period": audit.SNAPSHOT_PERIOD,
                    "industry_code": industry,
                    "component": column,
                    "matched_rows": count,
                    "raw_unit": "persons_per_store"
                    if column == "residents_per_store"
                    else "stores_per_km2",
                    "raw_p10": float(values.quantile(0.1, interpolation="linear")),
                    "raw_median": float(values.median()),
                    "raw_p90": float(values.quantile(0.9, interpolation="linear")),
                    "percentile_p10": float(percentile.quantile(0.1, interpolation="linear")),
                    "percentile_median": float(percentile.median()),
                    "percentile_p90": float(percentile.quantile(0.9, interpolation="linear")),
                    "percentile_mean": float(percentile.mean()),
                    "percentile_min": float(percentile.min()),
                    "percentile_max": float(percentile.max()),
                    "unique_percentiles": int(percentile.nunique()),
                }
            )
    return pd.DataFrame.from_records(rows), pd.DataFrame.from_records(component_rows)


def _snapshot_date(manifest: dict) -> datetime:
    created = datetime.fromisoformat(manifest["created_at_utc"])
    if created.tzinfo is None:
        raise ValueError("The snapshot acquisition timestamp must include its timezone")
    created = created.astimezone(UTC)
    if created.date() < periods.quarter_end(audit.SNAPSHOT_PERIOD):
        raise ValueError("Snapshot acquisition precedes the end of 2026Q2")
    for name in ("stores", "resident"):
        if audit.SNAPSHOT_PERIOD not in manifest["datasets"][name]["periods"]:
            raise ValueError("The snapshot manifest does not contain the comparison quarter")
    return created


def run(snapshot: Path, area: Path, output: Path) -> dict:
    snapshot, area, output = Path(snapshot), Path(area), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output directory already exists")
    database, manifest_path = snapshot / "snapshot.sqlite3", snapshot / "manifest.json"
    inputs = {"manifest": manifest_path, "database": database, "area": area}
    codes = {
        "runner": Path(__file__),
        "score_audit": Path(audit.__file__),
        "data": Path(data.__file__),
        "scoring": Path(scoring.__file__),
    }
    input_hashes = {name: file_sha256(path) for name, path in inputs.items()}
    code_hashes = {name: file_sha256(path) for name, path in codes.items()}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("mode") != "seoul_descriptive_research_only"
        or manifest.get("database_sha256") != input_hashes["database"]
        or scoring.POLICY_VERSION != "draft-v0.1"
    ):
        raise ValueError("Expected the matching statistics snapshot and draft-v0.1 policy")
    acquired = _snapshot_date(manifest)
    areas = audit.read_area(area)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        frame = audit.scored_rows(
            connection,
            (audit.SNAPSHOT_PERIOD,),
            areas,
            audit.SNAPSHOT_PERIOD,
            acquired_at=acquired.date(),
        )
    criteria, components = compare(frame)
    metadata = {
        "kind": KIND,
        "policy_version": scoring.POLICY_VERSION,
        "period": audit.SNAPSHOT_PERIOD,
        "statistics_snapshot_created_at": manifest["created_at_utc"],
        "matched_regions": int(criteria.matched_rows.iloc[0]),
        "service_industry_codes": audit.SERVICE_CODES,
        "criteria": CRITERIA,
        "quantile_method": "linear quantiles within each industry and the common cross-section",
        "in_sample_comparison_not_validation": True,
        "outcome_validated": False,
        "area_effective_period_verified": False,
        "area_assumption": "raw area interpreted as square metres and divided by 1000000",
        "population_sensitivity": "multiply residents by 1.1 with fixed thresholds",
        "percentile_definition": "(average ascending rank - 1) / (matched rows - 1) * 100",
        "percentile_components_combined": False,
        "ranking_ties": (
            "average rank for Spearman. For top decile, sort by rounded score descending "
            "and then original region code ascending."
        ),
        "inputs_sha256": input_hashes,
        "code_sha256": code_hashes,
        "limitations": [
            "Profitability and individual survival outcomes require validation.",
            "Population can repeat across quarters. Its source reference date requires evidence.",
            "Quantile cutoffs determine saturation rates by construction.",
            "New cross-sections can change cutoffs and relative positions without a local change.",
            "Unit and date verification is independent of relative positions.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        publish = Path(staging) / "result"
        publish.mkdir()
        criteria.to_csv(publish / "criteria-comparison.csv", index=False)
        components.to_csv(publish / "component-percentiles.csv", index=False)
        metadata["outputs_sha256"] = {
            name: file_sha256(publish / name)
            for name in ("criteria-comparison.csv", "component-percentiles.csv")
        }
        (publish / "summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if input_hashes != {name: file_sha256(path) for name, path in inputs.items()}:
            raise ValueError("Input changed during the criteria comparison")
        if code_hashes != {name: file_sha256(path) for name, path in codes.items()}:
            raise ValueError("Code changed during the criteria comparison")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output directory already exists")
        publish.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--area", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.snapshot, args.area, args.output)
    print(json.dumps({"output": str(args.output), "matched_regions": summary["matched_regions"]}))


if __name__ == "__main__":
    main()
