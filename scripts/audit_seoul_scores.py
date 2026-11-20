"""서울 실자료로 위험도·적합도 초안의 분포와 순위 변별력을 확인한다."""

import argparse
import json
import math
import re
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from pickplace_ai import periods, scoring
from pickplace_ai.artifacts import file_sha256
from pickplace_ai.schemas import Observation
from scripts.seoul_score_audit_metrics import distributions, risk_outcomes

SNAPSHOT_PERIOD = "2026Q2"
FIRST_RISK_BASE = "2024Q3"
SERVICE_CODES = ("CS100010", "CS100008", "CS100005")
AREA_COLUMNS = ["행정동_코드", "영역_면적"]

_SCORE_ROWS = """
SELECT s.region_code, s.industry_code, s.store_count, s.closing_count,
       cur.sales_amount, past.sales_amount, resident.total,
       out_sales.sales_amount, out_past.sales_amount, out_s.store_count,
       out_s.closing_count
FROM stores AS s
LEFT JOIN sales AS cur USING (period, region_code, industry_code)
LEFT JOIN sales AS past
  ON past.period = :past AND past.region_code = s.region_code
  AND past.industry_code = s.industry_code
LEFT JOIN resident ON resident.period = s.period AND resident.region_code = s.region_code
LEFT JOIN sales AS out_sales
  ON out_sales.period = :future AND out_sales.region_code = s.region_code
  AND out_sales.industry_code = s.industry_code
LEFT JOIN sales AS out_past
  ON out_past.period = :future_past AND out_past.region_code = s.region_code
  AND out_past.industry_code = s.industry_code
LEFT JOIN stores AS out_s
  ON out_s.period = :future AND out_s.region_code = s.region_code
  AND out_s.industry_code = s.industry_code
WHERE s.period = :base
"""


def read_area(path: Path) -> dict[str, float]:
    """기준일이 없는 서울 면적을 단일 분기의 ㎡ 가정 시나리오로 읽는다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            frame = pd.read_csv(path, encoding=encoding, dtype=str, usecols=AREA_COLUMNS)
            break
        except UnicodeDecodeError:
            if encoding == "cp949":
                raise
    result = {}
    for region, area in frame[AREA_COLUMNS].itertuples(index=False, name=None):
        if not isinstance(region, str) or not re.fullmatch(r"11[0-9]{6}", region):
            raise ValueError("Unexpected Seoul region code in area file")
        value = float(area)
        if not math.isfinite(value) or value <= 0 or region in result:
            raise ValueError("Expected unique region codes and finite positive areas")
        result[region] = value / 1_000_000
    if not result:
        raise ValueError("No area observations")
    return result


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def scored_rows(
    connection: sqlite3.Connection,
    periods_to_score: tuple[str, ...],
    area_snapshot: dict[str, float],
    snapshot_period: str,
    *,
    acquired_at: date,
) -> pd.DataFrame:
    """원본 코드로 같은 분기 지표를 묶고 실제 점수 계산 함수를 호출한다."""
    rows = []
    for base in periods_to_score:
        past = periods.shift_quarter(base, -4)
        future = periods.shift_quarter(base, 1)
        params = {
            "base": base,
            "past": past,
            "future": None if base == snapshot_period else future,
            "future_past": periods.shift_quarter(future, -4),
        }
        for (
            region,
            industry,
            stores,
            closings,
            sales,
            prior_sales,
            residents,
            future_sales,
            future_prior_sales,
            future_stores,
            future_closings,
        ) in connection.execute(_SCORE_ROWS, params):
            scope = {
                "region_code": region,
                "region_code_system": "seoul-adstrd",
                "industry_code": industry,
                "source_kind": "official",
                "source_dataset": "seoul_descriptive_research_snapshot",
            }
            # 확보 시점이 서로 다른 수정 자료는 API 요청이 아닌 오프라인 산식 검증에 쓴다.
            base_row = Observation(
                **scope,
                period=base,
                available_at=acquired_at,
                store_count=stores,
                closing_count=closings,
                sales_amount=sales,
                resident_population=residents,
                area_km2=area_snapshot.get(region) if base == snapshot_period else None,
            )
            history = [base_row]
            if prior_sales is not None:
                history.append(
                    Observation(
                        **scope, period=past, available_at=acquired_at, sales_amount=prior_sales
                    )
                )
            result = scoring.diagnose(history, base)
            risk, fit = result["risk"], result["fit"]
            components = {item["name"]: item for item in risk["components"] + fit["components"]}
            sales_risk = components.get("sales_decline", {})
            closure = components.get("closings_per_current_store", {})
            demand = components.get("residents_per_store", {})
            competition = components.get("store_density", {})
            next_sales_ratio = _ratio(future_sales, future_prior_sales)
            rows.append(
                {
                    "industry_code": industry,
                    "region_code": region,
                    "period": base,
                    "has_sales": sales is not None,
                    "risk_score": risk["score"],
                    "sales_risk_points": sales_risk.get("contribution_points"),
                    "closure_risk_points": closure.get("contribution_points"),
                    "sales_yoy": sales_risk.get("value"),
                    "closure_ratio": closure.get("value"),
                    "fit_score": fit["score"],
                    "demand_points": demand.get("contribution_points"),
                    "competition_points": competition.get("contribution_points"),
                    "residents_per_store": demand.get("value"),
                    "stores_per_km2": competition.get("value"),
                    "sales_per_store": _ratio(sales, stores),
                    "next_sales_yoy": None if next_sales_ratio is None else next_sales_ratio - 1,
                    "next_closure_ratio": _ratio(future_closings, future_stores),
                }
            )
    if not rows:
        raise ValueError("No eligible store periods in the snapshot")
    return pd.DataFrame.from_records(rows)


def fit_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    """세 업종의 공통 지역에서 입력과 임계값 변화에 따른 순위 민감도를 측정한다."""
    fit = frame.loc[frame.fit_score.notna() & frame.industry_code.isin(SERVICE_CODES)].copy()
    region_counts = fit.groupby("region_code").industry_code.nunique()
    common = region_counts.index[region_counts == len(SERVICE_CODES)]
    fit = fit.loc[fit.region_code.isin(common)]
    rows = []
    for industry, group in fit.groupby("industry_code"):
        demand, density = group.residents_per_store, group.stores_per_km2
        observed = group.fit_score
        scenarios = {
            "residents_plus_10pct": (
                (demand * 1.1 / 500).clip(0, 1) + (1 - density / 50).clip(0, 1)
            )
            * 50,
            "double_both_thresholds": ((demand / 1000).clip(0, 1) + (1 - density / 100).clip(0, 1))
            * 50,
        }
        top_count = math.ceil(len(group) * 0.1)
        top_before = set(
            group.sort_values(["fit_score", "region_code"], ascending=[False, True])
            .head(top_count)
            .region_code
        )
        for name, raw_score in scenarios.items():
            score = raw_score.map(lambda value: round(value, 2))
            delta = score - observed
            top_after = set(
                group.assign(scenario_score=score)
                .sort_values(["scenario_score", "region_code"], ascending=[False, True])
                .head(top_count)
                .region_code
            )
            rows.append(
                {
                    "industry_code": industry,
                    "scenario": name,
                    "matched_rows": len(group),
                    "median_score_change": delta.median(),
                    "min_score_change": delta.min(),
                    "max_score_change": delta.max(),
                    "unchanged_rows": int((delta == 0).sum()),
                    "rank_spearman": observed.rank().corr(score.rank()),
                    "top_decile_rows": top_count,
                    "top_decile_overlap": len(top_before & top_after),
                }
            )
    if not rows:
        raise ValueError("No common fit scopes for the three service industries")
    return pd.DataFrame.from_records(rows)


def run(snapshot_dir: Path, area_path: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("Output directory already exists")
    database = snapshot_dir / "snapshot.sqlite3"
    manifest_path = snapshot_dir / "manifest.json"
    input_hashes = {path: file_sha256(path) for path in (database, manifest_path, area_path)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest["mode"] != "seoul_descriptive_research_only"
        or manifest["database_sha256"] != input_hashes[database]
    ):
        raise ValueError("Expected a matching Seoul statistics snapshot and manifest")
    latest = SNAPSHOT_PERIOD
    first = FIRST_RISK_BASE
    risk_bases = tuple(
        periods.shift_quarter(first, offset)
        for offset in range(periods.quarter_index(latest) - periods.quarter_index(first))
    )
    area = read_area(area_path)
    acquired_at = datetime.fromisoformat(manifest["created_at_utc"]).astimezone(UTC).date()
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        frame = scored_rows(conn, (*risk_bases, latest), area, latest, acquired_at=acquired_at)
        latest_closing_count_exceeds_stores = conn.execute(
            "SELECT count(*) FROM stores WHERE period = ? AND closing_count > store_count",
            (latest,),
        ).fetchone()[0]
        resident_start = "2023Q4"
        resident_periods = periods.quarter_index(latest) - periods.quarter_index(resident_start) + 1
        complete_resident, unchanged_resident = conn.execute(
            """
            SELECT count(*), sum(distinct_totals = 1) FROM (
                SELECT count(DISTINCT total) AS distinct_totals
                FROM resident WHERE period BETWEEN ? AND ?
                GROUP BY region_code HAVING count(total) = ?
            )
            """,
            (resident_start, latest, resident_periods),
        ).fetchone()
    if set(frame.period) != {*risk_bases, latest}:
        raise ValueError("Requested analysis quarters are missing")
    last = frame.loc[frame.period == latest]
    history = frame.loc[frame.period != latest]
    result = distributions(last)
    sensitivity = fit_sensitivity(last)
    outcomes = pd.concat(
        [
            risk_outcomes(history).assign(base_period="ALL"),
            *(
                risk_outcomes(group).assign(base_period=base)
                for base, group in history.groupby("period")
            ),
        ],
        ignore_index=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging:
        publish = Path(staging) / "result"
        publish.mkdir()
        result.to_csv(publish / "industry-distribution.csv", index=False)
        outcomes.to_csv(publish / "risk-outcomes.csv", index=False)
        sensitivity.to_csv(publish / "fit-sensitivity.csv", index=False)
        metadata = {
            "kind": "seoul_draft_scores_research",
            "policy_version": scoring.POLICY_VERSION,
            "statistics_snapshot_created_at": manifest["created_at_utc"],
            "period": latest,
            "risk_base_periods": list(risk_bases),
            "service_industry_codes": SERVICE_CODES,
            "area_assumption": "assume_raw_area_m2_and_divide_by_1000000_for_latest_period",
            "area_source_rows": len(area),
            "area_effective_period_verified": False,
            "historical_publication_verified": False,
            "latest_closing_count_exceeds_store_count_rows": latest_closing_count_exceeds_stores,
            "resident_repeat_check": {
                "first_period": resident_start,
                "last_period": latest,
                "complete_regions": complete_resident,
                "unchanged_regions": unchanged_resident or 0,
            },
            "inputs_sha256": {path.name: digest for path, digest in input_hashes.items()},
            "runner_sha256": file_sha256(Path(__file__)),
            "metrics_code_sha256": file_sha256(
                Path(__file__).with_name("seoul_score_audit_metrics.py")
            ),
            "scoring_sha256": file_sha256(Path(scoring.__file__)),
            "outputs_sha256": {
                name: file_sha256(publish / name)
                for name in (
                    "industry-distribution.csv",
                    "risk-outcomes.csv",
                    "fit-sensitivity.csv",
                )
            },
        }
        (publish / "summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if any(file_sha256(path) != digest for path, digest in input_hashes.items()):
            raise ValueError("Input changed during the audit")
        if output.exists():
            raise FileExistsError("Output directory already exists")
        publish.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--area", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.snapshot, args.area, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
