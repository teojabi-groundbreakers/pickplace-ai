"""Retrospective comparison of the frozen 2026 forecast to public sales estimates.

This is not a historical live forecast or a prospective performance assessment.
2025 denominators are verified against the forecast's originally used snapshot.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

QUARTERS = ("2026Q1", "2026Q2")
KEYS = ("region_code", "industry_code", "target_period")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def records(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames
        if not names or len(names) != len(set(names)) or not required.issubset(names):
            raise ValueError(f"Missing or duplicate required headers in {path}")
        for row in reader:
            if None in row or any(x is None for x in row.values()):
                raise ValueError(f"Malformed CSV row in {path}")
            yield row


def number(value, *, positive=False):
    try:
        n = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid numeric field") from exc
    if not n.is_finite() or n < 0 or (positive and n <= 0):
        raise ValueError("Expected finite, nonnegative amount and positive denominator")
    return n


def measurement(group):
    model = [float(r["abs_error"]) for r in group]
    baseline = [float(r["zero_abs_error"]) for r in group]
    if not model or any(not math.isfinite(v) for v in model + baseline):
        raise ValueError("Empty or non-finite comparison")
    return {
        "rows": len(group),
        "model_mae_pp": 100 * statistics.fmean(model),
        "model_median_absolute_error_pp": 100 * statistics.median(model),
        "zero_growth_mae_pp": 100 * statistics.fmean(baseline),
        "zero_growth_median_absolute_error_pp": 100 * statistics.median(baseline),
        "denominator_below_1m_krw": sum(r["comparison_sales"] < Decimal(1000000) for r in group),
        "actual_growth_above_100pct": sum(r["actual_yoy_growth"] > 1 for r in group),
        "model_absolute_error_above_100pp": sum(r["abs_error"] > 1 for r in group),
    }


def evaluate(forecast_dir, snapshot_path, sales_paths, output_path):
    forecast_dir, snapshot_path, output_path = map(Path, (forecast_dir, snapshot_path, output_path))
    sales_paths = list(map(Path, sales_paths))
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    summary_path, forecast_path = forecast_dir / "summary.json", forecast_dir / "forecast-2026.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("artifact_kind") != "seoul_year_ahead_snapshot_forecast_v1":
        raise ValueError("Unexpected forecast artifact")
    if digest(snapshot_path) != summary.get("input_sha256"):
        raise ValueError("2025 denominator snapshot hash differs from forecast source")
    if digest(forecast_path) != summary.get("forecast_sha256"):
        raise ValueError("Frozen forecast file hash differs from forecast summary")
    forecast = {}
    required = set(KEYS) | {
        "comparison_period",
        "comparison_sales",
        "estimated_yoy_growth",
        "base_period",
        "snapshot_available_at",
        "as_of",
    }
    for row in records(forecast_path, required):
        period = row["target_period"]
        if period not in QUARTERS:
            continue
        if row["base_period"] != "2025Q4" or row["comparison_period"] != "2025Q" + period[-1]:
            raise ValueError("Forecast reference quarter differs from target quarter")
        if row["snapshot_available_at"] != summary.get("snapshot_available_at") or row[
            "as_of"
        ] != summary.get("as_of"):
            raise ValueError("Forecast dates differ from summary")
        key = tuple(row[k] for k in KEYS)
        if not key[0] or not key[1] or key in forecast:
            raise ValueError("Duplicate or missing forecast key")
        denominator = number(row["comparison_sales"], positive=True)
        try:
            estimate = Decimal(row["estimated_yoy_growth"])
        except InvalidOperation as exc:
            raise ValueError("Invalid forecast growth") from exc
        if not estimate.is_finite() or estimate < -1:
            raise ValueError("Invalid forecast growth")
        forecast[key] = {"comparison_sales": denominator, "estimated_yoy_growth": estimate}
    if not forecast or set(k[2] for k in forecast) != set(QUARTERS):
        raise ValueError("Both target quarters are required in the forecast")

    verified = set()
    for row in records(
        snapshot_path,
        {
            "region_code",
            "region_code_system",
            "industry_code",
            "period",
            "sales_amount",
            "available_at",
        },
    ):
        if row["period"] not in ("2025Q1", "2025Q2"):
            continue
        key = (row["region_code"], row["industry_code"], "2026Q" + row["period"][-1])
        if key not in forecast:
            continue
        if key in verified:
            raise ValueError("Duplicate denominator snapshot key")
        if row["region_code_system"] != "seoul-adstrd" or row["available_at"] != summary.get(
            "snapshot_available_at"
        ):
            raise ValueError("Denominator scope or snapshot date differs")
        if number(row["sales_amount"], positive=True) != forecast[key]["comparison_sales"]:
            raise ValueError("Forecast denominator differs from 2025 snapshot")
        verified.add(key)
    if len(verified) != len(forecast):
        raise ValueError("Forecast denominator missing from 2025 snapshot")

    actual, metas = {}, {}
    for path in sales_paths:
        meta_path = path.with_suffix(".metadata.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        period = meta.get("period")
        if meta.get("kind") != "sales" or period not in QUARTERS or period in metas:
            raise ValueError("Expected one API sales file per 2026 target quarter")
        if digest(path) != meta.get("sha256"):
            raise ValueError("API sales data hash mismatch")
        captured = 0
        for row in records(
            path, {"STDR_YYQU_CD", "ADSTRD_CD", "SVC_INDUTY_CD", "THSMON_SELNG_AMT"}
        ):
            if row["STDR_YYQU_CD"] != period.replace("Q", ""):
                raise ValueError("API row period differs from source metadata")
            key = (row["ADSTRD_CD"], row["SVC_INDUTY_CD"], period)
            if not key[0] or not key[1] or key in actual:
                raise ValueError("Duplicate or missing API sales key")
            actual[key] = number(row["THSMON_SELNG_AMT"])
            captured += 1
        if captured != meta.get("rows"):
            raise ValueError("API sales row count differs from source metadata")
        metas[period] = {
            "file": str(path),
            "sha256": meta["sha256"],
            "metadata_file": str(meta_path),
            "metadata_sha256": digest(meta_path),
            "rows": captured,
            "acquired_at_utc": meta.get("acquired_at_utc"),
        }
    if set(metas) != set(QUARTERS):
        raise ValueError("Both API quarters are required")

    common = sorted(forecast.keys() & actual.keys())
    result = []
    for key in common:
        denominator = forecast[key]["comparison_sales"]
        observed = actual[key] / denominator - 1
        prediction = forecast[key]["estimated_yoy_growth"]
        result.append(
            dict(zip(KEYS, key, strict=True))
            | {
                "comparison_sales": denominator,
                "actual_sales": actual[key],
                "actual_yoy_growth": observed,
                "estimated_yoy_growth": prediction,
                "abs_error": abs(prediction - observed),
                "zero_abs_error": abs(observed),
            }
        )
    if not result:
        raise ValueError("No matching observations")
    groups = {period: [r for r in result if r["target_period"] == period] for period in QUARTERS}
    if any(not group for group in groups.values()):
        raise ValueError("No matched observations in a target quarter")
    sources = {
        "forecast": {"file": str(forecast_path), "sha256": digest(forecast_path)},
        "forecast_summary": {"file": str(summary_path), "sha256": digest(summary_path)},
        "snapshot_2025": {"file": str(snapshot_path), "sha256": digest(snapshot_path)},
        "api_sales": metas,
    }
    # Recheck all inputs before writing the report to detect concurrent changes.
    if (
        digest(forecast_path) != summary["forecast_sha256"]
        or digest(snapshot_path) != summary["input_sha256"]
        or digest(summary_path) != sources["forecast_summary"]["sha256"]
    ):
        raise ValueError("Forecast inputs changed during evaluation")
    for meta in metas.values():
        if (
            digest(meta["file"]) != meta["sha256"]
            or digest(meta["metadata_file"]) != meta["metadata_sha256"]
        ):
            raise ValueError("API input changed during evaluation")
    report = {
        "kind": "retrospective_2026_sales_comparison_v1",
        "prospective_evaluation": False,
        "warning": (
            "2026Q1-Q2 had ended and their public data existed when the forecast was generated; "
            "this is not prospective evaluation. The 2025 denominator uses the original "
            "2026-11-03 snapshot; later revisions were not checked."
        ),
        "target_definition": "2026 public estimated sales / original 2025 snapshot sales - 1",
        "forecast_created_at": summary.get("created_at"),
        "forecast_as_of": summary.get("as_of"),
        "sources": sources,
        "coverage": {
            "forecast_rows": len(forecast),
            "api_rows": len(actual),
            "matched_rows": len(common),
            "forecast_without_api": len(forecast.keys() - actual.keys()),
            "api_without_forecast": len(actual.keys() - forecast.keys()),
        },
        "overall": measurement(result),
        "by_quarter": {
            k: measurement(v)
            | {
                "forecast_rows": sum(x[2] == k for x in forecast),
                "api_rows": sum(x[2] == k for x in actual),
            }
            for k, v in groups.items()
        },
        "model_selection": summary.get("selection"),
    }
    if summary.get("created_at"):
        datetime.fromisoformat(summary["created_at"])
    output_path.mkdir(parents=True, exist_ok=False)
    try:
        with (output_path / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
        report["comparison_sha256"] = digest(output_path / "comparison.csv")
        (output_path / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        # Remove only files created in the new directory if the write fails.
        for name in ("comparison.csv", "report.json"):
            (output_path / name).unlink(missing_ok=True)
        output_path.rmdir()
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-dir", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--sales", required=True, nargs=2, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.forecast_dir, args.snapshot, args.sales, args.output)
    print(
        json.dumps(
            {"coverage": result["coverage"], "overall": result["overall"]}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
