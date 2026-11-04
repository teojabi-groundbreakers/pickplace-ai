import csv
import json

import pytest

from scripts import evaluate_forecast_2026 as evaluation


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    forecast_dir = tmp_path / "forecast"
    forecast_dir.mkdir()
    forecast = []
    snapshot = []
    for region, quarter, estimate in [
        ("11110515", 1, "0.1"),
        ("11110516", 1, "0.2"),
        ("11110515", 2, "0.0"),
    ]:
        forecast.append(
            {
                "region_code": region,
                "industry_code": "CS100010",
                "target_period": f"2026Q{quarter}",
                "base_period": "2025Q4",
                "comparison_period": f"2025Q{quarter}",
                "comparison_sales": "100",
                "estimated_yoy_growth": estimate,
                "snapshot_available_at": "2026-11-03",
                "as_of": "2026-11-04",
            }
        )
        snapshot.append(
            {
                "region_code": region,
                "region_code_system": "seoul-adstrd",
                "industry_code": "CS100010",
                "period": f"2025Q{quarter}",
                "sales_amount": "100",
                "available_at": "2026-11-03",
            }
        )
    forecast_path = forecast_dir / "forecast-2026.csv"
    snapshot_path = tmp_path / "snapshot.csv"
    write_csv(forecast_path, forecast)
    write_csv(snapshot_path, snapshot)
    summary = {
        "artifact_kind": "seoul_year_ahead_snapshot_forecast_v1",
        "input_sha256": evaluation.digest(snapshot_path),
        "forecast_sha256": evaluation.digest(forecast_path),
        "snapshot_available_at": "2026-11-03",
        "as_of": "2026-11-04",
        "created_at": "2026-11-04T12:08:06+00:00",
        "selection": "huber",
    }
    (forecast_dir / "summary.json").write_text(json.dumps(summary))
    sales = []
    for quarter, regions in [
        (1, [("11110515", "120"), ("11110517", "110")]),
        (2, [("11110515", "90")]),
    ]:
        path = tmp_path / f"sales-{quarter}.csv"
        rows = [
            {
                "STDR_YYQU_CD": f"2026{quarter}",
                "ADSTRD_CD": region,
                "SVC_INDUTY_CD": "CS100010",
                "THSMON_SELNG_AMT": amount,
            }
            for region, amount in regions
        ]
        write_csv(path, rows)
        path.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "kind": "sales",
                    "period": f"2026Q{quarter}",
                    "sha256": evaluation.digest(path),
                    "rows": len(rows),
                    "acquired_at_utc": "2026-11-04T12:25:29+00:00",
                }
            )
        )
        sales.append(path)
    return forecast_dir, snapshot_path, sales, tmp_path / "result"


def test_evaluates_exact_join_and_baseline(inputs):
    report = evaluation.evaluate(*inputs)
    assert report["coverage"] == {
        "forecast_rows": 3,
        "api_rows": 3,
        "matched_rows": 2,
        "forecast_without_api": 1,
        "api_without_forecast": 1,
    }
    assert report["overall"]["model_mae_pp"] == pytest.approx(10)
    assert report["overall"]["zero_growth_mae_pp"] == pytest.approx(15)
    assert report["prospective_evaluation"] is False
    assert report["by_quarter"]["2026Q1"]["rows"] == 1
    assert evaluation.digest(inputs[3] / "comparison.csv") == report["comparison_sha256"]


def test_denominator_mismatch_refused_even_when_snapshot_hash_updated(inputs):
    forecast_dir, snapshot, sales, output = inputs
    text = snapshot.read_text().replace(",100,2026-11-03", ",101,2026-11-03", 1)
    snapshot.write_text(text)
    summary_path = forecast_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["input_sha256"] = evaluation.digest(snapshot)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="denominator differs"):
        evaluation.evaluate(forecast_dir, snapshot, sales, output)
    assert not output.exists()


def test_duplicate_api_row_refused(inputs):
    forecast_dir, snapshot, sales, output = inputs
    path = sales[0]
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.append(rows[0])
    write_csv(path, rows)
    meta_path = path.with_suffix(".metadata.json")
    meta = json.loads(meta_path.read_text())
    meta.update(sha256=evaluation.digest(path), rows=len(rows))
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Duplicate or missing API"):
        evaluation.evaluate(forecast_dir, snapshot, sales, output)
    assert not output.exists()


def test_hash_mismatch_and_existing_output_are_refused(inputs):
    forecast_dir, snapshot, sales, output = inputs
    snapshot.write_text(snapshot.read_text() + "\n")
    with pytest.raises(ValueError, match="snapshot hash"):
        evaluation.evaluate(forecast_dir, snapshot, sales, output)
    assert not output.exists()
    output.mkdir()
    (output / "existing").write_text("keep")
    with pytest.raises(FileExistsError):
        evaluation.evaluate(forecast_dir, snapshot, sales, output)
    assert (output / "existing").read_text() == "keep"


def test_metadata_and_period_mismatch_are_refused(inputs):
    forecast_dir, snapshot, sales, output = inputs
    meta = sales[0].with_suffix(".metadata.json")
    data = json.loads(meta.read_text())
    data["period"] = "2026Q2"
    meta.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="period differs"):
        evaluation.evaluate(forecast_dir, snapshot, sales, output)
    assert not output.exists()
