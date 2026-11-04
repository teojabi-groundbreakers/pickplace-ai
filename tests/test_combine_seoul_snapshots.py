import csv
import json
from datetime import date

import pytest

from pickplace_ai.artifacts import file_sha256
from pickplace_ai.data import read_observations, write_observations
from pickplace_ai.schemas import Observation
from scripts.combine_seoul_snapshots import combine


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def input_files(tmp_path):
    raw = tmp_path / "raw"
    for kind in ("sales", "stores"):
        (raw / kind).mkdir(parents=True)
        (raw / kind / "historic.csv").write_text(kind)
    manifest = tmp_path / "old-manifest.json"
    save_json(
        manifest,
        {
            "industry_code": "all",
            "sales_basis": "as_published",
            "snapshot_available_at": "2026-11-03",
            "source_files": [
                {
                    "category": k,
                    "name": "historic.csv",
                    "sha256": file_sha256(raw / k / "historic.csv"),
                }
                for k in ("sales", "stores")
            ],
        },
    )
    historical = tmp_path / "old.csv"
    write_observations(
        [
            Observation(
                region_code="11110515",
                region_code_system="seoul-adstrd",
                industry_code="CS100010",
                period="2025Q4",
                available_at=date(2026, 11, 3),
                source_kind="official",
                source_dataset="fixture",
                sales_amount=100,
                store_count=8,
            )
        ],
        historical,
    )
    summary = tmp_path / "summary.json"
    save_json(
        summary,
        {
            "input_sha256": file_sha256(historical),
            "raw_source_manifest_sha256": file_sha256(manifest),
            "snapshot_available_at": "2026-11-03",
            "input_observations": 1,
        },
    )
    for kind in ("sales", "stores"):
        path = raw / kind / f"seoul-api-{kind}-2026Q1.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            rows = [
                {
                    "STDR_YYQU_CD": "20261",
                    "ADSTRD_CD": "11110515",
                    "SVC_INDUTY_CD": "CS100010",
                    **(
                        {"THSMON_SELNG_AMT": "1000"}
                        if kind == "sales"
                        else {"SIMILR_INDUTY_STOR_CO": "10"}
                    ),
                }
            ]
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        save_json(
            path.with_suffix(".metadata.json"),
            {
                "kind": kind,
                "period": "2026Q1",
                "rows": 1,
                "sha256": file_sha256(path),
                "collection_started_at_utc": "2026-11-04T12:00:00+00:00",
                "acquired_at_utc": "2026-11-04T12:01:00+00:00",
                "catalog": "https://data.seoul.go.kr/",
            },
        )
    return {
        "historical_input": historical,
        "historical_summary": summary,
        "historical_manifest": manifest,
        "raw_root": raw,
        "periods": ["2026Q1"],
        "output": tmp_path / "out.csv",
    }


def test_combines_without_rewriting_acquisition_dates(input_files):
    result = combine(**input_files)
    rows = read_observations(input_files["output"])
    assert [(r.period, r.available_at.isoformat()) for r in rows] == [
        ("2025Q4", "2026-11-03"),
        ("2026Q1", "2026-11-04"),
    ]
    assert rows[1].sales_amount == 1000 and rows[1].store_count == 10
    assert result["api_rows_by_period"] == {"2026Q1": 1}
    assert result["output_sha256"] == file_sha256(input_files["output"])
    assert input_files["output"].with_suffix(".manifest.json").exists()
    with pytest.raises(FileExistsError):
        combine(**input_files)


def test_rejects_changed_api_csv_without_output(input_files):
    path = input_files["raw_root"] / "sales/seoul-api-sales-2026Q1.csv"
    path.write_text(path.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        combine(**input_files)
    assert not input_files["output"].exists()


def test_rejects_modified_historical_source(input_files):
    path = input_files["raw_root"] / "sales/historic.csv"
    path.write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        combine(**input_files)
    assert not input_files["output"].exists()


def test_rejects_invalid_or_duplicate_period(input_files):
    input_files["periods"] = ["2026Q1", "2026Q1"]
    with pytest.raises(ValueError, match="distinct"):
        combine(**input_files)


def test_rejects_duplicate_history_overlap(input_files):
    # The canonical snapshot can overlap API input only if its metadata and
    # digest are also internally consistent; even then combine must reject it.
    historical = input_files["historical_input"]
    old = read_observations(historical)[0]
    historical.unlink()
    write_observations([old.model_copy(update={"period": "2026Q1"})], historical)
    summary_path = input_files["historical_summary"]
    summary = json.loads(summary_path.read_text())
    summary["input_sha256"] = file_sha256(historical)
    save_json(summary_path, summary)
    with pytest.raises(ValueError, match="Duplicate scope"):
        combine(**input_files)
    assert not input_files["output"].exists()
