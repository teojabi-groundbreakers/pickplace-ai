import csv
import json
from datetime import date

import pytest

from scripts.make_near_term_request import main, make_near_term_request


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshot.csv"
    fields = [
        "region_code",
        "region_code_system",
        "industry_code",
        "period",
        "available_at",
        "source_kind",
        "source_dataset",
        "sales_amount",
        "store_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for period in ("2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"):
            writer.writerow(
                {
                    "region_code": "11110515",
                    "region_code_system": "seoul-adstrd",
                    "industry_code": "CS100010",
                    "period": period,
                    "available_at": "2026-11-03" if period.startswith("2025") else "2026-11-04",
                    "source_kind": "official",
                    "source_dataset": "fixture",
                    "sales_amount": 1000,
                    "store_count": 10,
                }
            )
    return path


def test_generates_five_quarters_without_old_variant(snapshot):
    result = make_near_term_request(snapshot, "11110515", "CS100010", "2026Q4", date(2026, 11, 4))
    assert result["target_period"] == "2026Q4"
    assert result["base_period"] == "2026Q2"
    assert [r["period"] for r in result["history"]] == [
        "2025Q2",
        "2025Q3",
        "2025Q4",
        "2026Q1",
        "2026Q2",
    ]
    assert "model_variant" not in result


def test_command_creates_output_but_never_overwrites(snapshot, tmp_path):
    output = tmp_path / "request.json"
    args = [
        "--input",
        str(snapshot),
        "--region-code",
        "11110515",
        "--industry-code",
        "CS100010",
        "--target-period",
        "2026Q3",
        "--as-of",
        "2026-11-04",
        "--output",
        str(output),
    ]
    assert main(args) == 0
    before = output.read_bytes()
    assert json.loads(before)["target_period"] == "2026Q3"
    assert main(args) == 2
    assert output.read_bytes() == before


def test_refuses_unknown_target_or_scope(snapshot):
    with pytest.raises(ValueError, match="Target period"):
        make_near_term_request(snapshot, "11110515", "CS100010", "2027Q1", date(2026, 11, 4))
    with pytest.raises(ValueError, match="eight-digit"):
        make_near_term_request(snapshot, "1111051500", "CS100010", "2026Q3", date(2026, 11, 4))
