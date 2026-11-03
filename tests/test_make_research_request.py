import csv
import json
from datetime import date

import pytest

from pickplace_ai.data import write_observations
from pickplace_ai.periods import shift_quarter
from pickplace_ai.schemas import Observation, PredictionRequest
from scripts.make_research_request import main, make_request


@pytest.fixture
def snapshot(tmp_path):
    """Invented test rows matching the official request shape; not real data."""
    base = "2025Q3"
    rows = [
        Observation(
            region_code="11111111",
            region_code_system="seoul-adstrd",
            industry_code="CS100010",
            period=shift_quarter(base, delta),
            available_at=date(2026, 11, 3),
            source_kind="official",
            source_dataset="invented-test-fixture",
            sales_amount=100_000 + (delta + 4) * 1000,
            store_count=10,
            opening_count=1,
            closing_count=1,
        )
        for delta in range(-4, 1)
    ]
    path = tmp_path / "input.csv"
    write_observations(rows, path)
    return path


def args(input_path, output_path):
    return [
        "--input",
        str(input_path),
        "--region-code",
        "11111111",
        "--industry-code",
        "CS100010",
        "--base-period",
        "2025Q3",
        "--as-of",
        "2026-11-03",
        "--output",
        str(output_path),
    ]


def test_cli_generates_valid_five_quarter_learned_request(snapshot, tmp_path):
    output = tmp_path / "reports" / "request.json"
    assert main([*args(snapshot, output), "--model-variant", "learned"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload.pop("model_variant") == "learned"
    request = PredictionRequest.model_validate(payload)
    assert [row.period for row in request.history] == [
        "2024Q3",
        "2024Q4",
        "2025Q1",
        "2025Q2",
        "2025Q3",
    ]
    assert all(row.available_at == date(2026, 11, 3) for row in request.history)


def test_duplicate_selected_quarter_fails(snapshot):
    with snapshot.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    with snapshot.open("a", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(rows[1])
    with pytest.raises(ValueError, match="Duplicate selected quarter"):
        make_request(snapshot, "11111111", "CS100010", "2025Q3", date(2026, 11, 3), "selected")


def test_missing_quarter_fails(snapshot):
    with snapshot.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    with snapshot.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows[:-1])
    with pytest.raises(ValueError, match="Missing required history quarters"):
        make_request(snapshot, "11111111", "CS100010", "2025Q3", date(2026, 11, 3), "selected")


@pytest.mark.parametrize("region", ["123", "22111111", "1111111111", "11abcdef"])
def test_non_seoul_region_code_is_rejected(snapshot, region):
    with pytest.raises(ValueError, match="eight-digit Seoul"):
        make_request(snapshot, region, "CS100010", "2025Q3", date(2026, 11, 3), "selected")


def test_asof_before_data_publication_fails(snapshot):
    with pytest.raises(ValueError, match="not available"):
        make_request(snapshot, "11111111", "CS100010", "2025Q3", date(2026, 1, 1), "selected")


def test_future_target_cannot_be_replayed(snapshot):
    with pytest.raises(ValueError, match="completed target quarter"):
        make_request(snapshot, "11111111", "CS100010", "2025Q3", date(2025, 10, 1), "selected")


def test_existing_output_is_not_overwritten(snapshot, tmp_path):
    output = tmp_path / "request.json"
    output.write_text("preserve", encoding="utf-8")
    assert main(args(snapshot, output)) == 2
    assert output.read_text(encoding="utf-8") == "preserve"


def test_invalid_industry_reports_missing_history(snapshot):
    with pytest.raises(ValueError, match="Missing required history quarters"):
        make_request(snapshot, "11111111", "CS100999", "2025Q3", date(2026, 11, 3), "selected")
