"""Synthetic fixtures only; no claim of a live API integration test."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "fetch_kosis_population", Path(__file__).parents[1] / "scripts/fetch_kosis_population.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

STATS = "test/101/DT_1B04005N/2/1/20261109_1"
ROW = {
    "ORG_ID": "101",
    "TBL_ID": "DT_1B04005N",
    "PRD_SE": "M",
    "PRD_DE": "202501",
    "C1": "001",
    "C1_NM": "행정동",
    "C2": "0",
    "ITM_ID": "T1",
    "DT": "123",
}


def test_writes_raw_values_and_sha256(tmp_path):
    target = tmp_path / "source.csv"
    count, digest = mod.fetch_month("202501", STATS, "private", target, request=lambda *args: [ROW])
    assert count == 1
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    metadata = json.loads(target.with_suffix(".csv.metadata.json").read_text())
    assert metadata["csv_sha256"] == digest
    assert metadata["coverage_verified"] is False
    assert metadata["publication_date"] is None
    assert "private" not in str(metadata)
    with target.open(encoding="utf-8-sig", newline="") as f:
        assert list(csv.DictReader(f)) == [ROW]


@pytest.mark.parametrize(
    "response",
    [
        {"err": "31", "errMsg": "too many"},
        [],
        [ROW, ROW],
        [{**ROW, "TBL_ID": "OTHER"}],
        [{**ROW, "PRD_DE": "202502"}],
        [{**ROW, "C1": ""}],
        [{key: value for key, value in ROW.items() if key != "DT"}],
        [ROW] * 40000,
    ],
)
def test_invalid_response_never_saves(tmp_path, response):
    target = tmp_path / "source.csv"
    with pytest.raises(mod.FetchError):
        mod.fetch_month("202501", STATS, "private", target, request=lambda *args: response)
    assert not target.exists()


def test_never_overwrites(tmp_path):
    target = tmp_path / "source.csv"
    target.write_text("original")
    with pytest.raises(FileExistsError):
        mod.fetch_month("202501", STATS, "private", target, request=lambda *args: [ROW])
    assert target.read_text() == "original"


def test_rejects_invalid_month_and_wrong_table(tmp_path):
    with pytest.raises(mod.FetchError):
        mod.fetch_month("202513", STATS, "private", tmp_path / "a")
    with pytest.raises(mod.FetchError):
        mod.fetch_month("202501", "test/101/OTHER/1", "private", tmp_path / "a")
