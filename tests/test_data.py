import pytest

from pickplace_ai.data import quality_report, read_observations, write_observations
from pickplace_ai.demo import synthetic_observations
from pickplace_ai.schemas import Observation


def test_csv_roundtrip_keeps_code_as_string_and_source_kind(tmp_path):
    rows = synthetic_observations(regions=1, quarters=10)
    rows = [row.model_copy(update={"region_code": "00123456"}) for row in rows]
    path = tmp_path / "data.csv"
    write_observations(rows, path)
    actual = read_observations(path)
    assert actual[0].region_code == "00123456"
    assert actual[0].source_kind == "synthetic"
    assert len(actual) == 10


def test_duplicate_dataset_and_source_kind_mix_are_rejected(tmp_path):
    rows = synthetic_observations(regions=1, quarters=10)
    with pytest.raises(ValueError, match="Duplicate"):
        write_observations(rows + [rows[0]], tmp_path / "duplicates.csv")
    rows[0] = rows[0].model_copy(update={"source_kind": "official"})
    with pytest.raises(ValueError, match="mix"):
        quality_report(rows)


def test_existing_dataset_is_not_overwritten(tmp_path):
    rows = synthetic_observations(regions=1, quarters=10)
    path = tmp_path / "data.csv"
    write_observations(rows, path)
    with pytest.raises(FileExistsError):
        write_observations(rows, path)


def test_nonfinite_sales_is_rejected():
    row = synthetic_observations(regions=1, quarters=10)[0].model_dump()
    row["sales_amount"] = float("inf")
    with pytest.raises(ValueError):
        Observation(**row)


def test_quality_report_counts_missing_quarters():
    rows = synthetic_observations(regions=1, quarters=10)
    report = quality_report(rows[:3] + rows[4:])
    assert report["histories"][0]["missing_quarters"] == 1


def test_duplicate_header_is_rejected_before_overwriting_a_code(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("region_code,region_code\n00123456,99123456\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate canonical CSV header"):
        read_observations(path)


def test_inconsistent_csv_row_length_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("region_code,period\n00123456\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inconsistent number"):
        read_observations(path)
