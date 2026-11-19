"""합성 CSV로 월말 주민등록인구 원문의 날짜·계층·정수 조건을 확인한다."""

import csv
from datetime import date

import pytest

from scripts.seoul_fit_population import read_mois_population

HEADER = ["행정구역", "2025년06월_총인구수"]
ROWS = [
    ["전국 (0000000000)", "5,000"],
    ["서울특별시 (1100000000)", "3,000"],
    ["서울특별시 종로구 (1111000000)", "3,000"],
    ["서울특별시 종로구 가상1동 (1111051500)", "1,000"],
    ["서울특별시 종로구 가상2동 (1111053000)", "2,000"],
    ["부산광역시 (2600000000)", "2,000"],
]


def write_fixture(path, rows=None, header=None, encoding="utf-8-sig"):
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER if header is None else header)
        writer.writerows(ROWS if rows is None else rows)
    return path


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp949"])
def test_extracts_dongs_with_full_h10_without_changing_source(tmp_path, encoding):
    path = write_fixture(tmp_path / "fictional.csv", encoding=encoding)
    before = path.read_bytes()
    assert read_mois_population(path, date(2025, 6, 30)) == {
        "1111051500": 1000,
        "1111053000": 2000,
    }
    assert read_mois_population(path, "2025-06-30") == {
        "1111051500": 1000,
        "1111053000": 2000,
    }
    assert path.read_bytes() == before


def test_original_row_order_does_not_define_parent_relationship(tmp_path):
    path = write_fixture(tmp_path / "fictional.csv", rows=list(reversed(ROWS)))
    assert len(read_mois_population(path, date(2025, 6, 30))) == 2


@pytest.mark.parametrize("reference", [date(2025, 6, 29), "2025-06", "20250630"])
def test_rejects_non_month_end_or_ambiguous_reference(tmp_path, reference):
    path = write_fixture(tmp_path / "fictional.csv")
    with pytest.raises(ValueError):
        read_mois_population(path, reference)


def test_month_mismatch_multi_month_and_duplicate_headers_are_rejected(tmp_path):
    for header in (
        ["행정구역", "2025년05월_총인구수"],
        ["행정구역", "2025년06월_총인구수", "2025년05월_세대수"],
        ["행정구역", "2025년06월_총인구수", "2025년06월_총인구수"],
    ):
        path = write_fixture(tmp_path / "fictional.csv", header=header)
        with pytest.raises(ValueError):
            read_mois_population(path, date(2025, 6, 30))


@pytest.mark.parametrize("value", ["", "1,00", "-1", "1.0", "1e3", "NaN", "inf", "1_000", "+1"])
def test_rejects_missing_or_malformed_integer_population(tmp_path, value):
    rows = [row.copy() for row in ROWS]
    rows[3][1] = value
    path = write_fixture(tmp_path / "fictional.csv", rows=rows)
    with pytest.raises(ValueError, match="integer"):
        read_mois_population(path, date(2025, 6, 30))


@pytest.mark.parametrize("conflicting_total", ["1,000", "9,000"])
def test_duplicate_code_is_rejected_even_with_equal_values(tmp_path, conflicting_total):
    path = write_fixture(tmp_path / "fictional.csv", rows=[*ROWS, [ROWS[3][0], conflicting_total]])
    with pytest.raises(ValueError, match="Duplicate"):
        read_mois_population(path, date(2025, 6, 30))


@pytest.mark.parametrize(
    "label",
    [
        "서울특별시 종로구 가상1동 (11110515)",
        "서울특별시 종로구 가상1동 (1111051501)",
        "서울특별시 종로구 가상1동 (1111000000)",
        "서울특별시 종로구 (1111051500)",
        "부산광역시 종로구 가상1동 (1111051500)",
        "서울특별시 종로구 가상1동 (2611051500)",
    ],
)
def test_label_depth_or_suffix_cannot_turn_a_dong_into_aggregate(tmp_path, label):
    rows = [row.copy() for row in ROWS]
    rows[3][0] = label
    path = write_fixture(tmp_path / "fictional.csv", rows=rows)
    with pytest.raises(ValueError):
        read_mois_population(path, date(2025, 6, 30))


def test_dong_parent_name_and_aggregate_totals_are_verified(tmp_path):
    rows = [row.copy() for row in ROWS]
    rows[3][0] = "서울특별시 중구 가상1동 (1111051500)"
    path = write_fixture(tmp_path / "fictional.csv", rows=rows)
    with pytest.raises(ValueError, match="parent"):
        read_mois_population(path, date(2025, 6, 30))
    for index in (1, 2):
        rows = [row.copy() for row in ROWS]
        rows[index][1] = "3,001"
        write_fixture(path, rows=rows)
        with pytest.raises(ValueError, match="total"):
            read_mois_population(path, date(2025, 6, 30))


def test_missing_aggregate_and_missing_record_cannot_look_complete(tmp_path):
    for deleted in (1, 2, 3):
        rows = [row.copy() for index, row in enumerate(ROWS) if index != deleted]
        path = write_fixture(tmp_path / "fictional.csv", rows=rows)
        with pytest.raises(ValueError):
            read_mois_population(path, date(2025, 6, 30))
