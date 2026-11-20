"""합성 원문으로 경기 시군·일반구·출장소 계층과 월말 인구를 검증한다."""

import csv
from datetime import date

import pytest

from scripts.gyeonggi_fit_population import read_mois_gyeonggi_population

HEADER = ["행정구역", "2026년06월_총인구수", "2026년06월_세대수"]
ROWS = [
    ["전국 (0000000000)", "501", "20"],
    ["서울특별시 (1100000000)", "1", "1"],
    ["경기도 (4100000000)", "500", "19"],
    ["경기도 수원시 (4111000000)", "300", "10"],
    ["경기도 수원시 장안구 (4111100000)", "100", "3"],
    ["경기도 수원시 장안구 가상1동 (4111156000)", "100", "3"],
    ["경기도 수원시 권선구 (4111300000)", "200", "7"],
    ["경기도 수원시 권선구 가상2동 (4111352000)", "0", "0"],
    ["경기도 수원시 권선구 가상3동 (4111355000)", "200", "7"],
    ["경기도 남양주시 (4136000000)", "200", "9"],
    ["경기도 남양주시 와부읍 (4136025000)", "50", "2"],
    ["경기도 남양주시 화도읍동부출장소 (4136025700)", "60", "3"],
    ["경기도 남양주시 화도읍남부출장소 (4136025800)", "90", "4"],
    ["경기도 풍양출장소 (4136200000)", "0", "0"],
]


def write_fixture(tmp_path, *, rows=None, header=None, encoding="utf-8-sig"):
    path = tmp_path / "fictional-population.csv"
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER if header is None else header)
        writer.writerows(ROWS if rows is None else rows)
    return path


@pytest.mark.parametrize("encoding", ["utf-8-sig", "cp949"])
def test_exact_h10_includes_local_branches_and_zero_but_excludes_aggregates(tmp_path, encoding):
    path = write_fixture(tmp_path, encoding=encoding)
    before = path.read_bytes()
    result = read_mois_gyeonggi_population(path, date(2026, 6, 30))
    assert result == {
        "4111156000": 100,
        "4111352000": 0,
        "4111355000": 200,
        "4136025000": 50,
        "4136025700": 60,
        "4136025800": 90,
    }
    assert list(result) == sorted(result)
    assert sum(result.values()) == 500
    assert read_mois_gyeonggi_population(path, "2026-06-30") == result
    assert path.read_bytes() == before


def test_parent_validation_is_independent_of_row_order(tmp_path):
    path = write_fixture(tmp_path, rows=list(reversed(ROWS)))
    assert len(read_mois_gyeonggi_population(path, "2026-06-30")) == 6


@pytest.mark.parametrize("reference", [date(2026, 6, 29), "2026-06", "20260630", "2026-02-30"])
def test_only_unambiguous_month_end_is_accepted(tmp_path, reference):
    path = write_fixture(tmp_path)
    with pytest.raises(ValueError):
        read_mois_gyeonggi_population(path, reference)


@pytest.mark.parametrize(
    "header",
    [
        ["행정구역", "2026년05월_총인구수", "2026년05월_세대수"],
        ["행정구역", "2026년06월_총인구수", "2026년05월_세대수"],
        ["행정구역", "2026년06월_총인구수", "2026년06월_총인구수"],
        ["행정구역", "2026년06월_계_총인구수", "2026년06월_세대수"],
        ["지역명", "2026년06월_총인구수", "2026년06월_세대수"],
    ],
)
def test_month_and_statmonth_header_are_verified(tmp_path, header):
    path = write_fixture(tmp_path, header=header)
    with pytest.raises(ValueError):
        read_mois_gyeonggi_population(path, "2026-06-30")


@pytest.mark.parametrize("value", ["", "1,00", "-1", "1.0", "1e3", "NaN", "inf", "+1", "01"])
def test_missing_malformed_or_negative_population_is_rejected(tmp_path, value):
    rows = [row.copy() for row in ROWS]
    rows[5][1] = value
    path = write_fixture(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="integer"):
        read_mois_gyeonggi_population(path, "2026-06-30")


def test_valid_comma_grouping_is_allowed(tmp_path):
    rows = [row.copy() for row in ROWS]
    for index in (0, 2, 3, 4, 5):
        rows[index][1] = f"{int(rows[index][1]) + 1000:,}"
    path = write_fixture(tmp_path, rows=rows)
    result = read_mois_gyeonggi_population(path, "2026-06-30")
    assert result["4111156000"] == 1100


@pytest.mark.parametrize("value", ["100", "101"])
def test_duplicate_h10_is_rejected_even_if_values_match(tmp_path, value):
    path = write_fixture(tmp_path, rows=[*ROWS, [ROWS[5][0], value, "3"]])
    with pytest.raises(ValueError, match="Duplicate"):
        read_mois_gyeonggi_population(path, "2026-06-30")


def test_same_name_with_different_h10_is_rejected(tmp_path):
    rows = [*ROWS, ["경기도 수원시 장안구 가상1동 (4111157000)", "0", "0"]]
    path = write_fixture(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="Conflicting"):
        read_mois_gyeonggi_population(path, "2026-06-30")


@pytest.mark.parametrize(
    "label",
    [
        "경기도 수원시 장안구 가상1동 (41111560)",
        "경기도 수원시 장안구 가상1동 (4111156001)",
        "경기도 수원시 장안구 가상1동 (4111100000)",
        "경기도 수원시 장안구 (4111156000)",
        "경기도 수원시 장안구 가상1동 (2611156000)",
        "부산광역시 수원시 장안구 가상1동 (4111156000)",
        "경기도 수원시 가상1동 (4111156000)",
        "경기도 수원시 장안구 가상1리 (4111156000)",
    ],
)
def test_code_suffix_name_depth_and_province_prefix_must_agree(tmp_path, label):
    rows = [row.copy() for row in ROWS]
    rows[5][0] = label
    path = write_fixture(tmp_path, rows=rows)
    with pytest.raises(ValueError):
        read_mois_gyeonggi_population(path, "2026-06-30")


@pytest.mark.parametrize(
    ("index", "label"),
    [
        (5, "경기도 수원시 권선구 가상1동 (4111156000)"),
        (4, "경기도 안양시 장안구 (4111100000)"),
        (3, "경기도 수원시 장안구 (4111000000)"),
        (2, "경기도 수원시 (4100000000)"),
    ],
)
def test_each_parent_code_and_name_are_checked(tmp_path, index, label):
    rows = [row.copy() for row in ROWS]
    rows[index][0] = label
    path = write_fixture(tmp_path, rows=rows)
    with pytest.raises(ValueError):
        read_mois_gyeonggi_population(path, "2026-06-30")


@pytest.mark.parametrize("index", [2, 3, 4, 6, 9, 13])
def test_inconsistent_aggregate_total_is_rejected_at_each_stratum(tmp_path, index):
    rows = [row.copy() for row in ROWS]
    rows[index][1] = str(int(rows[index][1]) + 1)
    path = write_fixture(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="total"):
        read_mois_gyeonggi_population(path, "2026-06-30")


@pytest.mark.parametrize("index", [2, 3, 4, 5, 9, 10, 11])
def test_missing_parent_or_positive_child_cannot_look_complete(tmp_path, index):
    path = write_fixture(tmp_path, rows=[row for i, row in enumerate(ROWS) if i != index])
    with pytest.raises(ValueError):
        read_mois_gyeonggi_population(path, "2026-06-30")


def test_orphan_upper_branch_cannot_be_silently_discarded(tmp_path):
    rows = [*ROWS, ["경기도 가상출장소 (4199200000)", "0", "0"]]
    path = write_fixture(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="parent"):
        read_mois_gyeonggi_population(path, "2026-06-30")


def test_blank_rows_empty_files_and_malformed_row_lengths_are_rejected(tmp_path):
    for rows in ([*ROWS, []], [*ROWS, ["경기도"]]):
        path = write_fixture(tmp_path, rows=rows)
        with pytest.raises(ValueError):
            read_mois_gyeonggi_population(path, "2026-06-30")
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        read_mois_gyeonggi_population(path, "2026-06-30")
