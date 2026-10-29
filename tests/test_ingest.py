import csv
from datetime import date

import pytest

from pickplace_ai.ingest import ImportDataError, import_seoul


def write_csv(path, headers, rows, encoding="utf-8-sig"):
    with path.open("w", encoding=encoding, newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


ENGLISH = ["STDR_YYQU_CD", "ADSTRD_CD", "SVC_INDUTY_CD", "THSMON_SELNG_AMT"]
KOREAN = ["기준_년분기_코드", "행정동_코드", "서비스_업종_코드", "당월_매출_금액"]


@pytest.mark.parametrize("headers,encoding", [(ENGLISH, "utf-8-sig"), (KOREAN, "cp949")])
def test_import_preserves_code_and_sales_basis(tmp_path, headers, encoding):
    source = write_csv(
        tmp_path / "sales.csv", headers, [["20251", "00123456", "CS100010", "1,200"]], encoding
    )
    rows = import_seoul([source], [], date(2025, 5, 1))
    assert len(rows) == 1
    assert rows[0].region_code == "00123456"
    assert rows[0].region_code_system == "seoul-adstrd"
    assert rows[0].period == "2025Q1"
    assert rows[0].sales_amount == 1200
    assert rows[0].store_count is None
    assert rows[0].available_at == date(2025, 5, 1)
    assert rows[0].source_kind == "official"


def test_exact_join_uses_total_stores_not_sum(tmp_path):
    sales = write_csv(
        tmp_path / "sales.csv",
        ENGLISH,
        [
            ["20251", "11200670", "CS100010", "1000"],
            ["20252", "11200670", "CS100010", "1500"],
        ],
    )
    stores = write_csv(
        tmp_path / "stores.csv",
        [
            "STDR_YYQU_CD",
            "ADSTRD_CD",
            "SVC_INDUTY_CD",
            "SIMILR_INDUTY_STOR_CO",
            "STOR_CO",
            "FRC_STOR_CO",
            "OPBIZ_STOR_CO",
            "CLSBIZ_STOR_CO",
        ],
        [["20251", "11200670", "CS100010", "10", "7", "3", "2", "1"]],
    )
    rows = import_seoul([sales], [stores], date(2025, 8, 1))
    assert rows[0].store_count == 10
    assert rows[0].opening_count == 2
    assert rows[0].closing_count == 1
    assert rows[1].store_count is None


def test_duplicate_quarter_scope_across_files_rejected(tmp_path):
    sales = write_csv(tmp_path / "sales.csv", ENGLISH, [["20251", "11200670", "CS100010", "1000"]])
    with pytest.raises(ImportDataError, match="Duplicate sales key"):
        import_seoul([sales, sales], [], date(2025, 5, 1))


def test_duplicate_store_rows_rejected(tmp_path):
    sales = write_csv(tmp_path / "sales.csv", ENGLISH, [["20251", "11200670", "CS100010", "1000"]])
    stores = write_csv(
        tmp_path / "stores.csv",
        [*ENGLISH[:3], "SIMILR_INDUTY_STOR_CO"],
        [["20251", "11200670", "CS100010", "10"]] * 2,
    )
    with pytest.raises(ImportDataError, match="Duplicate stores key"):
        import_seoul([sales], [stores], date(2025, 5, 1))


@pytest.mark.parametrize("period", ["20250", "20255", "2025Q0", "2025Q5", "2025.0"])
def test_bad_period_is_not_coerced(tmp_path, period):
    sales = write_csv(tmp_path / "sales.csv", ENGLISH, [[period, "11200670", "CS100010", "1000"]])
    with pytest.raises(ImportDataError, match="quarter"):
        import_seoul([sales], [], date(2026, 1, 1))


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-1", "12,34", "비공개"])
def test_invalid_numbers_are_not_zero_filled(tmp_path, amount):
    sales = write_csv(tmp_path / "sales.csv", ENGLISH, [["20251", "11200670", "CS100010", amount]])
    with pytest.raises(ImportDataError):
        import_seoul([sales], [], date(2025, 5, 1))


def test_missing_sales_stays_missing_and_zero_stays_zero(tmp_path):
    sales = write_csv(
        tmp_path / "sales.csv",
        ENGLISH,
        [
            ["20251", "11200670", "CS100010", ""],
            ["20252", "11200670", "CS100010", "0"],
        ],
    )
    rows = import_seoul([sales], [], date(2025, 8, 1))
    assert rows[0].sales_amount is None
    assert rows[1].sales_amount == 0


def test_early_publication_rejected(tmp_path):
    sales = write_csv(tmp_path / "sales.csv", ENGLISH, [["20251", "11200670", "CS100010", "1000"]])
    with pytest.raises(ImportDataError, match="available_at"):
        import_seoul([sales], [], date(2025, 2, 1))


def test_industry_filter_is_exact(tmp_path):
    sales = write_csv(
        tmp_path / "sales.csv",
        ENGLISH,
        [
            ["20251", "11200670", "CS100010", "1000"],
            ["20251", "11200670", "CS100001", "800"],
        ],
    )
    rows = import_seoul([sales], [], date(2025, 5, 1), industry_code="CS100010")
    assert [row.industry_code for row in rows] == ["CS100010"]


def test_unknown_industry_filter_reports_no_data(tmp_path):
    sales = write_csv(tmp_path / "sales.csv", ENGLISH, [["20251", "11200670", "CS100010", "1000"]])
    with pytest.raises(ImportDataError, match="No sales rows"):
        import_seoul([sales], [], date(2025, 5, 1), industry_code="coffee")


def test_ambiguous_header_rejected(tmp_path):
    sales = write_csv(
        tmp_path / "sales.csv",
        [*ENGLISH, "당월_매출_금액"],
        [["20251", "11200670", "CS100010", "1000", "1000"]],
    )
    with pytest.raises(ImportDataError, match="ambiguous aliases"):
        import_seoul([sales], [], date(2025, 5, 1))


def test_fractional_store_count_rejected(tmp_path):
    sales = write_csv(tmp_path / "sales.csv", ENGLISH, [["20251", "11200670", "CS100010", "1000"]])
    stores = write_csv(
        tmp_path / "stores.csv",
        [*ENGLISH[:3], "SIMILR_INDUTY_STOR_CO"],
        [["20251", "11200670", "CS100010", "2.5"]],
    )
    with pytest.raises(ImportDataError, match="integer count"):
        import_seoul([sales], [stores], date(2025, 5, 1))
