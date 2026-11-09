"""Review-only preparation uses fictional data and preserves all suspect rows."""

import csv
import json
import zipfile
from pathlib import Path

import pytest

from pickplace_ai import gyeonggi_candidates as prep


def codebook(path: Path) -> None:
    shared = (
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        "<si><t>업종 중분류 코드</t></si><si><t>업종 중분류 코드명</t></si>"
        "<si><t>Q13</t></si><si><t>커피/음료</t></si></sst>"
    )
    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData><row r="2"><c r="B2" t="s"><v>0</v></c>'
        '<c r="C2" t="s"><v>1</v></c></row>'
        '<row r="3"><c r="B3" t="s"><v>2</v></c>'
        '<c r="C3" t="s"><v>3</v></c></row></sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w") as book:
        book.writestr("xl/sharedStrings.xml", shared)
        book.writestr("xl/worksheets/sheet1.xml", sheet)


def source(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="cp949", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(prep.HEADERS)
        writer.writerows(rows)


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run(raw: Path, book: Path, output: Path) -> dict:
    region_file = raw.parent / "regions.csv"
    if not region_file.exists():
        with region_file.open("w", encoding="cp949", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["행정구역", "2026년10월_계_총인구수"])
            writer.writerow(["서울특별시 (1100000000)", "1"])
            writer.writerow(["서울특별시 종로구 청운효자동(1111051500)", "1"])
    return prep.prepare(
        source=raw,
        industries=book,
        current_regions=region_file,
        region_reference_month="2026-10",
        output_dir=output,
    )


def test_all_duplicate_candidates_are_quarantined_before_value_checks(tmp_path):
    book = tmp_path / "codes.xlsx"
    raw = tmp_path / "source.csv"
    output = tmp_path / "result"
    codebook(book)
    source(
        raw,
        [
            ["202401", "1111051500", "Q13", "10"],
            ["202401", "1111051500", "Q13", "10.0"],
            ["202402", "1111051500", "Q13", "30"],
            ["202402", "1111051500", "Q13", "-1"],
            ["202403", "1111051500", "Q13", "0"],
            ["202403", "1111051500", "TO", "40"],
            ["202413", "1111051500", "Q13", "55"],
            ["202404", "1111051500", "Q13", "NaN"],
            ["202405", "123", "X", ""],
            ["202406", "1111053000", "Q13", "5"],
            ["000001", "1111051500", "Q13", "5"],
        ],
    )
    with raw.open("ab") as blank:
        blank.write(b"\r\n")
    before = raw.read_bytes()
    report = run(raw, book, output)
    candidates, excluded = read(output / "candidates.csv"), read(output / "excluded.csv")
    assert raw.read_bytes() == before
    assert report["counts"] == {"candidate": 1, "excluded": 10}
    assert report["blank_records"] == 1
    assert report["duplicate_groups"] == 2
    assert report["duplicate_rows"] == 4
    assert report["excluded_reason_counts"]["duplicate_key"] == 4
    assert report["official_h10_mapping_verified"] is False
    assert report["purpose"] == "review_only_not_training_or_prediction"
    assert candidates[0]["매출금액"] == "0"
    assert candidates[0]["source_region_code_system"].endswith("unverified")
    assert len(excluded) == 10
    assert next(x for x in excluded if x["기준년월"] == "000001")["reason"] == "invalid_month"
    assert next(x for x in excluded if x["행정동코드"] == "1111053000")["reason"] == (
        "current_reference_absent"
    )
    negative = next(x for x in excluded if x["매출금액"] == "-1")
    assert negative["reason"] == "duplicate_key;invalid_sales_amount"
    multiple = next(x for x in excluded if x["행정동코드"] == "123")
    assert multiple["reason"] == (
        "invalid_source_region_code;industry_not_in_latest_codebook;invalid_sales_amount"
    )
    assert json.loads((output / "manifest.json").read_text())["available_at"] is None


def test_invalid_header_and_truncated_row_never_publish_output(tmp_path):
    book, raw, output = tmp_path / "codes.xlsx", tmp_path / "data.csv", tmp_path / "out"
    codebook(book)
    raw.write_text("다른헤더\n", encoding="cp949")
    with pytest.raises(ValueError, match="headers"):
        run(raw, book, output)
    assert not output.exists()
    source(raw, [["202401", "1111051500", "Q13"]])
    with pytest.raises(ValueError, match="Malformed"):
        run(raw, book, output)
    assert not output.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    book, raw, output = tmp_path / "codes.xlsx", tmp_path / "data.csv", tmp_path / "out"
    codebook(book)
    source(raw, [["202401", "1111051500", "Q13", "1"]])
    output.mkdir()
    (output / "keep").write_text("unchanged")
    with pytest.raises(FileExistsError):
        run(raw, book, output)
    assert (output / "keep").read_text() == "unchanged"


def test_dangling_output_symlink_is_not_overwritten(tmp_path):
    book, raw, output = tmp_path / "codes.xlsx", tmp_path / "data.csv", tmp_path / "out"
    codebook(book)
    source(raw, [["202401", "1111051500", "Q13", "1"]])
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        run(raw, book, output)
    assert output.is_symlink()


@pytest.mark.parametrize(
    "rows,month",
    [
        ([["서울 (1111051500)", "10"]], "2026-09"),
        ([["서울 (1111051500)", "10"], ["서울 (1111051500)", "11"]], "2026-10"),
        ([["서울 (1111051500)"]], "2026-10"),
    ],
)
def test_invalid_current_reference_never_publishes(tmp_path, rows, month):
    raw, book = tmp_path / "source.csv", tmp_path / "codes.xlsx"
    region_file, out = tmp_path / "regions.csv", tmp_path / "out"
    codebook(book)
    source(raw, [["202410", "1111051500", "Q13", "1"]])
    with region_file.open("w", encoding="cp949", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["행정구역", "2026년10월_계_총인구수"])
        writer.writerows(rows)
    with pytest.raises(ValueError):
        prep.prepare(
            source=raw,
            industries=book,
            current_regions=region_file,
            region_reference_month=month,
            output_dir=out,
        )
    assert not out.exists()


def test_changed_input_is_rejected_without_output(tmp_path, monkeypatch):
    book, raw, output = tmp_path / "codes.xlsx", tmp_path / "data.csv", tmp_path / "out"
    codebook(book)
    source(raw, [["202401", "1111051500", "Q13", "1"]])
    real_sha = prep._sha
    calls = 0

    def changed_sha(path):
        nonlocal calls
        calls += 1
        return "changed" if calls == 4 else real_sha(path)

    monkeypatch.setattr(prep, "_sha", changed_sha)
    with pytest.raises(ValueError, match="changed"):
        run(raw, book, output)
    assert not output.exists()


def test_codebook_must_not_contain_duplicate_codes(tmp_path):
    book = tmp_path / "codes.xlsx"
    codebook(book)
    # The duplicate file content is constructed separately; no workbook is modified.
    with zipfile.ZipFile(book) as src:
        sheet = src.read("xl/worksheets/sheet1.xml").decode()
        shared = src.read("xl/sharedStrings.xml")
    duplicate = tmp_path / "duplicate.xlsx"
    sheet = sheet.replace(
        "</sheetData>",
        '<row r="4"><c r="B4" t="s"><v>2</v></c><c r="C4" t="s"><v>3</v></c></row></sheetData>',
    )
    with zipfile.ZipFile(duplicate, "w") as dst:
        dst.writestr("xl/sharedStrings.xml", shared)
        dst.writestr("xl/worksheets/sheet1.xml", sheet)
    with pytest.raises(ValueError, match="duplicate"):
        prep.load_industries(duplicate)
