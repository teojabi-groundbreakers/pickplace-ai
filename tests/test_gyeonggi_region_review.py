"""Fictional dates and rows: original partition is never altered."""

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from pickplace_ai import gyeonggi_region_review as review


def h_codebook(path: Path) -> None:
    rows = [b"\xc7\xe0\xc1\xa4\xb5\xbf\xc4\xda\xb5\xe5 " + b" " * 289]
    for code, start, end in [
        ("4111111111", "20240115", ""),
        ("4111111112", "20100101", "20240615"),
    ]:
        value = f"{code} ".encode() + b" " * 93 + start.encode() + b" " + end.encode().ljust(8)
        rows.append(value.ljust(300, b" "))
    path.write_bytes(b"\r\n".join(rows) + b"\r\n")


def rows(path, records, excluded=False):
    fields = ["source_row", "기준년월", "행정동코드"]
    if excluded:
        fields.append("reason")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(records)


@pytest.mark.parametrize(
    "ym,dates,expected",
    [
        ("202312", (date(2024, 1, 15), None), "before_creation_month"),
        ("202401", (date(2024, 1, 15), None), "boundary_month"),
        ("202402", (date(2024, 1, 15), None), "active_entire_month"),
        ("202406", (date(2010, 1, 1), date(2024, 6, 15)), "boundary_month"),
        ("202407", (date(2010, 1, 1), date(2024, 6, 15)), "after_retirement_month"),
        ("202401", None, "code_not_found"),
        ("202413", None, "invalid_month"),
        ("202401", (date(2024, 2, 1), date(2024, 1, 1)), "invalid_code_date_range"),
    ],
)
def test_month_boundaries(ym, dates, expected):
    assert review.month_status(ym, dates) == expected


def test_annotates_without_changing_existing_decisions_or_input(tmp_path):
    h, candidate, excluded = (tmp_path / x for x in ("h", "candidate.csv", "excluded.csv"))
    h_codebook(h)
    rows(candidate, [["2", "202312", "4111111111"], ["3", "202401", "4111111111"]])
    rows(
        excluded,
        [
            ["4", "202405", "4111111112", "current_reference_absent"],
            ["5", "202407", "4111111112", "current_reference_absent;duplicate_key"],
            ["6", "202401", "9999999999", "current_reference_absent"],
        ],
        True,
    )
    hashes = {p: review.sha256(p) for p in (h, candidate, excluded)}
    out = tmp_path / "out"
    result = review.prepare(
        candidates=candidate,
        excluded=excluded,
        h_file=h,
        reference=date(2026, 10, 26),
        output_dir=out,
    )
    with (out / "region-status.csv").open(encoding="utf-8", newline="") as stream:
        result_rows = list(csv.DictReader(stream))
    assert len(result_rows) == 5
    assert result_rows[0]["sales_month_status"] == "before_creation_month"
    assert result_rows[1]["sales_month_status"] == "boundary_month"
    assert result_rows[2]["existing_partition"] == "excluded"
    assert result_rows[2]["기존_제외사유"] == "current_reference_absent"
    assert result_rows[2]["sales_month_status"] == "active_entire_month"
    assert result_rows[2]["reference_date_status"] == "retired"
    assert result_rows[3]["sales_month_status"] == "after_retirement_month"
    assert result_rows[4]["sales_month_status"] == "code_not_found"
    assert result["original_excluded_reasons"] == {
        "current_reference_absent": 3,
        "duplicate_key": 1,
    }
    assert result["purpose"] == "review_only_not_training_or_prediction"
    assert result["source_code_system_verified_h10"] is False
    assert json.loads((out / "manifest.json").read_text())["output_sha256"] == review.sha256(
        out / "region-status.csv"
    )
    assert {p: review.sha256(p) for p in hashes} == hashes
    with pytest.raises(FileExistsError):
        review.prepare(
            candidates=candidate,
            excluded=excluded,
            h_file=h,
            reference=date(2026, 10, 26),
            output_dir=out,
        )


def test_bad_codebook_and_output_do_not_publish(tmp_path):
    h, candidate, excluded = (tmp_path / x for x in ("h", "candidate.csv", "excluded.csv"))
    h.write_text("invalid", encoding="utf-8")
    rows(candidate, [["2", "202312", "4111111111"]])
    rows(excluded, [], True)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        review.prepare(
            candidates=candidate,
            excluded=excluded,
            h_file=h,
            reference=date(2026, 10, 26),
            output_dir=tmp_path / "bad",
        )
    assert not (tmp_path / "bad").exists()
