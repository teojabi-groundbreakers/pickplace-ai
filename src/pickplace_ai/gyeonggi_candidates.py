"""Conservative review-only preparation of Gyeonggi's monthly card data.

No geographic mapping, historical publication date, units or model labels are inferred.
All duplicate keys are quarantined before checking row values.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import zipfile
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from xml.etree import ElementTree

HEADERS = ("기준년월", "행정동코드", "중분류업종코드", "매출금액")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_industries(path: Path) -> dict[str, str]:
    """Read the source workbook's checked B/C columns without editing it."""
    with zipfile.ZipFile(path) as book:
        shared = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
        strings = ["".join(x.itertext()) for x in shared.findall("s:si", NS)]
        sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))
    pairs: list[tuple[str, str]] = []
    for row in sheet.findall("s:sheetData/s:row", NS):
        cells = {}
        for cell in row.findall("s:c", NS):
            ref = cell.get("r", "")
            col = re.match(r"[A-Z]+", ref)
            if not col or col.group() not in {"B", "C"}:
                continue
            if cell.find("s:f", NS) is not None or cell.get("t") != "s":
                raise ValueError("Industry codebook must contain shared text values, not formulas")
            value = cell.find("s:v", NS)
            if value is None or value.text is None:
                raise ValueError("Missing industry cell")
            cells[col.group()] = strings[int(value.text)].strip()
        if cells:
            if set(cells) != {"B", "C"}:
                raise ValueError("Incomplete industry codebook row")
            pairs.append((cells["B"], cells["C"]))
    if not pairs or pairs[0] != ("업종 중분류 코드", "업종 중분류 코드명"):
        raise ValueError("Unexpected industry codebook header")
    codes: dict[str, str] = {}
    for code, name in pairs[1:]:
        if not re.fullmatch(r"[A-Z][0-9]{2}", code) or not name or code in codes:
            raise ValueError("Invalid or duplicate industry codebook value")
        codes[code] = name
    if not codes:
        raise ValueError("Empty industry codebook")
    return codes


def load_current_regions(path: Path, reference_month: str) -> set[str]:
    """Read original 10-digit lower-level codes from the explicitly dated population file."""
    if not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", reference_month):
        raise ValueError("Reference month must be YYYY-MM")
    year, month = reference_month.split("-")
    expected = f"{year}년{month}월_계_총인구수"
    codes: set[str] = set()
    seen: set[str] = set()
    with path.open(encoding="cp949", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not {"행정구역", expected} <= set(fields):
            raise ValueError("Invalid current region reference header or month")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed current region row {line}")
            match = re.fullmatch(r".+?\s*\(([0-9]{10})\)", row["행정구역"])
            if match is None:
                raise ValueError(f"Invalid current region code at row {line}")
            code = match.group(1)
            if code in seen:
                raise ValueError("Duplicate current region code")
            seen.add(code)
            if code[5:] != "00000":
                codes.add(code)
    if not codes:
        raise ValueError("Empty current localities reference")
    return codes


def _invalid(
    ym: str, region: str, industry: str, amount: str, codes: dict[str, str], current: set[str]
) -> str:
    reasons = []
    if not re.fullmatch(r"[0-9]{4}(0[1-9]|1[0-2])", ym) or ym[:4] == "0000":
        reasons.append("invalid_month")
    if not re.fullmatch(r"[0-9]{10}", region):
        reasons.append("invalid_source_region_code")
    elif region not in current:
        reasons.append("current_reference_absent")
    if industry not in codes:
        reasons.append("industry_not_in_latest_codebook")
    try:
        value = Decimal(amount)
    except InvalidOperation:
        reasons.append("invalid_sales_amount")
    else:
        if not value.is_finite() or value < 0:
            reasons.append("invalid_sales_amount")
    return ";".join(reasons)


def prepare(
    *,
    source: Path,
    industries: Path,
    current_regions: Path,
    region_reference_month: str,
    output_dir: Path,
) -> dict:
    """Write independent candidates and excluded rows; never overwrite a target."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output already exists: {output_dir}")
    if not all(p.is_file() for p in (source, industries, current_regions)):
        raise ValueError("Source CSV, industry codebook and current region reference are required")
    hashes = {
        "source_sha256": _sha(source),
        "industry_sha256": _sha(industries),
        "region_sha256": _sha(current_regions),
    }
    codebook = load_industries(industries)
    region_codes = load_current_regions(current_regions, region_reference_month)
    row_count = 0
    blank_records = 0
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".pickplace-gyeonggi-review-", dir=output_dir.parent
    ) as directory:
        with closing(sqlite3.connect(str(Path(directory) / "rows.sqlite"))) as db, db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute(
                "CREATE TABLE rows (line INTEGER PRIMARY KEY, ym TEXT, region TEXT, "
                "industry TEXT, amount TEXT, invalid TEXT)"
            )
            batch = []
            with source.open(encoding="cp949", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                if tuple(next(reader, ())) != HEADERS:
                    raise ValueError("Unexpected source headers or encoding")
                for row in reader:
                    line = reader.line_num
                    if not row:
                        blank_records += 1
                        continue
                    if len(row) != 4:
                        raise ValueError(f"Malformed CSV at row {line}")
                    ym, region, industry, amount = row
                    reason = _invalid(ym, region, industry, amount, codebook, region_codes)
                    batch.append((line, ym, region, industry, amount, reason))
                    row_count += 1
                    if len(batch) == 10000:
                        db.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?)", batch)
                        batch.clear()
                if batch:
                    db.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?)", batch)
            if not row_count:
                raise ValueError("Empty source data")
            db.execute("CREATE INDEX rows_key ON rows(ym, region, industry)")
            db.execute(
                "CREATE TABLE duplicates AS SELECT ym, region, industry, count(*) AS n "
                "FROM rows GROUP BY ym, region, industry HAVING count(*) > 1"
            )
            db.execute("CREATE UNIQUE INDEX duplicates_key ON duplicates(ym, region, industry)")
            duplicate_groups, duplicate_rows = db.execute(
                "SELECT count(*), coalesce(sum(n),0) FROM duplicates"
            ).fetchone()
            if hashes != {
                "source_sha256": _sha(source),
                "industry_sha256": _sha(industries),
                "region_sha256": _sha(current_regions),
            }:
                raise ValueError("Source files changed while being read")
            staged = Path(directory) / "output"
            staged.mkdir()
            counts: Counter[str] = Counter()
            reasons_count: Counter[str] = Counter()
            with (
                (staged / "candidates.csv").open("x", encoding="utf-8", newline="") as valid,
                (staged / "excluded.csv").open("x", encoding="utf-8", newline="") as excluded,
            ):
                writer = csv.writer(valid)
                quarantined = csv.writer(excluded)
                writer.writerow(
                    (
                        "source_row",
                        "source_region_code_system",
                        *HEADERS,
                        "industry_name_current_reference",
                    )
                )
                quarantined.writerow(("source_row", *HEADERS, "reason"))
                query = (
                    "SELECT r.line,r.ym,r.region,r.industry,r.amount,r.invalid,d.n "
                    "FROM rows r LEFT JOIN duplicates d USING(ym,region,industry) "
                    "ORDER BY r.line"
                )
                for line, ym, region, industry, amount, invalid, occurrences in db.execute(query):
                    reasons = (["duplicate_key"] if occurrences else []) + (
                        invalid.split(";") if invalid else []
                    )
                    if reasons:
                        counts["excluded"] += 1
                        reasons_count.update(reasons)
                        quarantined.writerow(
                            (line, ym, region, industry, amount, ";".join(reasons))
                        )
                    else:
                        counts["candidate"] += 1
                        writer.writerow(
                            (
                                line,
                                "gyeonggi-admdong-10-unverified",
                                ym,
                                region,
                                industry,
                                amount,
                                codebook[industry],
                            )
                        )
            manifest = {
                "schema_version": 1,
                "generated_at": datetime.now(UTC).isoformat(),
                "source": {"filename": source.name, "sha256": hashes["source_sha256"]},
                "industry_codebook": {
                    "filename": industries.name,
                    "sha256": hashes["industry_sha256"],
                    "count": len(codebook),
                },
                "current_region_reference": {
                    "filename": current_regions.name,
                    "sha256": hashes["region_sha256"],
                    "reference_month": region_reference_month,
                    "locality_count": len(region_codes),
                },
                "rows": row_count,
                "blank_records": blank_records,
                "counts": dict(sorted(counts.items())),
                "excluded_reason_counts": dict(sorted(reasons_count.items())),
                "duplicate_groups": duplicate_groups,
                "duplicate_rows": duplicate_rows,
                "candidate_sha256": _sha(staged / "candidates.csv"),
                "excluded_sha256": _sha(staged / "excluded.csv"),
                "official_h10_mapping_verified": False,
                "historical_boundary_verified": False,
                "industry_history_verified": False,
                "sales_units_verified": False,
                "publication_date_verified": False,
                "available_at": None,
                "source_kind": "user_supplied_unverified",
                "purpose": "review_only_not_training_or_prediction",
                "warnings": [
                    "All rows of every duplicate key are excluded, including equal values.",
                    "No crosswalk is inferred and 10-digit source codes are not certified H10.",
                    "Current reference match is not historical boundary validation; "
                    "absent codes are not classified as errors or abolished.",
                    "Industry names are current-reference labels, not historical verification.",
                    "Candidate rows require official code, unit and temporal verification.",
                ],
            }
            with (staged / "manifest.json").open("x", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
                f.write("\n")
            if output_dir.exists() or output_dir.is_symlink():
                raise FileExistsError(f"Output already exists: {output_dir}")
            os.rename(staged, output_dir)
            return manifest
