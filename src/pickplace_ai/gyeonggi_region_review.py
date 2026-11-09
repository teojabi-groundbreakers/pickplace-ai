"""Annotate existing Gyeonggi review rows with code dates; never infer a crosswalk."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_regions(path: Path) -> dict[str, tuple[date, date | None]]:
    """Read MOIS H file's CP949 fixed-byte fields (not the B or mix file)."""
    records = {}
    with path.open("rb") as stream:
        header = stream.readline()
        if not header.decode("cp949").startswith("행정동코드 "):
            raise ValueError("Not an administrative-dong H code file")
        for line_no, line in enumerate(stream, 2):
            line = line.rstrip(b"\r\n")
            if len(line) != 300 or line[10:11] != b" " or line[112:113] != b" ":
                raise ValueError(f"Unexpected H fixed-width format at line {line_no}")
            line.decode("cp949")
            code = line[:10].decode("ascii")
            start = line[104:112].decode("ascii")
            end = line[113:121].decode("ascii").strip()
            if not re.fullmatch(r"\d{10}", code) or code in records:
                raise ValueError(f"Invalid or duplicate H code at line {line_no}")
            try:
                created = datetime.strptime(start, "%Y%m%d").date()
                retired = datetime.strptime(end, "%Y%m%d").date() if end else None
            except ValueError as exc:
                raise ValueError(f"Invalid H date at line {line_no}") from exc
            records[code] = (created, retired)
    if not records:
        raise ValueError("Empty H code file")
    return records


def month_status(ym: str, dates: tuple[date, date | None] | None) -> str:
    if not re.fullmatch(r"\d{4}(0[1-9]|1[0-2])", ym) or ym[:4] == "0000":
        return "invalid_month"
    if dates is None:
        return "code_not_found"
    created, retired = dates
    if retired and retired < created:
        return "invalid_code_date_range"
    month = int(ym)
    start_month = created.year * 100 + created.month
    end_month = retired.year * 100 + retired.month if retired else None
    if month < start_month:
        return "before_creation_month"
    if end_month is not None and month > end_month:
        return "after_retirement_month"
    if month == start_month or month == end_month:
        return "boundary_month"
    return "active_entire_month"


def reference_status(dates: tuple[date, date | None] | None, reference: date) -> str:
    if dates is None:
        return "code_not_found"
    created, retired = dates
    if retired and retired < created:
        return "invalid_code_date_range"
    if reference < created:
        return "not_yet_created"
    if retired and reference >= retired:
        return "retired"
    return "active"


def prepare(
    *, candidates: Path, excluded: Path, h_file: Path, reference: date, output_dir: Path
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output already exists: {output_dir}")
    sources = {
        "candidates": candidates,
        "excluded": excluded,
        "administrative_codebook": h_file,
    }
    fingerprints = {key: sha256(path) for key, path in sources.items()}
    regions = read_regions(h_file)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    by_month = Counter()
    excluded_reasons = Counter()
    excluded_reason_statuses = Counter()
    code_counts: dict[tuple[str, str], set[str]] = {}
    output_fields = (
        "source_row",
        "existing_partition",
        "기준년월",
        "행정동코드",
        "기존_제외사유",
        "sales_month_status",
        "reference_date_status",
        "code_created",
        "code_retired",
    )
    with tempfile.TemporaryDirectory(prefix=".region-review-", dir=output_dir.parent) as temp:
        staged = Path(temp) / "output"
        staged.mkdir()
        with (staged / "region-status.csv").open("x", encoding="utf-8", newline="") as dest:
            writer = csv.writer(dest)
            writer.writerow(output_fields)
            for partition, source in (
                ("candidate", candidates),
                ("excluded", excluded),
            ):
                with source.open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    required = {"source_row", "기준년월", "행정동코드"}
                    if partition == "excluded":
                        required.add("reason")
                    if not reader.fieldnames or not required.issubset(reader.fieldnames):
                        raise ValueError(f"Unexpected {partition} CSV headers")
                    if len(reader.fieldnames) != len(set(reader.fieldnames)):
                        raise ValueError(f"Duplicate {partition} CSV headers")
                    for row in reader:
                        if None in row or any(value is None for value in row.values()):
                            raise ValueError(f"Malformed {partition} row at line {reader.line_num}")
                        ym, code = row["기준년월"], row["행정동코드"]
                        dates = regions.get(code)
                        monthly = month_status(ym, dates)
                        current = reference_status(dates, reference)
                        reason = row.get("reason", "")
                        if partition == "excluded" and not reason:
                            raise ValueError("Excluded row has no original reason")
                        if partition == "excluded":
                            excluded_reasons.update(reason.split(";"))
                            excluded_reason_statuses[(reason, monthly, current)] += 1
                        writer.writerow(
                            (
                                row["source_row"],
                                partition,
                                ym,
                                code,
                                reason,
                                monthly,
                                current,
                                dates[0].isoformat() if dates else "",
                                dates[1].isoformat() if dates and dates[1] else "",
                            )
                        )
                        counts[(partition, monthly, current)] += 1
                        by_month[(partition, ym, monthly)] += 1
                        code_counts.setdefault((partition, monthly), set()).add(code)
        if fingerprints != {key: sha256(path) for key, path in sources.items()}:
            raise ValueError("Source files changed while being read")
        summary = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "inputs": {
                key: {"filename": path.name, "sha256": fingerprints[key]}
                for key, path in sources.items()
            },
            "codebook_reference_date": reference.isoformat(),
            "codebook_invalid_date_range_codes": sorted(
                code
                for code, (created, retired) in regions.items()
                if retired and retired < created
            ),
            "purpose": "review_only_not_training_or_prediction",
            "assumption": "Some source records may use later code dates; this is not verified.",
            "source_code_system_verified_h10": False,
            "historical_boundaries_verified": False,
            "publication_dates_verified": False,
            "counts": [
                {
                    "partition": p,
                    "sales_month_status": m,
                    "reference_date_status": r,
                    "rows": count,
                }
                for (p, m, r), count in sorted(counts.items())
            ],
            "unique_codes_by_partition_and_month_status": [
                {"partition": p, "sales_month_status": m, "unique_codes": len(codes)}
                for (p, m), codes in sorted(code_counts.items())
            ],
            "counts_by_month": [
                {"partition": p, "sales_month": ym, "status": m, "rows": n}
                for (p, ym, m), n in sorted(by_month.items())
            ],
            "original_excluded_reasons": dict(sorted(excluded_reasons.items())),
            "excluded_reason_statuses": [
                {
                    "original_reason": reason,
                    "sales_month_status": monthly,
                    "reference_date_status": current,
                    "rows": count,
                }
                for (reason, monthly, current), count in sorted(excluded_reason_statuses.items())
            ],
            "output_sha256": sha256(staged / "region-status.csv"),
            "warnings": [
                "This output annotates the existing partition; excluded records are not restored.",
                "Created/retired month means boundary_month, not a confirmed mismatch.",
                "Earlier-than-creation records are not proof of retroactive reaggregation.",
                "No spatial crosswalk or H10 equivalence is inferred.",
                "The reference date is the supplied codebook date, not a sales creation date.",
            ],
        }
        (staged / "manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Output already exists: {output_dir}")
        os.rename(staged, output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--h-file", type=Path, required=True)
    parser.add_argument("--reference-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(
        candidates=args.candidates,
        excluded=args.excluded,
        h_file=args.h_file,
        reference=args.reference_date,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {"purpose": result["purpose"], "counts": result["counts"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
