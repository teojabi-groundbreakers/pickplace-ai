"""서울 점포·주민·행정동 경계를 동일 분기로 결합해 비교표를 생성한다."""

import argparse
import csv
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pickplace_ai.artifacts import file_sha256
from pickplace_ai.fit_reference import CODE_SYSTEM, KIND, ROW_FIELDS, FitReference
from pickplace_ai.seoul_stats_ingest import MODE
from scripts.seoul_fit_population import read_mois_population

STORES_URL = "https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do"
SERVICE_CODES = ("CS100010", "CS100008", "CS100005")


def _read_dated(path, fieldnames, reference_date):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != ("reference_date", *fieldnames):
            raise ValueError(f"Unexpected columns in {path.name}")
        for row in reader:
            if (
                None in row
                or any(value is None or not value.strip() for value in row.values())
                or date.fromisoformat(row["reference_date"]) != reference_date
            ):
                raise ValueError(f"Invalid row or reference date in {path.name}")
            yield row


def _read_areas(path, reference_date):
    areas = {}
    for row in _read_dated(path, ("sgis_code", "area_m2"), reference_date):
        sgis, area = row["sgis_code"], float(row["area_m2"])
        if len(sgis) != 8 or not sgis.isascii() or not sgis.isdigit() or sgis in areas:
            raise ValueError("Expected unique original SGIS area codes")
        if not math.isfinite(area) or area <= 0:
            raise ValueError("Expected a positive area in square metres")
        areas[sgis] = area
    return areas


def _read_mapping(path, reference_date):
    mapping = {}
    h10_codes, sgis_codes = set(), set()
    for row in _read_dated(path, ("sgis_code", "h10_code", "seoul_code"), reference_date):
        sgis, h10, seoul = row["sgis_code"], row["h10_code"], row["seoul_code"]
        for code, length, prefix in ((sgis, 8, "11"), (h10, 10, "11"), (seoul, 8, "11")):
            if (
                len(code) != length
                or not code.isascii()
                or not code.isdigit()
                or not code.startswith(prefix)
            ):
                raise ValueError("Invalid original Seoul region code")
        if seoul in mapping or sgis in sgis_codes or h10 in h10_codes:
            raise ValueError("Expected a one-to-one region lookup")
        sgis_codes.add(sgis)
        h10_codes.add(h10)
        mapping[seoul] = (sgis, h10)
    return mapping


def build(
    snapshot: Path,
    areas_path: Path,
    population_path: Path,
    mapping_path: Path,
    evidence_path: Path,
    output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError("Fit reference output already exists")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    reference_date = date.fromisoformat(evidence["reference_date"])
    as_of_dates = []
    input_paths = {"area": areas_path, "population": population_path, "mapping": mapping_path}
    sources = evidence["sources"]
    if set(sources) != input_paths.keys():
        raise ValueError("Expected area, population and mapping evidence")
    for key, path in input_paths.items():
        source = sources[key]
        if date.fromisoformat(source["reference_date"]) != reference_date:
            raise ValueError(f"{key} uses a different reference date")
        if file_sha256(path) != source["sha256"]:
            raise ValueError(f"{key} file does not match the recorded SHA-256")
        acquired = date.fromisoformat(source["acquired_at"])
        if acquired < reference_date:
            raise ValueError(f"{key} was acquired before the reference date")
        as_of_dates.append(acquired)

    database = snapshot / "snapshot.sqlite3"
    snapshot_manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    snapshot_hash = file_sha256(database)
    quarter = f"{reference_date.year}Q{(reference_date.month - 1) // 3 + 1}"
    if (
        snapshot_manifest["mode"] != MODE
        or snapshot_manifest["database_sha256"] != snapshot_hash
        or quarter not in snapshot_manifest["datasets"]["stores"]["periods"]
    ):
        raise ValueError("Expected a matching Seoul stores snapshot covering the reference quarter")
    stores_acquired = (
        datetime.fromisoformat(snapshot_manifest["created_at_utc"])
        .astimezone(ZoneInfo("Asia/Seoul"))
        .date()
    )
    areas = _read_areas(areas_path, reference_date)
    population = read_mois_population(population_path, reference_date)
    mapping = _read_mapping(mapping_path, reference_date)
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        stores = connection.execute(
            """
            SELECT region_code, industry_code, store_count
            FROM stores
            WHERE period = ? AND industry_code IN (?, ?, ?)
            ORDER BY industry_code, region_code
            """,
            (quarter, *SERVICE_CODES),
        ).fetchall()

    selected, excluded = [], Counter()
    for seoul, industry, count in stores:
        if count is None or count <= 0:
            excluded["nonpositive_or_missing_store_count"] += 1
        elif seoul not in mapping:
            excluded["region_mapping_missing"] += 1
        else:
            sgis, h10 = mapping[seoul]
            if sgis not in areas or h10 not in population:
                excluded["area_missing" if sgis not in areas else "population_missing"] += 1
            else:
                selected.append((seoul, industry, count, population[h10], areas[sgis]))

    # 읽기 도중 원천이 바뀌면 만들어진 비교표를 발행하지 않는다.
    for key, path in input_paths.items():
        if file_sha256(path) != sources[key]["sha256"]:
            raise ValueError(f"{key} file changed while building")
    if file_sha256(database) != snapshot_hash:
        raise ValueError("Stores snapshot changed while building")

    metadata = {
        "kind": KIND,
        "source_kind": "official",
        "region_code_system": CODE_SYSTEM,
        "base_period": quarter,
        "stores_period": quarter,
        "reference_date": reference_date.isoformat(),
        "available_at": max(*as_of_dates, stores_acquired).isoformat(),
        **{f"{key}_reference_date": sources[key]["reference_date"] for key in sources},
        "sources": {
            **sources,
            "stores": {
                "url": STORES_URL,
                "sha256": snapshot_hash,
                "acquired_at": stores_acquired.isoformat(),
            },
        },
        "candidate_rows": len(stores),
        "row_count": len(selected),
        "cohort_sizes": dict(sorted(Counter(row[1] for row in selected).items())),
        "exclusions": dict(sorted(excluded.items())),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        stage = Path(directory)
        reference = stage / "reference.csv"
        with reference.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(ROW_FIELDS)
            writer.writerows(selected)
        metadata["reference_sha256"] = file_sha256(reference)
        (stage / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        FitReference(stage)
        output.mkdir()
        try:
            for name in ("reference.csv", "manifest.json"):
                os.link(stage / name, output / name)
        except BaseException:
            for name in ("reference.csv", "manifest.json"):
                (output / name).unlink(missing_ok=True)
            output.rmdir()
            raise
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "areas", "population", "mapping", "evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        args.snapshot, args.areas, args.population, args.mapping, args.evidence, args.output
    )
    print(f"{result['base_period']} fit reference: {result['cohort_sizes']}")
    print(f"Excluded rows: {result['exclusions']}")


if __name__ == "__main__":
    main()
