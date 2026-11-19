"""Prepare nationwide source snapshots without guessing code mappings."""

from __future__ import annotations

import calendar
import codecs
import csv
import hashlib
import json
import re
import sqlite3
import tempfile
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

STORE_COLUMNS = (
    "region_code_system",
    "region_code",
    "region_name",
    "sido_name",
    "sigungu_name",
    "industry_code_system",
    "industry_code",
    "industry_name",
    "store_count",
)
POPULATION_COLUMNS = (
    "region_code_system",
    "region_code",
    "region_name",
    "resident_population",
)
STORE_REGION_SYSTEM = "sbiz-admin-8"
POPULATION_REGION_SYSTEM = "mois-resident-admin-10"
INDUSTRY_SYSTEM = "sbiz-commercial-2026-09"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path) -> dict:
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _encoding(path: Path) -> str:
    with path.open("rb") as source:
        sample = source.read(65536)
    try:
        codecs.getincrementaldecoder("utf-8-sig")().decode(sample, final=False)
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp949"


def _reader(path: Path):
    # Callers own and close this stream.
    return path.open("r", encoding=_encoding(path), newline="")


def _check_fields(reader: csv.DictReader, required: set[str], name: str) -> None:
    fields = reader.fieldnames or []
    if len(set(fields)) != len(fields) or not required.issubset(fields):
        raise ValueError(f"{name}: 누락되거나 중복된 열이 있습니다")


def _value(row: dict, key: str, name: str, line: int) -> str:
    value = row.get(key)
    if value is None or not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}:{line}: {key} 값이 없습니다")
    return value.strip()


def _aggregate_stores(paths: list[Path]) -> tuple[list[dict], int, list[dict]]:
    required = {
        "상가업소번호",
        "행정동코드",
        "행정동명",
        "시도명",
        "시군구명",
        "상권업종소분류코드",
        "상권업종소분류명",
    }
    sources = [_source(path) for path in paths]
    if len({x["sha256"] for x in sources}) != len(sources):
        raise ValueError("동일한 상가 원본 파일이 중복되었습니다")
    counts: Counter = Counter()
    regions: dict[str, tuple[str, str, str]] = {}
    industries: dict[str, str] = {}
    row_count = 0
    # A temporary on-disk index avoids storing millions of identifiers in RAM.
    with (
        tempfile.TemporaryDirectory(prefix="pickplace-store-ids-") as directory,
        closing(sqlite3.connect(str(Path(directory) / "ids.sqlite"))) as db,
        db,
    ):
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("CREATE TABLE ids (id TEXT PRIMARY KEY) WITHOUT ROWID")
        for path in paths:
            with _reader(path) as handle:
                reader = csv.DictReader(handle)
                _check_fields(reader, required, path.name)
                for line, row in enumerate(reader, start=2):
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"{path.name}:{line}: 열 수가 일치하지 않습니다")
                    identifier = _value(row, "상가업소번호", path.name, line)
                    region = _value(row, "행정동코드", path.name, line)
                    industry = _value(row, "상권업종소분류코드", path.name, line)
                    if not re.fullmatch(r"[0-9]{8}", region):
                        raise ValueError(f"{path.name}:{line}: 지원하지 않는 행정동 코드")
                    if not re.fullmatch(r"[A-Z0-9]{6}", industry):
                        raise ValueError(f"{path.name}:{line}: 지원하지 않는 업종 코드")
                    region_names = tuple(
                        _value(row, k, path.name, line) for k in ("행정동명", "시도명", "시군구명")
                    )
                    industry_name = _value(row, "상권업종소분류명", path.name, line)
                    if region in regions and regions[region] != region_names:
                        raise ValueError(
                            f"{path.name}:{line}: 지역 코드의 명칭이 일치하지 않습니다"
                        )
                    if industry in industries and industries[industry] != industry_name:
                        raise ValueError(
                            f"{path.name}:{line}: 업종 코드의 명칭이 일치하지 않습니다"
                        )
                    regions[region] = region_names
                    industries[industry] = industry_name
                    try:
                        db.execute("INSERT INTO ids VALUES (?)", (identifier,))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"{path.name}:{line}: 중복 상가업소번호") from error
                    counts[region, industry] += 1
                    row_count += 1
    if any(_source(path) != source for path, source in zip(paths, sources, strict=True)):
        raise ValueError("상가 원본이 처리 도중 변경되었습니다")
    if not row_count:
        raise ValueError("상가 데이터가 비어 있습니다")
    result = []
    for (region, industry), count in sorted(counts.items()):
        region_name, sido_name, sigungu_name = regions[region]
        result.append(
            dict(
                zip(
                    STORE_COLUMNS,
                    (
                        STORE_REGION_SYSTEM,
                        region,
                        region_name,
                        sido_name,
                        sigungu_name,
                        INDUSTRY_SYSTEM,
                        industry,
                        industries[industry],
                        count,
                    ),
                    strict=True,
                )
            )
        )
    return result, row_count, sources


def _population(path: Path, month: str) -> tuple[list[dict], dict, str]:
    try:
        year, mon = (int(part) for part in month.split("-"))
        last = calendar.monthrange(year, mon)[1]
    except (ValueError, TypeError, calendar.IllegalMonthError) as error:
        raise ValueError("인구 기준월은 YYYY-MM이어야 합니다") from error
    if not re.fullmatch(r"[0-9]{4}-(0[1-9]|1[0-2])", month):
        raise ValueError("인구 기준월은 YYYY-MM이어야 합니다")
    field = f"{year}년{mon:02d}월_계_총인구수"
    result = []
    seen: set[str] = set()
    source = _source(path)
    with _reader(path) as handle:
        reader = csv.DictReader(handle)
        _check_fields(reader, {"행정구역", field}, path.name)
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}:{line}: 열 수가 일치하지 않습니다")
            region = _value(row, "행정구역", path.name, line)
            match = re.fullmatch(r"(.+?)\s*\(([0-9]{10})\)", region)
            if not match:
                raise ValueError(f"{path.name}:{line}: 행정기관 코드 형식 오류")
            name, code = match.group(1).strip(), match.group(2)
            if code in seen:
                raise ValueError(f"{path.name}:{line}: 중복 행정기관 코드")
            seen.add(code)
            # The published 주민등록 code has an all-zero 읍면동 portion for upper-level totals.
            # Keep each remaining original 10-digit code intact; do not infer store-code mapping.
            if code[5:] == "00000":
                continue
            number = row.get(field)
            if number is None:
                raise ValueError(f"{path.name}:{line}: 열 수가 일치하지 않습니다")
            number = number.strip().replace(",", "")
            if number and (not number.isascii() or not number.isdigit()):
                raise ValueError(f"{path.name}:{line}: 인구 수 형식 오류")
            result.append(
                dict(
                    zip(
                        POPULATION_COLUMNS,
                        (POPULATION_REGION_SYSTEM, code, name, number),
                        strict=True,
                    )
                )
            )
    if _source(path) != source:
        raise ValueError("인구 원본이 처리 도중 변경되었습니다")
    if not result:
        raise ValueError("읍면동 인구 데이터가 없습니다")
    result.sort(key=lambda item: item["region_code"])
    return result, source, f"{year:04d}-{mon:02d}-{last:02d}"


def prepare_nationwide(
    *,
    stores: list[Path],
    population: Path,
    output_dir: Path,
    store_reference_label: str,
    population_reference_month: str,
) -> dict:
    """Aggregate two source datasets independently; provenance never implies an unverified join."""
    if output_dir.exists():
        raise FileExistsError(f"출력 경로가 이미 존재합니다: {output_dir}")
    if not stores or len({str(p.resolve()) for p in stores}) != len(stores):
        raise ValueError("상가 파일이 없거나 같은 경로가 중복되었습니다")
    if store_reference_label != "20260930":
        raise ValueError("현재 상가 업종 코드체계는 20260930 자료만 지원합니다")
    store_rows, total, store_sources = _aggregate_stores(stores)
    population_rows, population_source, population_date = _population(
        population, population_reference_month
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, columns, rows in (
        ("stores.csv", STORE_COLUMNS, store_rows),
        ("population.csv", POPULATION_COLUMNS, population_rows),
    ):
        with (output_dir / filename).open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "stores": {
            "reference_label": store_reference_label,
            "reference_label_verified": False,
            "national_coverage_verified": False,
            "observed_date": None,
            "available_at": None,
            "acquired_at": None,
            "source_files": store_sources,
            "row_count": total,
            "group_count": len(store_rows),
            "output_sha256": sha256_file(output_dir / "stores.csv"),
        },
        "population": {
            "reference_month": population_reference_month,
            "observed_date": population_date,
            "available_at": None,
            "acquired_at": None,
            "source_file": population_source,
            "row_count": len(population_rows),
            "output_sha256": sha256_file(output_dir / "population.csv"),
        },
        "crosswalk_verified": False,
    }
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return manifest
