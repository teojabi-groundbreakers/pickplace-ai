"""Prepare a read-only, source-preserving Seoul descriptive statistics snapshot."""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from pickplace_ai.ingest import ImportDataError, _columns, _number, _period, _read_csv

MODE = "seoul_descriptive_research_only"
TOTAL_FIELDS = {
    "resident": ("TOT_REPOP_CO", "총_상주인구_수"),
    "floating": ("TOT_FLPOP_CO", "총_유동인구_수"),
}
FIELD_GROUPS = {
    "sales": ("sales_amount",),
    "stores": ("store_count", "opening_count", "closing_count"),
    "resident": ("total",),
    "floating": ("total",),
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _columns_for(kind: str, headers: list[str], path: Path) -> dict[str, str]:
    if kind in ("sales", "stores"):
        required = ("period", "region_code", "industry_code", *FIELD_GROUPS[kind])
        return _columns(headers, required, path)
    columns = _columns(headers, ("period", "region_code"), path)
    found = [name for name in TOTAL_FIELDS[kind] if name in headers]
    if len(found) != 1:
        raise ImportDataError(f"{path}: missing or ambiguous {kind} total column")
    columns["total"] = found[0]
    return columns


def _schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE sales (
            period TEXT NOT NULL, region_code TEXT NOT NULL, industry_code TEXT NOT NULL,
            sales_amount INTEGER, PRIMARY KEY (period, region_code, industry_code)
        ) WITHOUT ROWID;
        CREATE TABLE stores (
            period TEXT NOT NULL, region_code TEXT NOT NULL, industry_code TEXT NOT NULL,
            store_count INTEGER, opening_count INTEGER, closing_count INTEGER,
            PRIMARY KEY (period, region_code, industry_code)
        ) WITHOUT ROWID;
        CREATE TABLE resident (
            period TEXT NOT NULL, region_code TEXT NOT NULL, total INTEGER,
            PRIMARY KEY (period, region_code)
        ) WITHOUT ROWID;
        CREATE TABLE floating (
            period TEXT NOT NULL, region_code TEXT NOT NULL, total INTEGER,
            PRIMARY KEY (period, region_code)
        ) WITHOUT ROWID;
        """
    )


def _insert_file(connection: sqlite3.Connection, kind: str, path: Path) -> tuple[int, set[str]]:
    before = _digest(path)
    count = 0
    periods: set[str] = set()
    names = ["period", "region_code"]
    if kind in ("sales", "stores"):
        names.append("industry_code")
    names.extend(FIELD_GROUPS[kind])
    sql = f"INSERT INTO {kind} ({','.join(names)}) VALUES ({','.join('?' for _ in names)})"
    with _read_csv(path) as (headers, rows):
        columns = _columns_for(kind, headers, path)
        for line_number, row in rows:
            try:
                period = _period(row[columns["period"]])
                region = row[columns["region_code"]]
                if not re.fullmatch(r"11\d{6}", region):
                    raise ImportDataError("Expected unmodified eight-digit Seoul region code")
                values = [period, region]
                if kind in ("sales", "stores"):
                    industry = row[columns["industry_code"]]
                    if not re.fullmatch(r"CS\d{6}", industry):
                        raise ImportDataError("Expected unmodified Seoul CS industry code")
                    values.append(industry)
                for field in FIELD_GROUPS[kind]:
                    values.append(_number(row[columns[field]], integer=True))
                connection.execute(sql, values)
                count += 1
                periods.add(period)
            except (ValueError, OverflowError, sqlite3.IntegrityError) as exc:
                raise ImportDataError(
                    f"{path.name}:{line_number}: invalid or duplicate row"
                ) from exc
    if not count:
        raise ImportDataError(f"{path.name}: source file contains no rows")
    if before != _digest(path):
        raise ImportDataError(f"{path.name}: source changed while processing")
    return count, periods


def prepare_snapshot(
    sales: list[Path], stores: list[Path], resident: Path, floating: Path, output: Path
) -> dict:
    """Create independent tables; retain nulls and fail on duplicate keys and changed files."""
    groups = {
        "sales": [Path(path) for path in sales],
        "stores": [Path(path) for path in stores],
        "resident": [Path(resident)],
        "floating": [Path(floating)],
    }
    if not sales or not stores:
        raise ImportDataError("At least one sales and store file is required")
    all_paths = [path.resolve(strict=True) for paths in groups.values() for path in paths]
    if len(set(all_paths)) != len(all_paths):
        raise ImportDataError("The same source file was supplied more than once")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".seoul-snapshot-", dir=output.parent))
    reserved = False
    try:
        database = stage / "snapshot.sqlite3"
        datasets = {}
        with closing(sqlite3.connect(database)) as connection, connection:
            _schema(connection)
            for kind, paths in groups.items():
                count = 0
                periods: set[str] = set()
                sources = []
                for path in paths:
                    source_hash = _digest(path)
                    rows, present = _insert_file(connection, kind, path)
                    if source_hash != _digest(path):
                        raise ImportDataError(f"{path.name}: source changed while processing")
                    count += rows
                    periods.update(present)
                    sources.append({"name": path.name, "sha256": source_hash})
                datasets[kind] = {
                    "row_count": count,
                    "periods": sorted(periods),
                    "source_files": sources,
                    "publication_date_verified": False,
                }
            # Earlier inputs can change while later files are being parsed.
            for kind, paths in groups.items():
                for path, recorded in zip(paths, datasets[kind]["source_files"], strict=True):
                    if _digest(path) != recorded["sha256"]:
                        raise ImportDataError(f"{path.name}: source changed while processing")
        metadata = {
            "schema_version": 1,
            "mode": MODE,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "database_sha256": _digest(database),
            "datasets": datasets,
        }
        (stage / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Create the output directory exclusively; no path can be replaced.
        output.mkdir()
        reserved = True
        for name in ("snapshot.sqlite3", "manifest.json"):
            os.link(stage / name, output / name)
        return metadata
    except BaseException:
        if reserved:
            for name in ("snapshot.sqlite3", "manifest.json"):
                (output / name).unlink(missing_ok=True)
            try:
                output.rmdir()
            except OSError:
                pass
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)
