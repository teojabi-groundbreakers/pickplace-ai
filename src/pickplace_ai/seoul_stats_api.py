"""Read-only Seoul source indicators; quarterly labels are not monthly observations."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
MODE = "seoul_descriptive_research_only"
DATASETS = {
    "sales": ("sales_amount",),
    "stores": ("store_count", "opening_count", "closing_count"),
    "resident": ("total",),
    "floating": ("total",),
}
KEYS = {
    "sales": ("period", "region_code", "industry_code"),
    "stores": ("period", "region_code", "industry_code"),
    "resident": ("period", "region_code"),
    "floating": ("period", "region_code"),
}
REGION_SYSTEM = "seoul-adstrd"
INDUSTRY_SYSTEM = "seoul-service-industry"
PERIOD_PATTERN = r"^[0-9]{4}Q[1-4]$"


class SnapshotError(ValueError):
    """Unusable research snapshot."""


@dataclass(frozen=True)
class SeoulStatsSettings:
    data_dir: Path
    api_key: str

    @classmethod
    def from_env(cls) -> SeoulStatsSettings:
        return cls(
            data_dir=Path(
                os.environ.get("PICKPLACE_SEOUL_STATS_DIR", "data/processed/seoul/stats")
            ),
            api_key=os.environ.get("PICKPLACE_SEOUL_STATS_API_KEY", ""),
        )


def _file(root: Path, name: str) -> Path:
    path = root / name
    try:
        if path.is_symlink():
            raise SnapshotError("Snapshot file must not be a symlink")
        resolved = path.resolve(strict=True)
        if resolved.parent != root or not resolved.is_file():
            raise SnapshotError("Invalid snapshot path")
        return resolved
    except OSError as exc:
        raise SnapshotError("Snapshot file unavailable") from exc


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _connection(path: Path) -> sqlite3.Connection:
    # URI built from a trusted resolved path; filename characters are URL encoded.
    db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA trusted_schema=OFF")
    return db


def _validate_metadata(raw: object) -> dict[str, Any]:
    if (
        not isinstance(raw, dict)
        or type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != 1
        or raw.get("mode") != MODE
        or not isinstance(raw.get("database_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", raw["database_sha256"])
        or not isinstance(raw.get("datasets"), dict)
        or set(raw["datasets"]) != set(DATASETS)
    ):
        raise SnapshotError("Invalid snapshot manifest")
    for source in raw["datasets"].values():
        if (
            not isinstance(source, dict)
            or type(source.get("row_count")) is not int
            or source["row_count"] < 0
            or source.get("publication_date_verified") is not False
            or not isinstance(source.get("periods"), list)
            or any(
                not isinstance(value, str) or not re.fullmatch(PERIOD_PATTERN, value)
                for value in source["periods"]
            )
            or len(set(source["periods"])) != len(source["periods"])
            or not isinstance(source.get("source_files"), list)
            or not source["source_files"]
        ):
            raise SnapshotError("Invalid dataset metadata")
        for source_file in source["source_files"]:
            if (
                not isinstance(source_file, dict)
                or not isinstance(source_file.get("name"), str)
                or not source_file["name"]
                or not isinstance(source_file.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", source_file["sha256"])
            ):
                raise SnapshotError("Invalid source metadata")
    return raw


class SeoulSnapshot:
    def __init__(self, data_dir: Path):
        try:
            root = data_dir.resolve(strict=True)
            if not root.is_dir():
                raise SnapshotError("Snapshot directory unavailable")
            manifest = _validate_metadata(
                json.loads(_file(root, "manifest.json").read_text("utf-8"))
            )
            database = _file(root, "snapshot.sqlite3")
            if _sha256(database) != manifest["database_sha256"]:
                raise SnapshotError("Snapshot digest mismatch")
            with closing(_connection(database)) as connection:
                if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise SnapshotError("Snapshot integrity check failed")
                for table, columns in DATASETS.items():
                    fields = connection.execute(f"PRAGMA table_info({table})").fetchall()
                    expected = (*KEYS[table], *columns)
                    if tuple(field["name"] for field in fields) != expected:
                        raise SnapshotError("Invalid dataset schema")
                    # Require uniqueness of the exact source key; SQL NULLs are forbidden in keys.
                    pk = sorted((field["pk"], field["name"]) for field in fields if field["pk"])
                    if tuple(name for _, name in pk) != KEYS[table]:
                        raise SnapshotError("Invalid dataset key")
                    for field in fields:
                        if field["name"] in KEYS[table] and not field["notnull"]:
                            raise SnapshotError("Dataset keys must be non-null")
                    row_count = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                    periods = {
                        row[0] for row in connection.execute(f"SELECT DISTINCT period FROM {table}")
                    }
                    if row_count != manifest["datasets"][table]["row_count"] or periods != set(
                        manifest["datasets"][table]["periods"]
                    ):
                        raise SnapshotError("Dataset count or periods mismatch")
            self.database = database
            file_stat = database.stat()
            self.file_identity = (
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_size,
                file_stat.st_mtime_ns,
            )
            self.manifest = manifest
        except (OSError, json.JSONDecodeError, UnicodeError, sqlite3.Error) as exc:
            raise SnapshotError("Could not validate Seoul snapshot") from exc

    def query(self, region: str, industry: str, start: str, end: str) -> dict[str, dict]:
        result: dict[str, dict] = {}
        try:
            file_stat = self.database.stat()
            identity = (
                file_stat.st_dev,
                file_stat.st_ino,
                file_stat.st_size,
                file_stat.st_mtime_ns,
            )
            if self.database.is_symlink() or identity != self.file_identity:
                raise SnapshotError("Snapshot file changed after initialization")
            with closing(_connection(self.database)) as connection:
                for table, columns in DATASETS.items():
                    by_industry = "industry_code" in KEYS[table]
                    params = [region, start, end]
                    where = "region_code = ? AND period >= ? AND period <= ?"
                    if by_industry:
                        where += " AND industry_code = ?"
                        params.append(industry)
                    fields = ", ".join(("period", *columns))
                    rows = connection.execute(
                        f"SELECT {fields} FROM {table} WHERE {where} ORDER BY period", params
                    )
                    result[table] = {}
                    for row in rows:
                        if any(
                            row[column] is not None
                            and (type(row[column]) is not int or row[column] < 0)
                            for column in columns
                        ):
                            raise SnapshotError("Invalid source metric value")
                        result[table][row["period"]] = dict(row)
        except (sqlite3.Error, OSError) as exc:
            raise SnapshotError("Could not query Seoul snapshot") from exc
        return result


def _index(period: str) -> int:
    return int(period[:4]) * 4 + int(period[-1]) - 1


def _period(index: int) -> str:
    return f"{index // 4:04d}Q{index % 4 + 1}"


def _metric(value: int | None, missing_row: bool) -> dict:
    reason = (
        "scope_not_in_snapshot"
        if missing_row
        else ("source_value_missing" if value is None else None)
    )
    messages = {
        "scope_not_in_snapshot": "해당 조건의 원본 관측이 없습니다.",
        "source_value_missing": "원본에서 해당 값이 제공되지 않았습니다.",
    }
    return {"value": value, "reason": reason, "reason_message": messages.get(reason)}


def _indicator(table: str, row: dict | None) -> dict:
    values = {
        field: _metric(None if row is None else row[field], row is None)
        for field in DATASETS[table]
    }
    if row is None:
        status, reason = "unavailable", "scope_not_in_snapshot"
    elif all(metric["value"] is None for metric in values.values()):
        status, reason = "unavailable", "source_value_missing"
    elif any(metric["value"] is None for metric in values.values()):
        status, reason = "partial", "source_value_missing"
    else:
        status, reason = "available", None
    return {"status": status, "reason": reason, "values": values}


def create_app(settings: SeoulStatsSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = settings or SeoulStatsSettings.from_env()
        if not config.api_key or not config.api_key.isascii():
            raise ValueError("PICKPLACE_SEOUL_STATS_API_KEY must be nonempty ASCII")
        application.state.settings = config
        application.state.snapshot = None
        try:
            application.state.snapshot = SeoulSnapshot(config.data_dir)
        except SnapshotError:
            logger.warning("Seoul research snapshot unavailable or invalid")
        yield
        application.state.snapshot = None

    application = FastAPI(title="PickPlace Seoul Descriptive Research API", lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
            for item in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    def authorize(
        request: Request, api_key: Annotated[str | None, Header(alias="X-API-Key")] = None
    ) -> None:
        if (
            api_key is None
            or not api_key.isascii()
            or not secrets.compare_digest(api_key, request.app.state.settings.api_key)
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    def snapshot(request: Request) -> SeoulSnapshot:
        result = request.app.state.snapshot
        if result is None:
            raise HTTPException(status_code=503, detail="seoul_snapshot_unavailable")
        return result

    @application.get("/health")
    def health(request: Request) -> dict:
        ready = request.app.state.snapshot is not None
        return {"status": "ok" if ready else "unavailable", "mode": MODE, "snapshot_ready": ready}

    @application.get("/research/seoul/metadata", dependencies=[Depends(authorize)])
    def metadata(request: Request) -> dict:
        return snapshot(request).manifest

    @application.get("/research/seoul/indicators", dependencies=[Depends(authorize)])
    def indicators(
        request: Request,
        region_code: Annotated[str, Query(pattern=r"^11[0-9]{6}$")],
        industry_code: Annotated[str, Query(pattern=r"^CS[0-9]{6}$")],
        start_period: Annotated[str, Query(pattern=PERIOD_PATTERN)],
        end_period: Annotated[str, Query(pattern=PERIOD_PATTERN)],
    ) -> dict:
        begin, end = _index(start_period), _index(end_period)
        if end < begin or end - begin >= 40:
            raise HTTPException(status_code=422, detail="invalid_or_excessive_period_range")
        source = snapshot(request)
        try:
            rows = source.query(region_code, industry_code, start_period, end_period)
        except SnapshotError:
            raise HTTPException(status_code=503, detail="seoul_snapshot_unavailable") from None
        return {
            "mode": MODE,
            "region_code_system": REGION_SYSTEM,
            "region_code": region_code,
            "industry_code_system": INDUSTRY_SYSTEM,
            "industry_code": industry_code,
            "period_granularity": "quarter",
            "source_provider": "서울특별시 / 서울 열린데이터광장",
            "source_urls": {
                "sales": "https://data.seoul.go.kr/dataList/OA-22175/S/1/datasetView.do",
                "stores": "https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do",
                "resident": "https://data.seoul.go.kr/dataList/OA-22183/S/1/datasetView.do",
                "floating": "https://data.seoul.go.kr/dataList/OA-22178/S/1/datasetView.do",
            },
            "dataset_scope": {
                "sales": "period_region_industry",
                "stores": "period_region_industry",
                "resident": "period_region",
                "floating": "period_region",
            },
            "snapshot_created_at_utc": source.manifest.get("created_at_utc"),
            "publication_date_verified": False,
            "source_units": {
                "sales.sales_amount": "KRW (원본 표기; 기간 집계 의미 미확인)",
                "stores.store_count": "개 (원본 정의; 당기 폐업 점포 포함 가능)",
                "stores.opening_count": "개",
                "stores.closing_count": "개",
                "resident.total": "명 (원본 표기)",
                "floating.total": "명 (원본 표기; 일평균 여부 미확인)",
            },
            "source_columns": {
                "sales.sales_amount": "당월_매출_금액",
                "stores.store_count": "유사_업종_점포_수 / 전체_점포_수",
                "stores.opening_count": "개업_점포_수",
                "stores.closing_count": "폐업_점포_수",
                "resident.total": "총_상주인구_수",
                "floating.total": "총_유동인구_수",
            },
            "observations": [
                {
                    "period": (period := _period(i)),
                    **{table: _indicator(table, rows[table].get(period)) for table in DATASETS},
                }
                for i in range(begin, end + 1)
            ],
            "unavailable_metrics": {
                "monthly_average_sales_per_store": {
                    "value": None,
                    "reason": "source_time_basis_and_denominator_unverified",
                    "reason_message": "매출의 기간 기준과 점포당 산출 분모를 확인하지 못했습니다.",
                },
                "daily_average_floating_population": {
                    "value": None,
                    "reason": "daily_average_definition_unverified",
                    "reason_message": "원본 유동인구가 일평균인지 확인하지 못했습니다.",
                },
                "monthly_series": {
                    "value": None,
                    "reason": "quarterly_source_not_monthly_observations",
                    "reason_message": "분기 자료로 월별 관측값을 만들 수 없습니다.",
                },
            },
            "warnings": [
                "latest_snapshot_not_proof_of_historical_publication",
                "sales_period_basis_unverified_values_preserved_as_published",
                "resident_population_annual_values_may_repeat",
                "floating_source_total_is_not_confirmed_daily_average",
                "store_source_total_may_include_closed_stores",
                "region_and_industry_code_crosswalk_unverified",
                "descriptive_statistics_not_prediction_or_score",
            ],
        }

    return application


app = create_app()
