"""Read-only research API for separately aggregated nationwide snapshots.

The two sources have different geographic code systems and are never joined.
Checksums detect accidental changes; they do not authenticate the producer.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
MODE = "descriptive_research_only"
STORE_SYSTEM = "sbiz-admin-8"
INDUSTRY_SYSTEM = "sbiz-commercial-2026-09"
POPULATION_SYSTEM = "mois-resident-admin-10"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_NONNEGATIVE_INT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
STORE_COLUMNS = {
    "region_code_system",
    "region_code",
    "region_name",
    "sido_name",
    "sigungu_name",
    "industry_code_system",
    "industry_code",
    "industry_name",
    "store_count",
}
POPULATION_COLUMNS = {
    "region_code_system",
    "region_code",
    "region_name",
    "resident_population",
}


class NationwideDataError(ValueError):
    """Snapshot files cannot be trusted for this local research service."""


@dataclass(frozen=True)
class NationwideSettings:
    data_dir: Path
    api_key: str

    @classmethod
    def from_env(cls) -> NationwideSettings:
        return cls(
            data_dir=Path(
                os.environ.get("PICKPLACE_NATIONWIDE_DATA_DIR", "data/processed/nationwide/current")
            ),
            api_key=os.environ.get("PICKPLACE_NATIONWIDE_API_KEY", ""),
        )


def _read_file(root: Path, name: str) -> bytes:
    path = root / name
    if path.is_symlink():
        raise NationwideDataError("Artifact symlinks are not allowed")
    try:
        resolved = path.resolve(strict=True)
        if resolved.parent != root or not resolved.is_file():
            raise NationwideDataError("Invalid artifact path")
        return resolved.read_bytes()
    except OSError as exc:
        raise NationwideDataError("Cannot read artifact") from exc


def _rows(content: bytes, required: set[str]) -> csv.DictReader:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise NationwideDataError("Invalid artifact encoding") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if (
        not reader.fieldnames
        or set(reader.fieldnames) != required
        or len(reader.fieldnames) != len(required)
    ):
        raise NationwideDataError("Invalid artifact columns")
    return reader


def _int(value: Any, allow_empty: bool = False) -> int | None:
    if allow_empty and value == "":
        return None
    if not isinstance(value, str) or not _NONNEGATIVE_INT.fullmatch(value):
        raise NationwideDataError("Invalid nonnegative integer")
    return int(value)


class NationwideSnapshot:
    def __init__(self, data_dir: Path):
        try:
            root = data_dir.resolve(strict=True)
        except OSError as exc:
            raise NationwideDataError("Snapshot directory unavailable") from exc
        if not root.is_dir():
            raise NationwideDataError("Snapshot directory unavailable")
        try:
            manifest = json.loads(_read_file(root, "manifest.json"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise NationwideDataError("Invalid manifest") from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or manifest.get("crosswalk_verified") is not False
        ):
            raise NationwideDataError("Invalid snapshot metadata")
        store_meta, pop_meta = manifest.get("stores"), manifest.get("population")
        if not isinstance(store_meta, dict) or not isinstance(pop_meta, dict):
            raise NationwideDataError("Missing source metadata")
        store_bytes, pop_bytes = (
            _read_file(root, "stores.csv"),
            _read_file(root, "population.csv"),
        )
        for metadata, content in ((store_meta, store_bytes), (pop_meta, pop_bytes)):
            expected = metadata.get("output_sha256")
            if (
                not isinstance(expected, str)
                or not _SHA.fullmatch(expected)
                or hashlib.sha256(content).hexdigest() != expected
            ):
                raise NationwideDataError("Snapshot digest mismatch")
        self.stores: dict[tuple[str, str], dict[str, Any]] = {}
        self.population: dict[str, dict[str, Any]] = {}
        for row in _rows(store_bytes, STORE_COLUMNS):
            if None in row or any(value is None for value in row.values()):
                raise NationwideDataError("Invalid store row column count")
            code, industry = row["region_code"], row["industry_code"]
            if (
                row["region_code_system"] != STORE_SYSTEM
                or row["industry_code_system"] != INDUSTRY_SYSTEM
                or not re.fullmatch(r"[0-9]{8}", code)
                or not re.fullmatch(r"[A-Z0-9]{6}", industry)
                or not row["region_name"]
                or not row["industry_name"]
            ):
                raise NationwideDataError("Invalid store row")
            key = (code, industry)
            count = _int(row["store_count"])
            if key in self.stores or count is None or count == 0:
                raise NationwideDataError("Duplicate or invalid store group")
            self.stores[key] = {
                "region_name": row["region_name"],
                "industry_name": row["industry_name"],
                "store_count": count,
            }
        for row in _rows(pop_bytes, POPULATION_COLUMNS):
            if None in row or any(value is None for value in row.values()):
                raise NationwideDataError("Invalid population row column count")
            code = row["region_code"]
            if (
                row["region_code_system"] != POPULATION_SYSTEM
                or not re.fullmatch(r"[0-9]{10}", code)
                or not row["region_name"]
                or code in self.population
            ):
                raise NationwideDataError("Invalid or duplicate population row")
            self.population[code] = {
                "region_name": row["region_name"],
                "resident_population": _int(row["resident_population"], allow_empty=True),
            }
        groups, pop_rows = store_meta.get("group_count"), pop_meta.get("row_count")
        if (
            isinstance(groups, bool)
            or not isinstance(groups, int)
            or groups != len(self.stores)
            or groups < 1
        ):
            raise NationwideDataError("Store group count mismatch")
        if (
            isinstance(pop_rows, bool)
            or not isinstance(pop_rows, int)
            or pop_rows != len(self.population)
            or pop_rows < 1
        ):
            raise NationwideDataError("Population row count mismatch")
        self.manifest = manifest

    def metadata(self) -> dict[str, Any]:
        stores, population = self.manifest["stores"], self.manifest["population"]
        return {
            "mode": MODE,
            "crosswalk_verified": False,
            "stores": {
                "region_code_system": STORE_SYSTEM,
                "industry_code_system": INDUSTRY_SYSTEM,
                "reference_label": stores.get("reference_label"),
                "reference_label_verified": False,
                "national_coverage_verified": False,
                "observed_date": stores.get("observed_date"),
                "available_at": stores.get("available_at"),
                "group_count": stores["group_count"],
            },
            "population": {
                "region_code_system": POPULATION_SYSTEM,
                "reference_month": population.get("reference_month"),
                "observed_date": population.get("observed_date"),
                "available_at": population.get("available_at"),
                "row_count": population["row_count"],
            },
            "warnings": [
                "descriptive_snapshot_not_prediction",
                "source_code_crosswalk_not_verified",
                "source_publication_dates_not_verified",
                "source_coverage_and_reference_label_not_verified",
                "residents_are_not_floating_population",
            ],
        }


def create_app(settings: NationwideSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = settings or NationwideSettings.from_env()
        if not config.api_key or not config.api_key.isascii():
            raise ValueError("PICKPLACE_NATIONWIDE_API_KEY must be nonempty ASCII")
        application.state.settings = config
        application.state.snapshot = None
        try:
            application.state.snapshot = NationwideSnapshot(config.data_dir)
        except Exception:
            logger.exception("Nationwide research snapshot is unavailable")
        yield
        application.state.snapshot = None

    application = FastAPI(title="PickPlace Nationwide Snapshot Research API", lifespan=lifespan)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": list(item["loc"]), "msg": item["msg"], "type": item["type"]}
            for item in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    def authorize(
        request: Request,
        api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        if (
            api_key is None
            or not api_key.isascii()
            or not secrets.compare_digest(api_key, request.app.state.settings.api_key)
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    def snapshot(request: Request) -> NationwideSnapshot:
        result = request.app.state.snapshot
        if result is None:
            raise HTTPException(status_code=503, detail="nationwide_snapshot_unavailable")
        return result

    @application.get("/health")
    def health(request: Request) -> dict:
        ready = request.app.state.snapshot is not None
        return {
            "status": "ok" if ready else "unavailable",
            "mode": MODE,
            "snapshot_ready": ready,
        }

    @application.get("/research/nationwide/metadata", dependencies=[Depends(authorize)])
    def metadata(request: Request) -> dict:
        return snapshot(request).metadata()

    @application.get("/research/nationwide/stores", dependencies=[Depends(authorize)])
    def stores(
        request: Request,
        region_code: Annotated[str, Query(pattern=r"^[0-9]{8}$")],
        industry_code: Annotated[str, Query(pattern=r"^[A-Z0-9]{6}$")],
    ) -> dict:
        source = snapshot(request)
        item = source.stores.get((region_code, industry_code))
        meta = source.metadata()["stores"]
        return {
            "mode": MODE,
            "status": "available" if item is not None else "unavailable",
            "reason": None if item is not None else "scope_not_in_snapshot",
            "region_code": region_code,
            "industry_code": industry_code,
            "region_name": item["region_name"] if item else None,
            "industry_name": item["industry_name"] if item else None,
            "store_count": item["store_count"] if item else None,
            "source": meta,
        }

    @application.get("/research/nationwide/population", dependencies=[Depends(authorize)])
    def population(
        request: Request, region_code: Annotated[str, Query(pattern=r"^[0-9]{10}$")]
    ) -> dict:
        source = snapshot(request)
        item = source.population.get(region_code)
        value = item["resident_population"] if item else None
        reason = (
            "region_not_in_snapshot"
            if item is None
            else ("population_missing" if value is None else None)
        )
        return {
            "mode": MODE,
            "status": "available" if reason is None else "unavailable",
            "reason": reason,
            "region_code": region_code,
            "region_name": item["region_name"] if item else None,
            "resident_population": value,
            "source": source.metadata()["population"],
        }

    return application


app = create_app()
