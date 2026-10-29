"""Explicit adapters for locally downloaded official Seoul CSV exports.

The adapter preserves source identifiers and published values. It does not infer
historical release dates, convert administrative codes, or reinterpret a field
named ``당월_매출_금액`` as a monthly total.
"""

import csv
import io
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from pickplace_ai.periods import quarter_index
from pickplace_ai.schemas import Observation

ALIASES = {
    "period": ("STDR_YYQU_CD", "기준_년분기_코드"),
    "region_code": ("ADSTRD_CD", "행정동_코드"),
    "industry_code": ("SVC_INDUTY_CD", "서비스_업종_코드"),
    "sales_amount": ("THSMON_SELNG_AMT", "당월_매출_금액"),
    "store_count": ("SIMILR_INDUTY_STOR_CO", "전체_점포_수"),
    "opening_count": ("OPBIZ_STOR_CO", "개업_점포_수"),
    "closing_count": ("CLSBIZ_STOR_CO", "폐업_점포_수"),
}

KEY_FIELDS = ("period", "region_code", "industry_code")
Key = tuple[str, str, str]


class ImportDataError(ValueError):
    """Source shape or data does not meet the explicit import contract."""


def _read_csv(path: Path) -> tuple[list[str], list[tuple[int, dict[str, str]]]]:
    content = path.read_bytes()
    decoded = None
    for encoding in ("utf-8-sig", "cp949"):
        try:
            decoded = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise ImportDataError(f"{path}: expected UTF-8 or CP949 encoded CSV")
    reader = csv.DictReader(io.StringIO(decoded))
    if reader.fieldnames is None:
        raise ImportDataError(f"{path}: missing CSV header")
    headers = [name.strip() for name in reader.fieldnames]
    if len(set(headers)) != len(headers):
        raise ImportDataError(f"{path}: duplicate CSV header")
    rows = []
    for line_number, raw in enumerate(reader, start=2):
        if None in raw or any(value is None for value in raw.values()):
            raise ImportDataError(f"{path}:{line_number}: inconsistent number of CSV fields")
        row = {name.strip(): value.strip() for name, value in raw.items()}
        if not any(row.values()):
            continue
        rows.append((line_number, row))
    return headers, rows


def _columns(headers: list[str], required: tuple[str, ...], path: Path) -> dict[str, str]:
    result = {}
    for field, alternatives in ALIASES.items():
        present = [name for name in alternatives if name in headers]
        if len(present) > 1:
            raise ImportDataError(f"{path}: ambiguous aliases for {field}: {present}")
        if present:
            result[field] = present[0]
        elif field in required:
            raise ImportDataError(f"{path}: missing required column {field}: {alternatives}")
    return result


def _period(value: str) -> str:
    if re.fullmatch(r"\d{4}[1-4]", value):
        value = f"{value[:4]}Q{value[4]}"
    quarter_index(value)
    return value


def _number(value: str, *, integer: bool = False) -> int | float | None:
    if value == "":
        return None
    # Comma-separated values must have proper thousands grouping, not decimal commas.
    if "," in value and not re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", value):
        raise ImportDataError(f"Malformed numeric grouping: {value!r}")
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation as exc:
        raise ImportDataError(f"Invalid number: {value!r}") from exc
    if not number.is_finite() or number < 0:
        raise ImportDataError(f"Expected finite non-negative number: {value!r}")
    if integer:
        if number != number.to_integral_value():
            raise ImportDataError(f"Expected integer count: {value!r}")
        return int(number)
    return float(number)


def _source_rows(
    paths: list[Path],
    kind: Literal["sales", "stores"],
    industry_code: str | None,
) -> dict[Key, dict]:
    required = (*KEY_FIELDS, "sales_amount" if kind == "sales" else "store_count")
    result: dict[Key, dict] = {}
    provenance: dict[Key, str] = {}
    fields = (
        ("sales_amount",) if kind == "sales" else ("store_count", "opening_count", "closing_count")
    )
    for path in paths:
        headers, rows = _read_csv(Path(path))
        columns = _columns(headers, required, Path(path))
        for line_number, row in rows:
            location = f"{path}:{line_number}"
            try:
                source_industry = row[columns["industry_code"]]
                if industry_code is not None and source_industry != industry_code:
                    continue
                period = _period(row[columns["period"]])
                region = row[columns["region_code"]]
                if not re.fullmatch(r"\d{6,12}", region):
                    raise ImportDataError(f"Expected an unmodified string region code: {region!r}")
                if not source_industry:
                    raise ImportDataError("Missing industry code")
                key = (period, region, source_industry)
                if key in result:
                    raise ImportDataError(
                        f"Duplicate {kind} key {key}; first seen at {provenance[key]}"
                    )
                result[key] = {
                    name: _number(row[columns[name]], integer=kind == "stores")
                    if name in columns
                    else None
                    for name in fields
                }
                provenance[key] = location
            except ValueError as exc:
                raise ImportDataError(f"{location}: {exc}") from exc
    return result


def import_seoul(
    sales_paths: list[Path],
    stores_paths: list[Path],
    available_at: date,
    industry_code: str | None = None,
    *,
    sales_basis: Literal["as_published"] = "as_published",
) -> list[Observation]:
    """Left-join exported sales to store data using exact quarter/region/industry.

    ``available_at`` is the caller-confirmed time BOTH sources were available,
    not an automatically inferred quarter publication date. ``as_published``
    preserves the source sales value without multiplying or dividing by three.
    Missing store rows/counts stay null; duplicate rows are rejected.
    """
    if sales_basis != "as_published":
        raise ImportDataError("Only sales_basis='as_published' is supported")
    if not sales_paths:
        raise ImportDataError("At least one sales CSV is required")
    sales = _source_rows(sales_paths, "sales", industry_code)
    stores = _source_rows(stores_paths, "stores", industry_code)
    if not sales:
        raise ImportDataError("No sales rows match the requested industry filter")
    observations = []
    for (period, region_code, source_industry), values in sorted(sales.items()):
        key = (period, region_code, source_industry)
        source_dataset = "seoul-sales-adstrd:as_published"
        if key in stores:
            source_dataset += "+seoul-stores-adstrd"
        try:
            observations.append(
                Observation(
                    region_code=region_code,
                    region_code_system="seoul-adstrd",
                    industry_code=source_industry,
                    period=period,
                    available_at=available_at,
                    source_kind="official",
                    source_dataset=source_dataset,
                    **values,
                    **stores.get(key, {}),
                )
            )
        except ValidationError as exc:
            raise ImportDataError(f"Invalid imported observation {key}: {exc}") from exc
    return observations
