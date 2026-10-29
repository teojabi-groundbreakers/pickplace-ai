"""Read and validate the canonical dataset, preserving codes and provenance."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pickplace_ai.periods import quarter_index
from pickplace_ai.schemas import Observation

INTEGER_FIELDS = {"store_count", "opening_count", "closing_count"}
FLOAT_FIELDS = {
    "sales_amount",
    "resident_population",
    "working_population",
    "floating_population",
    "area_km2",
}


def validate_observations(rows: list[Observation]) -> list[Observation]:
    if not rows:
        raise ValueError("Dataset contains no observations")
    seen = set()
    kinds = set()
    for row in rows:
        key = (row.region_code_system, row.region_code, row.industry_code, row.period)
        if key in seen:
            raise ValueError(f"Duplicate scope/quarter: {key}")
        seen.add(key)
        kinds.add(row.source_kind)
    if len(kinds) != 1:
        raise ValueError("Cannot mix official and synthetic observations in one dataset")
    return rows


def read_observations(path: Path) -> list[Observation]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        raw_rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw_rows, list):
            raise ValueError("Canonical JSON must contain a list of observations")
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("Missing CSV header")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("Duplicate canonical CSV header")
            if any(not name for name in reader.fieldnames):
                raise ValueError("Empty canonical CSV header")
            raw_rows = []
            for raw in reader:
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError("Canonical CSV row has an inconsistent number of columns")
                row = {key: value for key, value in raw.items()}
                for key in INTEGER_FIELDS | FLOAT_FIELDS:
                    if key in row:
                        value = row[key]
                        row[key] = (
                            None
                            if value in (None, "")
                            else int(value)
                            if key in INTEGER_FIELDS
                            else float(value)
                        )
                raw_rows.append(row)
    return validate_observations([Observation.model_validate(row) for row in raw_rows])


def write_observations(rows: list[Observation], path: Path) -> None:
    validate_observations(rows)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            row.region_code_system,
            row.region_code,
            row.industry_code,
            quarter_index(row.period),
        ),
    )
    if path.suffix.lower() == ".json":
        path.write_text(
            json.dumps([r.model_dump(mode="json") for r in ordered], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(Observation.model_fields))
            writer.writeheader()
            writer.writerows(row.model_dump(mode="json") for row in ordered)


def quality_report(rows: list[Observation]) -> dict:
    validate_observations(rows)
    groups = defaultdict(list)
    missing = Counter()
    for row in rows:
        groups[(row.region_code_system, row.region_code, row.industry_code)].append(row)
        for field in INTEGER_FIELDS | FLOAT_FIELDS:
            if getattr(row, field) is None:
                missing[field] += 1
    histories = []
    for scope, records in sorted(groups.items()):
        indices = sorted(quarter_index(r.period) for r in records)
        histories.append(
            {
                "region_code_system": scope[0],
                "region_code": scope[1],
                "industry_code": scope[2],
                "rows": len(records),
                "missing_quarters": indices[-1] - indices[0] + 1 - len(indices),
            }
        )
    return {
        "rows": len(rows),
        "scopes": len(groups),
        "source_kind": rows[0].source_kind,
        "period_min": min(rows, key=lambda r: quarter_index(r.period)).period,
        "period_max": max(rows, key=lambda r: quarter_index(r.period)).period,
        "missing_counts": dict(sorted(missing.items())),
        "histories": histories,
        "limitations": [
            "available_at is supplied provenance, not independently verified publication evidence",
            "Data quality checks do not establish representativeness or forecast accuracy",
        ],
    }
