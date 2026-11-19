"""기준일이 맞는 서울 자료에서 업종 내 상대 위치를 계산한다."""

import csv
import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from datetime import date
from io import StringIO
from pathlib import Path

from pickplace_ai.periods import quarter_end
from pickplace_ai.schemas import PredictionRequest

KIND = "seoul_date_aligned_fit_reference_v1"
CODE_SYSTEM = "seoul-adstrd"
ROW_FIELDS = ("region_code", "industry_code", "store_count", "resident_population", "area_m2")
UNITS = {"residents_per_store": "persons_per_store", "stores_per_km2": "stores_per_km2"}


def validate_metadata(manifest: dict) -> None:
    if (
        not isinstance(manifest, dict)
        or manifest.get("kind") != KIND
        or manifest.get("region_code_system") != CODE_SYSTEM
    ):
        raise ValueError("Unexpected fit reference kind or region code system")
    if manifest.get("source_kind") != "official":
        raise ValueError("Fit reference requires official inputs")
    effective = date.fromisoformat(manifest["reference_date"])
    acquired = date.fromisoformat(manifest["available_at"])
    if quarter_end(manifest["base_period"]) != effective or acquired < effective:
        raise ValueError("Fit reference dates do not align")
    for field in ("area_reference_date", "population_reference_date", "mapping_reference_date"):
        if date.fromisoformat(manifest[field]) != effective:
            raise ValueError("Area, population and region-code mapping must use the same date")
    if manifest["stores_period"] != manifest["base_period"]:
        raise ValueError("Stores quarter must match the reference quarter")
    if type(manifest["row_count"]) is not int or manifest["row_count"] < 1:
        raise ValueError("Expected a positive reference row count")
    for key in ("area", "population", "mapping", "stores"):
        source = manifest["sources"][key]
        if not source["url"].startswith("https://") or not re.fullmatch(
            r"[0-9a-f]{64}", source["sha256"]
        ):
            raise ValueError("Missing official reference source or SHA-256")


def rank_percentile(value: float, sorted_values: list[float]) -> float | None:
    """동률은 평균 순위로 처리하고 최소·최대 순위를 0~100에 놓는다."""
    if len(sorted_values) < 2:
        return None
    low, high = bisect_left(sorted_values, value), bisect_right(sorted_values, value)
    return round((low + high - 1) * 50 / (len(sorted_values) - 1), 2)


class FitReference:
    """지문이 맞는 참조표를 시작 시 한 번 읽고 요청마다 같은 업종의 관측값을 찾는다."""

    def __init__(self, root: Path):
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        validate_metadata(manifest)
        self.manifest = manifest
        self.available_at = date.fromisoformat(manifest["available_at"])
        content = (root / "reference.csv").read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["reference_sha256"]:
            raise ValueError("Fit reference digest mismatch")
        reader = csv.DictReader(StringIO(content.decode("utf-8")))
        if tuple(reader.fieldnames or ()) != ROW_FIELDS:
            raise ValueError("Invalid fit reference columns")
        self.scopes = {}
        self.distributions = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Invalid fit reference row")
            region, industry = row["region_code"], row["industry_code"]
            stores, population = int(row["store_count"]), int(row["resident_population"])
            area = float(row["area_m2"])
            if (
                not re.fullmatch(r"11[0-9]{6}", region)
                or not re.fullmatch(r"CS[0-9]{6}", industry)
                or stores <= 0
                or population < 0
                or not math.isfinite(area)
                or area <= 0
                or (region, industry) in self.scopes
            ):
                raise ValueError("Invalid or duplicate fit reference scope")
            values = (population / stores, stores / (area / 1_000_000))
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Invalid fit reference ratios")
            self.scopes[region, industry] = (stores, population, area / 1_000_000, values)
            demand, density = self.distributions.setdefault(industry, ([], []))
            demand.append(values[0])
            density.append(values[1])
        if not self.scopes or len(self.scopes) != manifest["row_count"]:
            raise ValueError("Fit reference row count mismatch")
        for demand, density in self.distributions.values():
            demand.sort()
            density.sort()

    def indicators(self, request: PredictionRequest) -> dict:
        if request.source_kind != "official" or request.region_code_system != CODE_SYSTEM:
            return unavailable_fit(request.base_period, "unsupported_fit_reference_scope")
        if request.base_period != self.manifest["base_period"]:
            return unavailable_fit(request.base_period, "fit_reference_period_mismatch")
        if request.as_of < self.available_at:
            return unavailable_fit(request.base_period, "fit_reference_after_as_of")
        scope = self.scopes.get((request.region_code, request.industry_code))
        if scope is None:
            return unavailable_fit(request.base_period, "fit_scope_not_in_reference")
        stores, population, area, values = scope
        populations = self.distributions[request.industry_code]
        result = {
            "base_period": request.base_period,
            "reference_date": self.manifest["reference_date"],
            "reference_available_at": self.manifest["available_at"],
            "reference_kind": KIND,
            "cohort_size": len(populations[0]),
            "store_count": stores,
            "resident_population": population,
            "area_km2": area,
        }
        for (name, unit), value, distribution in zip(
            UNITS.items(), values, populations, strict=True
        ):
            percentile = rank_percentile(value, distribution)
            result[name] = {
                "status": "available",
                "value": value,
                "unit": unit,
                "percentile": percentile,
                "reason": None,
                "percentile_reason": "insufficient_reference_areas" if percentile is None else None,
            }
        return result


def unavailable_fit(base_period: str, reason: str) -> dict:
    return {
        "base_period": base_period,
        "reference_date": None,
        "reference_available_at": None,
        "reference_kind": None,
        "cohort_size": None,
        "store_count": None,
        "resident_population": None,
        "area_km2": None,
        **{
            name: {
                "status": "unavailable",
                "value": None,
                "unit": unit,
                "percentile": None,
                "reason": reason,
                "percentile_reason": reason,
            }
            for name, unit in UNITS.items()
        },
    }
