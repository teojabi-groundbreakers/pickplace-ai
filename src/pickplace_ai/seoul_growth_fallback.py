"""서울 2026Q4 성장률을 위한 2026Q2 기반 연구 모델 연결."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from pickplace_ai.artifacts import file_sha256
from pickplace_ai.data import read_observations
from pickplace_ai.periods import shift_quarter
from pickplace_ai.schemas import Observation, PredictionRequest

REQUESTED_BASE = "2026Q3"
FALLBACK_BASE = shift_quarter(REQUESTED_BASE, -1)
TARGET = shift_quarter(REQUESTED_BASE, 1)
CODE_SYSTEM = "seoul-adstrd"
REQUIRED_PERIODS = frozenset(shift_quarter(FALLBACK_BASE, offset) for offset in range(-4, 1))
MISSING_INPUT_REASONS = frozenset({"missing_sales", "insufficient_history"})


def supports_request(payload: PredictionRequest) -> bool:
    return (
        payload.base_period == REQUESTED_BASE
        and payload.region_code_system == CODE_SYSTEM
        and payload.source_kind == "official"
    )


def _source_digests(rows):
    return sorted(
        (row["kind"], row["period"], row["sha256"], row["metadata_sha256"]) for row in rows
    )


class SeoulGrowthFallback:
    def __init__(self, model_dir: Path, snapshot: Path):
        from pickplace_ai.near_term_api import NearTermRegistry

        registry = NearTermRegistry(model_dir)
        manifest = json.loads(snapshot.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        digest = file_sha256(snapshot)
        if (
            manifest["output_sha256"] != digest
            or manifest["region_code_system"] != CODE_SYSTEM
            or manifest["sales_basis"] != "as_published"
            or manifest["historical_input_sha256"] != registry.summary["historical_input_sha256"]
            or _source_digests(manifest["api_sources"])
            != _source_digests(registry.summary["new_sources"])
        ):
            raise ValueError("Fallback snapshot does not match the near-term model sources")

        observations = read_observations(snapshot)
        history: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        if len(observations) != manifest["rows"] or any(
            row.source_kind != "official"
            or row.region_code_system != CODE_SYSTEM
            or row.period > FALLBACK_BASE
            or row.available_at > registry.snapshot_at
            for row in observations
        ):
            raise ValueError("Invalid observations in the fallback snapshot")
        for row in observations:
            # 원본 서울 코드와 업종이 정확히 같은 최근 5분기만 요청에 사용한다.
            if row.period in REQUIRED_PERIODS:
                history[row.region_code, row.industry_code].append(row)
        if file_sha256(snapshot) != digest:
            raise ValueError("Fallback snapshot changed during loading")
        self.registry = registry
        self.history = dict(history)

    def predict(self, payload: PredictionRequest) -> dict:
        from pickplace_ai.near_term_api import NearTermRequest

        observations = self.history.get((payload.region_code, payload.industry_code), ())
        history = [row for row in observations if row.available_at <= payload.as_of]
        if {row.period for row in history} != REQUIRED_PERIODS:
            return {"reason": "fallback_insufficient_history"}
        if any(row.sales_amount is None for row in history):
            return {"reason": "fallback_missing_sales"}
        request = NearTermRequest(
            region_code=payload.region_code,
            region_code_system=CODE_SYSTEM,
            industry_code=payload.industry_code,
            base_period=FALLBACK_BASE,
            target_period=TARGET,
            source_kind="official",
            as_of=payload.as_of,
            history=history,
        )
        result = self.registry.predict(request)
        if result["status"] != "available":
            return {"reason": f"fallback_{result['reason']}"}
        return {
            "status": "available",
            "reason": None,
            "estimated_yoy_growth": result["estimated_yoy_growth"],
            "base_period": FALLBACK_BASE,
            "horizon_quarters": 2,
            "model_version": result["model_sha256"],
            "release_status": result["mode"],
        }
