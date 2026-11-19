"""Internal Seoul/Gyeonggi pooled-model research API; caller inputs are unverified."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import secrets
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from pickplace_ai import features, periods, seoul_guard_policy, unified_features
from pickplace_ai.artifacts import file_sha256, package_versions
from scripts import train_gyeonggi_research, train_unified_growth

logger = logging.getLogger(__name__)
KOREA = ZoneInfo("Asia/Seoul")
HASH = re.compile(r"[0-9a-f]{64}\Z")
SYSTEMS = {"seoul": "seoul-adstrd", "gyeonggi": "gyeonggi-admdong-10-unverified"}
UNIT = {"seoul": "KRW", "gyeonggi": "source_unit_unverified"}
SOURCE_DEFINITIONS = {
    "seoul": "seoul_administrative_dong_estimated_sales",
    "gyeonggi": "gyeonggi_administrative_dong_card_sales",
}
SCORE_REASON = "score_not_provided_by_research_model"


class Quarter(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    period: str = Field(pattern=r"^\d{4}Q[1-4]$")
    sales_amount: float | None = Field(default=None, ge=0, strict=True)

    @field_validator("period")
    @classmethod
    def valid_period(cls, value: str) -> str:
        periods.quarter_index(value)
        return value


class UnifiedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    source: Literal["seoul", "gyeonggi"]
    region_code_system: str = Field(min_length=1, max_length=80)
    region_code: str = Field(min_length=1, max_length=32)
    industry_code: str = Field(min_length=1, max_length=32)
    sales_unit: Literal["KRW", "source_unit_unverified"]
    base_period: str = Field(pattern=r"^\d{4}Q[1-4]$")
    target_period: str = Field(pattern=r"^\d{4}Q[1-4]$")
    as_of: date
    data_available_by: date | None = None
    history: list[Quarter] = Field(default_factory=list, max_length=20)

    @field_validator("base_period", "target_period")
    @classmethod
    def valid_period(cls, value: str) -> str:
        periods.quarter_index(value)
        return value


def _file(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink():
        raise ValueError("Artifact symlink not allowed")
    resolved = path.resolve(strict=True)
    if resolved.parent != root or not resolved.is_file():
        raise ValueError("Invalid artifact path")
    return resolved


def _date(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError("Missing acquisition date")
    return date.fromisoformat(value)


class Registry:
    """A hash checks integrity, not safety: only load operator-trusted joblib."""

    def __init__(self, root: Path):
        root = root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Model root is not a directory")
        metadata = json.loads(_file(root, "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("Invalid metadata")
        if metadata.get("artifact_kind") != train_unified_growth.KIND:
            raise ValueError("Incorrect artifact type")
        if metadata.get("purpose") != "retrospective_research_only_not_service":
            raise ValueError("Incorrect artifact purpose")
        if metadata.get("source_code_systems") != SYSTEMS:
            raise ValueError("Incorrect source code systems")
        if metadata.get("target") != "quarter_sales[t+1]/quarter_sales[t-3]-1":
            raise ValueError("Incorrect target")
        final_periods = metadata.get("final_training_target_periods")
        if not isinstance(final_periods, list) or not final_periods:
            raise ValueError("Missing training periods")
        if max(final_periods, key=periods.quarter_index) != "2026Q2":
            raise ValueError("Unexpected training range")
        dates = metadata.get("source_known_by")
        if not isinstance(dates, dict) or set(dates) != set(SYSTEMS):
            raise ValueError("Missing source acquisition dates")
        self.source_known_by = {source: _date(value) for source, value in dates.items()}
        try:
            created = datetime.fromisoformat(metadata["created_at_utc"])
            if created.tzinfo is None:
                raise ValueError("Creation time missing timezone")
            self.created_at = created.astimezone(KOREA).date()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Missing creation time") from exc
        if self.created_at < max(self.source_known_by.values()):
            raise ValueError("Creation predates source acquisition")
        scopes = metadata.get("source_scopes")
        if not isinstance(scopes, dict) or set(scopes) != set(SYSTEMS):
            raise ValueError("Missing source scopes")
        self.scopes = {}
        for source, items in scopes.items():
            if not isinstance(items, list) or not items:
                raise ValueError("Missing region/industry scopes")
            parsed: set[tuple[str, str]] = set()
            for item in items:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError("Invalid scope")
                region, industry = item
                if not isinstance(region, str) or not isinstance(industry, str):
                    raise ValueError("Invalid scope code")
                if not _valid_codes(source, region, industry) or (region, industry) in parsed:
                    raise ValueError("Invalid or duplicate scope code")
                parsed.add((region, industry))
            self.scopes[source] = parsed
        code_hashes = metadata.get("code_sha256")
        if not isinstance(code_hashes, dict):
            raise ValueError("Missing code hashes")
        for name, filename in (
            ("trainer", train_unified_growth.__file__),
            ("feature_code", features.__file__),
            ("unified_feature_code", unified_features.__file__),
            ("base_trainer", train_gyeonggi_research.__file__),
        ):
            if code_hashes.get(name) != file_sha256(Path(filename)):
                raise ValueError("Model code does not match artifact")
        versions = metadata.get("versions")
        if not isinstance(versions, dict):
            raise ValueError("Missing runtime versions")
        for name, actual in package_versions().items():
            expected = versions.get(name)
            if name == "python":
                valid = (
                    isinstance(expected, str) and expected.split(".")[:2] == actual.split(".")[:2]
                )
            else:
                valid = expected == actual
            if not valid:
                raise ValueError("Runtime version mismatch")
        contents = _file(root, "final-model.joblib").read_bytes()
        hashes = metadata.get("model_sha256")
        digest = hashes.get("final-model.joblib") if isinstance(hashes, dict) else None
        if not isinstance(digest, str) or not HASH.fullmatch(digest):
            raise ValueError("Missing model checksum")
        if hashlib.sha256(contents).hexdigest() != digest:
            raise ValueError("Model checksum mismatch")
        # A checksum is integrity only, not protection against malicious pickle code.
        bundle = joblib.load(io.BytesIO(contents))
        if not isinstance(bundle, dict) or bundle.get("artifact_kind") != train_unified_growth.KIND:
            raise ValueError("Invalid research bundle")
        if bundle.get("feature_columns") != list(unified_features.UNIFIED_RESEARCH_FEATURES):
            raise ValueError("Feature schema mismatch")
        if not callable(getattr(bundle.get("model"), "predict", None)):
            raise ValueError("Invalid model")
        self.bundle = bundle
        self.metadata = metadata
        self.model_sha256 = digest
        self.seoul_policy_ready = seoul_guard_policy.policy_compatible(metadata, digest)

    def predict(self, request: UnifiedRequest) -> dict:
        today = datetime.now(KOREA).date()
        estimate_kind = (
            "completed_quarter_unobserved_estimate"
            if periods.quarter_end(request.target_period) <= min(today, request.as_of)
            else "in_progress_or_future_quarter_estimate"
        )
        result = {
            "mode": "experimental_research_only",
            "status": "unavailable",
            "reason": None,
            "reason_message": None,
            "risk_signals": [],
            "risk_signal_messages": [],
            "risk_details": [],
            "risk_policy_prospectively_validated": False if request.source == "seoul" else None,
            "risk_policy_version": seoul_guard_policy.POLICY_VERSION
            if request.source == "seoul"
            else None,
            "risk_policy_status": "not_evaluated"
            if request.source == "seoul"
            else "not_applicable",
            "risk_policy_reason": (
                None if request.source == "seoul" else "gyeonggi_risk_policy_unverified"
            ),
            "risk_policy_reason_message": (
                None
                if request.source == "seoul"
                else "경기도 매출 보류 기준은 아직 검증되지 않아 적용되지 않습니다."
            ),
            "estimated_yoy_growth": None,
            "source": request.source,
            "source_definition": SOURCE_DEFINITIONS[request.source],
            "region_code": request.region_code,
            "region_code_system": request.region_code_system,
            "industry_code": request.industry_code,
            "base_period": request.base_period,
            "target_period": request.target_period,
            "comparison_period": periods.shift_quarter(request.target_period, -4),
            "estimate_kind": estimate_kind,
            "as_of": request.as_of.isoformat(),
            "data_available_by": request.data_available_by.isoformat()
            if request.data_available_by
            else None,
            "model_created_at": self.metadata["created_at_utc"],
            "model_sha256": self.model_sha256,
            "source_date_basis": self.metadata.get("source_date_basis"),
            "input_values_verified": False,
            "region_scope_verified": False,
            "scope_in_model": None,
            "sales_unit_verified": False,
            "requested_sales_unit": request.sales_unit,
            "source_expected_sales_unit": UNIT[request.source],
            **_unprovided_fields(),
            "warnings": [
                "Research-only pooled model; historical publication and "
                "prospective performance unverified.",
                "This model and all results are not approved for production or business decisions.",
                "Caller-supplied history and code mappings are not independently verified.",
                "Seoul estimated sales and Gyeonggi card sales are different measures.",
                "Existing FE research forecast labels Gyeonggi results as Seoul; "
                "do not display as-is.",
            ],
        }

        def unavailable(reason: str) -> dict:
            result["reason"] = reason
            return result

        if request.region_code_system != SYSTEMS[request.source]:
            return unavailable("unsupported_source_or_region_code_system")
        if not _valid_codes(request.source, request.region_code, request.industry_code):
            return unavailable("unsupported_region_or_industry_code")
        if request.sales_unit != UNIT[request.source]:
            return unavailable("unsupported_or_unverified_sales_unit")
        if request.base_period != "2026Q2" or request.target_period != "2026Q3":
            return unavailable("unsupported_forecast_period")
        if request.as_of > today:
            return unavailable("as_of_in_future")
        if request.as_of < max(self.created_at, *self.source_known_by.values()):
            return unavailable("as_of_before_model_snapshot")
        if request.data_available_by is None:
            return unavailable("input_known_by_missing")
        if request.data_available_by > request.as_of:
            return unavailable("input_known_after_as_of")
        if request.data_available_by < periods.quarter_end(request.base_period):
            return unavailable("input_known_before_base_period_end")
        if (request.region_code, request.industry_code) not in self.scopes[request.source]:
            result["scope_in_model"] = False
            return unavailable("unsupported_or_insufficient_scope")
        result["scope_in_model"] = True
        by_period = {row.period: row for row in request.history}
        if len(by_period) != len(request.history):
            return unavailable("duplicate_history_period")
        if any(
            periods.quarter_index(p) > periods.quarter_index(request.base_period) for p in by_period
        ):
            return unavailable("history_after_base_period")
        needed = [periods.shift_quarter(request.base_period, offset) for offset in range(-3, 1)]
        if any(p not in by_period for p in needed):
            return unavailable("insufficient_history")
        values = [by_period[p].sales_amount for p in needed]
        if any(value is None for value in values):
            return unavailable("missing_sales")
        if any(value <= 0 for value in values[:3]):
            return unavailable("invalid_sales_denominator")
        if request.source == "seoul":
            if not self.seoul_policy_ready:
                result["risk_policy_status"] = "unavailable"
                result["risk_policy_reason"] = "seoul_guard_policy_mismatch"
                result["reason_message"] = (
                    "현재 모델에 적용할 서울 보류 기준이 확인되지 않았습니다."
                )
                result["risk_policy_reason_message"] = result["reason_message"]
                return unavailable("seoul_guard_policy_mismatch")
            try:
                details = seoul_guard_policy.describe_sales_history(values, needed)
            except (ValueError, FloatingPointError, OverflowError):
                return unavailable("invalid_or_nonfinite_features")
            signals = [detail["signal"] for detail in details]
            result["risk_policy_status"] = "evaluated"
            result["risk_signals"] = signals
            result["risk_details"] = details
            result["risk_signal_messages"] = [
                seoul_guard_policy.SIGNALS[signal] for signal in signals
            ]
            result["warnings"].append(
                "Seoul risk thresholds are exploratory and have not been prospectively validated."
            )
            if signals:
                result["reason_message"] = (
                    "서울 매출 이력이 연구용 보류 기준에 해당해 예측을 보류했습니다."
                )
                return unavailable("seoul_sales_history_unstable")
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                frame = unified_features.unified_research_features(
                    np.array([values], dtype=float), 2, request.source
                )
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                return unavailable("invalid_or_nonfinite_features")
        except (ValueError, FloatingPointError, OverflowError):
            return unavailable("invalid_or_nonfinite_features")
        prediction = train_unified_growth.predict_unified_model(self.bundle, frame)
        if len(prediction) != 1 or not math.isfinite(float(prediction[0])):
            return unavailable("model_inference_failed")
        result.update(status="available", reason=None, estimated_yoy_growth=float(prediction[0]))
        return result


def _valid_codes(source: str, region: str, industry: str) -> bool:
    if source == "seoul":
        return bool(re.fullmatch(r"11[0-9]{6}", region) and re.fullmatch(r"CS[0-9]{6}", industry))
    if source == "gyeonggi":
        return bool(
            re.fullmatch(r"41[0-9]{8}", region) and re.fullmatch(r"[A-Z][0-9]{2}", industry)
        )
    return False


def create_app(model_root: Path | None = None, api_key: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        key = api_key if api_key is not None else os.environ.get("PICKPLACE_UNIFIED_API_KEY", "")
        if not key or not key.isascii():
            raise ValueError("PICKPLACE_UNIFIED_API_KEY must be nonempty ASCII")
        root = model_root or Path(
            os.environ.get("PICKPLACE_UNIFIED_MODEL_ROOT", "artifacts/unified")
        )
        application.state.api_key = key
        application.state.registry = None
        try:
            application.state.registry = Registry(root)
        except Exception:
            logger.exception("Unified research model unavailable")
        yield
        application.state.registry = None

    app = FastAPI(title="PickPlace Seoul/Gyeonggi Research API", lifespan=lifespan)

    def authorize(request: Request, x_api_key: Annotated[str | None, Header()] = None) -> None:
        if (
            x_api_key is None
            or not x_api_key.isascii()
            or not secrets.compare_digest(x_api_key, request.app.state.api_key)
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    def get_registry(request: Request) -> Registry:
        registry = request.app.state.registry
        if registry is None:
            raise HTTPException(status_code=503, detail="unified_model_unavailable")
        return registry

    @app.get("/health")
    def health(request: Request) -> dict:
        ready = request.app.state.registry is not None
        return {
            "status": "ok" if ready else "unavailable",
            "model_ready": ready,
            "seoul_risk_policy_ready": (
                request.app.state.registry.seoul_policy_ready if ready else None
            ),
            "mode": "experimental_research_only",
        }

    @app.get("/research/unified/model-info", dependencies=[Depends(authorize)])
    def model_info(request: Request) -> dict:
        registry = get_registry(request)
        return {
            "mode": "experimental_research_only",
            "model_sha256": registry.model_sha256,
            "model_created_at": registry.metadata["created_at_utc"],
            "base_period": "2026Q2",
            "target_periods": ["2026Q3"],
            "source_known_by": {
                source: value.isoformat() for source, value in registry.source_known_by.items()
            },
            "source_date_basis": registry.metadata.get("source_date_basis"),
            "source_code_systems": SYSTEMS,
            "source_definitions": SOURCE_DEFINITIONS,
            "seoul_risk_policy": {
                "version": seoul_guard_policy.POLICY_VERSION,
                "ready": registry.seoul_policy_ready,
                "prospectively_validated": False,
                "calibration_training_target": "2025Q2",
                "thresholds_selected_after_prior_analysis": True,
                "reference_sales_below_krw": seoul_guard_policy.LOW_REFERENCE_SALES_KRW,
                "latest_sales_below_krw": seoul_guard_policy.LOW_LATEST_SALES_KRW,
                "qoq_increase_above": seoul_guard_policy.MAX_QOQ_INCREASE,
                "qoq_decrease_below": seoul_guard_policy.MIN_QOQ_DECREASE,
            },
            "limitations": registry.metadata.get("limitations", []),
        }

    @app.post("/research/unified/predict", dependencies=[Depends(authorize)])
    def predict(payload: UnifiedRequest, request: Request) -> dict:
        registry = request.app.state.registry
        if registry is None:
            return JSONResponse(
                status_code=503, content=_model_failure(payload, "unified_model_unavailable")
            )
        try:
            return registry.predict(payload)
        except Exception:
            logger.exception("Unified research prediction failed")
            return JSONResponse(
                status_code=503, content=_model_failure(payload, "model_inference_failed")
            )

    return app


def _model_failure(payload: UnifiedRequest, reason: str) -> dict:
    return {
        "mode": "experimental_research_only",
        "status": "unavailable",
        "reason": reason,
        "risk_details": [],
        "estimated_yoy_growth": None,
        "source": payload.source,
        "source_definition": SOURCE_DEFINITIONS[payload.source],
        "region_code": payload.region_code,
        "region_code_system": payload.region_code_system,
        "industry_code": payload.industry_code,
        "base_period": payload.base_period,
        "target_period": payload.target_period,
        **_unprovided_fields(),
        "warnings": ["Research-only pooled model is unavailable."],
    }


def _unprovided_fields() -> dict:
    result = {
        "scores": {name: None for name in ("overall", "growth", "risk", "fit")},
        "score_unavailable_reasons": {
            name: SCORE_REASON for name in ("overall", "growth", "risk", "fit")
        },
    }
    reasons = {
        "factors": "validated_explanations_not_available",
        "statistics": "statistics_not_provided_by_research_model",
        "floating_population": "floating_population_data_not_available",
        "competition": "competition_data_not_available",
        "map": "map_data_not_available",
        "summary": "validated_summary_not_available",
        "recommendations": "validated_recommendations_not_available",
        "prediction_interval": "prediction_interval_not_validated",
        "retrospective_test": "final_model_not_independently_tested",
    }
    result.update({name: None for name in reasons})
    result["unavailable_reasons"] = reasons
    return result


app = create_app()
