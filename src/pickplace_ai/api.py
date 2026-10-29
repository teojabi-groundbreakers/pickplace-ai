"""Internal AI API for Spring Boot; it never substitutes demo results for official data."""

from __future__ import annotations

import logging
import math
import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pickplace_ai import __version__
from pickplace_ai.artifacts import ModelBundle, load_bundle
from pickplace_ai.features import FeatureUnavailable, feature_record
from pickplace_ai.periods import quarter_end, shift_quarter
from pickplace_ai.schemas import PredictionRequest
from pickplace_ai.scoring import diagnose

logger = logging.getLogger(__name__)


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false")
    return value.lower() == "true"


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path("artifacts/current")
    api_key: str | None = None
    require_api_key: bool = False
    allow_synthetic: bool = False
    allow_experimental: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_dir=Path(os.environ.get("PICKPLACE_MODEL_DIR", "artifacts/current")),
            api_key=os.environ.get("PICKPLACE_API_KEY") or None,
            require_api_key=_environment_flag("PICKPLACE_REQUIRE_API_KEY"),
            allow_synthetic=_environment_flag("PICKPLACE_ALLOW_SYNTHETIC"),
            allow_experimental=_environment_flag("PICKPLACE_ALLOW_EXPERIMENTAL"),
        )


def _model_block_reason(metadata: dict[str, Any], settings: Settings) -> str | None:
    if metadata.get("source_kind") not in {"official", "synthetic"}:
        return "invalid_model_provenance"
    if metadata["source_kind"] == "synthetic" and not settings.allow_synthetic:
        return "synthetic_model_disabled"
    release_status = metadata.get("release_status")
    if release_status not in {"experimental", "validated"}:
        return "invalid_release_status"
    if release_status != "validated" and not settings.allow_experimental:
        return "experimental_model_disabled"
    if not isinstance(metadata.get("supported_scopes"), list):
        return "invalid_model_scope"
    return None


def _temporal_block_reason(metadata: dict[str, Any], as_of: date) -> str | None:
    """Model selection and reported evaluation must not know dates after the request."""
    try:
        for name in ("train", "validation", "test"):
            partition = metadata["split"][name]
            for field in ("latest_as_of", "latest_label_available_at"):
                if date.fromisoformat(partition[field]) > as_of:
                    return "model_information_after_as_of"
    except (KeyError, TypeError, ValueError):
        return "invalid_model_temporal_metadata"
    return None


def _growth_explanations(
    bundle: ModelBundle, features: dict[str, float]
) -> tuple[list[dict[str, Any]], list[str]]:
    explain = getattr(bundle, "explain", None)
    if not callable(explain):
        return [], ["growth_explanations_unavailable"]
    try:
        explanations = explain(features, max_items=3)
        if not isinstance(explanations, list):
            raise ValueError("Model explanation must be a list")
        for item in explanations:
            if (
                not isinstance(item, dict)
                or item.get("method") != "single_feature_reference_sensitivity"
                or not isinstance(item.get("feature"), str)
                or not item["feature"]
            ):
                raise ValueError("Unsupported model explanation format")
            for key in ("observed_value", "reference_value", "delta_growth"):
                value = item.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Explanation values must be finite numbers")
                if not math.isfinite(value):
                    raise ValueError("Explanation values must be finite numbers")
        warnings = (
            ["single_feature_reference_sensitivity_is_not_shap_causal_or_additive"]
            if explanations
            else []
        )
        return explanations[:3], warnings
    except Exception:
        # Explanation failure must not relabel a valid growth estimate as a failed model.
        logger.exception("Growth explanation could not be produced")
        return [], ["growth_explanations_unavailable"]


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings if settings is not None else Settings.from_env()
    if config.require_api_key and not config.api_key:
        raise ValueError("PICKPLACE_API_KEY is required when PICKPLACE_REQUIRE_API_KEY=true")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.bundle = None
        application.state.model_reason = "model_not_found"
        if config.model_dir.is_dir():
            try:
                bundle = load_bundle(config.model_dir)
                application.state.bundle = bundle
                application.state.model_reason = _model_block_reason(bundle.metadata, config)
            except Exception:
                # Malformed or incompatible artifacts must not disable basic diagnostics.
                logger.exception("Configured local model could not be loaded")
                application.state.model_reason = "model_load_failed"
        yield
        application.state.bundle = None

    application = FastAPI(
        title="PickPlace Internal AI API",
        version=__version__,
        description=(
            "Spring Boot에서 호출하는 내부 분석 API. 성장률 예측과 규칙 기반 진단지수를 "
            "구분하며, 실제 데이터와 합성 예시를 자동 전환하지 않습니다."
        ),
        lifespan=lifespan,
    )
    application.state.settings = config

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Do not echo raw observations. NaN/Infinity inputs also cannot be JSON-encoded safely.
        details = [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    def authorize(
        api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        if config.api_key and (
            api_key is None or not secrets.compare_digest(api_key, config.api_key)
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

    @application.get("/health")
    def health(request: Request) -> dict[str, Any]:
        reason = request.app.state.model_reason
        return {
            "status": "ok",
            "version": __version__,
            "basic_diagnosis_ready": True,
            "growth_model_ready": reason is None,
            "growth_model_reason": reason,
            "authentication_enabled": config.api_key is not None,
        }

    @application.get("/model-info", dependencies=[Depends(authorize)])
    def model_info(request: Request) -> dict[str, Any]:
        bundle: ModelBundle | None = request.app.state.bundle
        visible_keys = (
            "model_version",
            "source_kind",
            "release_status",
            "feature_schema_version",
            "feature_columns",
            "supported_scopes",
            "selected_model",
            "trained_at",
            "validation_metrics",
            "test_metrics",
            "limitations",
            "explanation_method",
            "explanation_limitations",
            "split",
        )
        metadata = {} if bundle is None else bundle.metadata
        return {
            "ready": request.app.state.model_reason is None,
            "reason": request.app.state.model_reason,
            "model": {key: metadata[key] for key in visible_keys if key in metadata},
        }

    @application.post("/predict", dependencies=[Depends(authorize)])
    def predict(payload: PredictionRequest, request: Request) -> dict[str, Any]:
        if payload.source_kind == "synthetic" and not config.allow_synthetic:
            raise HTTPException(status_code=403, detail="Synthetic requests are disabled")

        diagnosis = diagnose(payload.history, payload.base_period)
        target_period = shift_quarter(payload.base_period, 1)
        comparison_period = shift_quarter(payload.base_period, -3)
        bundle: ModelBundle | None = request.app.state.bundle
        model_version = None if bundle is None else bundle.metadata.get("model_version")
        growth: dict[str, Any] = {
            "status": "unavailable",
            "reason": request.app.state.model_reason,
            "estimated_yoy_growth": None,
            "target_period": target_period,
            "comparison_period": comparison_period,
            "model_version": model_version,
            "explanations": [],
        }
        warnings = list(diagnosis.get("warnings", []))
        if payload.source_kind == "synthetic":
            warnings.append("합성 예시 데이터의 결과이며 실제 지역 진단이나 성능이 아닙니다.")

        if quarter_end(target_period) <= payload.as_of:
            growth["reason"] = "target_not_future"
        elif request.app.state.model_reason is None and bundle is not None:
            scope = {
                "region_code": payload.region_code,
                "region_code_system": payload.region_code_system,
                "industry_code": payload.industry_code,
            }
            if bundle.metadata["source_kind"] != payload.source_kind:
                growth["reason"] = "source_kind_mismatch"
            elif scope not in bundle.metadata["supported_scopes"]:
                growth["reason"] = "unsupported_scope"
            elif temporal_reason := _temporal_block_reason(bundle.metadata, payload.as_of):
                growth["reason"] = temporal_reason
            else:
                try:
                    features = feature_record(payload.history, payload.base_period, payload.as_of)
                    estimate = float(bundle.predict(features))
                    if not math.isfinite(estimate) or estimate < -1:
                        growth["reason"] = "prediction_out_of_bounds"
                    else:
                        growth.update(
                            status="available", reason=None, estimated_yoy_growth=estimate
                        )
                        explanations, explanation_warnings = _growth_explanations(bundle, features)
                        growth["explanations"] = explanations
                        warnings.extend(explanation_warnings)
                except FeatureUnavailable as exc:
                    growth["reason"] = exc.reason
                except Exception:
                    logger.exception("Model inference failed")
                    growth["reason"] = "model_inference_failed"
        return {
            "schema_version": "1.0",
            "region_code": payload.region_code,
            "region_code_system": payload.region_code_system,
            "industry_code": payload.industry_code,
            "source_kind": payload.source_kind,
            "as_of": payload.as_of,
            "base_period": payload.base_period,
            "model_version": model_version,
            "growth": growth,
            "risk": diagnosis["risk"],
            "fit": diagnosis["fit"],
            "policy_version": diagnosis["policy_version"],
            "warnings": warnings,
        }

    return application


app = create_app()
