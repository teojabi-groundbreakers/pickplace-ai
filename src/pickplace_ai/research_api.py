"""Authenticated historical-replay endpoint; never a real-time forecast API."""

import logging
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pickplace_ai.features import FEATURE_SCHEMA_VERSION, FeatureUnavailable, feature_record
from pickplace_ai.periods import quarter_end, shift_quarter
from pickplace_ai.research_models import ResearchModelError, ResearchRegistry, UnsupportedIndustry
from pickplace_ai.schemas import PredictionRequest

logger = logging.getLogger(__name__)


class ResearchRequest(PredictionRequest):
    model_variant: Literal["selected", "learned"] = "selected"


@dataclass(frozen=True)
class ResearchSettings:
    model_root: Path
    api_key: str
    cache_size: int = 4

    @classmethod
    def from_env(cls) -> "ResearchSettings":
        return cls(
            model_root=Path(os.environ.get("PICKPLACE_RESEARCH_MODEL_ROOT", "artifacts/research")),
            api_key=os.environ.get("PICKPLACE_RESEARCH_API_KEY", ""),
        )


def create_app(settings: ResearchSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = settings if settings is not None else ResearchSettings.from_env()
        if not config.api_key or not config.api_key.isascii():
            raise ValueError("PICKPLACE_RESEARCH_API_KEY must be nonempty ASCII")
        application.state.settings = config
        application.state.registry = None
        application.state.registry_error = None
        try:
            application.state.registry = ResearchRegistry(config.model_root, config.cache_size)
        except (OSError, ResearchModelError):
            logger.exception("Cannot load retrospective research registry")
            application.state.registry_error = "research_registry_unavailable"
        yield
        application.state.registry = None

    application = FastAPI(
        title="PickPlace Retrospective Research API",
        description="Test-only historical replay of revised data. Not a real-time forecast.",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
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

    def get_registry(request: Request) -> ResearchRegistry:
        registry = request.app.state.registry
        if registry is None:
            raise HTTPException(status_code=503, detail="research_registry_unavailable")
        return registry

    @application.get("/health")
    def health(request: Request) -> dict:
        registry = request.app.state.registry
        return {
            "status": "ok" if registry else "unavailable",
            "mode": "retrospective_research_only",
            "registry_ready": registry is not None,
            "industry_count": len(registry.entries) if registry else 0,
            "reason": request.app.state.registry_error,
            "model_artifacts_verified": False,
            "model_loading": "on_demand",
        }

    @application.get("/research/industries", dependencies=[Depends(authorize)])
    def industries(request: Request) -> dict:
        registry = get_registry(request)
        return {
            "mode": "retrospective_research_only",
            "snapshot_available_at": registry.snapshot_at.isoformat(),
            "industries": registry.list_industries(),
        }

    @application.post("/research/predict", dependencies=[Depends(authorize)])
    def predict(payload: ResearchRequest, request: Request) -> dict:
        registry = get_registry(request)
        if payload.source_kind != "official" or payload.region_code_system != "seoul-adstrd":
            raise HTTPException(
                status_code=422, detail="Only official seoul-adstrd data are supported"
            )
        if len(payload.region_code) != 8 or not payload.region_code.startswith("11"):
            raise HTTPException(
                status_code=422, detail="Expected an eight-digit Seoul administrative code"
            )
        if payload.as_of < registry.snapshot_at:
            raise HTTPException(status_code=422, detail="as_of precedes the research snapshot")
        if any(row.available_at < registry.snapshot_at for row in payload.history):
            raise HTTPException(
                status_code=422, detail="history availability precedes the research snapshot"
            )
        try:
            model = registry.load(payload.industry_code, payload.model_variant)
        except UnsupportedIndustry as exc:
            raise HTTPException(status_code=404, detail="unsupported_industry") from exc
        except (OSError, ResearchModelError) as exc:
            logger.warning("Research model unavailable for industry %s", payload.industry_code)
            raise HTTPException(status_code=503, detail="research_model_unavailable") from exc
        target_period = shift_quarter(payload.base_period, 1)
        if quarter_end(target_period) > payload.as_of:
            raise HTTPException(
                status_code=422, detail="Only completed historical target quarters may be replayed"
            )
        if target_period not in model.metadata["quarter_splits"]["test"]:
            raise HTTPException(
                status_code=422, detail="Target quarter is outside this industry's test period"
            )
        try:
            values = feature_record(payload.history, payload.base_period, payload.as_of)
            estimate = model.predict(values)
        except FeatureUnavailable as exc:
            raise HTTPException(status_code=422, detail=exc.reason) from exc
        except Exception as exc:
            logger.exception("Retrospective research prediction failed")
            raise HTTPException(status_code=503, detail="research_prediction_unavailable") from exc
        key = "selection" if payload.model_variant == "selected" else "learned_model_selection"
        return {
            "mode": "retrospective_research_only",
            "status": "available",
            "region_code": payload.region_code,
            "region_code_system": payload.region_code_system,
            "industry_code": payload.industry_code,
            "base_period": payload.base_period,
            "target_period": target_period,
            "comparison_period": shift_quarter(payload.base_period, -3),
            "as_of": payload.as_of.isoformat(),
            "snapshot_available_at": registry.snapshot_at.isoformat(),
            "model_variant": payload.model_variant,
            "model_choice": model.metadata[key],
            "model_sha256": model.metadata[
                "model_sha256" if payload.model_variant == "selected" else "learned_model_sha256"
            ],
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "estimated_yoy_growth": estimate,
            "region_scope_verified": False,
            "input_values_verified": False,
            "warnings": [
                "Retrospective replay using revised data, not a live forecast "
                "or historical point-in-time validation.",
                "Region coverage and request data provenance were not verified against "
                "original observations; the caller supplies the history.",
                *model.entry.get("warnings", []),
            ],
        }

    return application


app = create_app()
