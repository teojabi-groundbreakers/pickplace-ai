"""Internal, explicitly research-only access to the trusted 2026Q2 model.

The caller supplies observations; their provenance cannot be independently
verified. A valid checksum does not make an untrusted joblib file safe.
"""

import csv
import hashlib
import io
import json
import logging
import math
import os
import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field

from pickplace_ai import data, features, forecast_2026, ingest, periods, training
from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.features import FeatureUnavailable, feature_record
from pickplace_ai.schemas import PredictionRequest
from scripts import near_term_forecast, retrospective_all_industries

logger = logging.getLogger(__name__)
_CODE = re.compile(r"CS[0-9]{6}\Z")
_SCOPE = re.compile(r"11[0-9]{6}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")


class NearTermModelError(ValueError):
    """The trusted artifact is unavailable, inconsistent, or incompatible."""


class NearTermRequest(PredictionRequest):
    target_period: str = Field(pattern=r"^\d{4}Q[1-4]$")


@dataclass(frozen=True)
class NearTermSettings:
    model_root: Path
    api_key: str

    @classmethod
    def from_env(cls) -> "NearTermSettings":
        return cls(
            model_root=Path(
                os.environ.get("PICKPLACE_NEAR_TERM_MODEL_ROOT", "artifacts/near-term")
            ),
            api_key=os.environ.get("PICKPLACE_NEAR_TERM_API_KEY", ""),
        )


def _artifact(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink():
        raise NearTermModelError("Artifact symlinks are not allowed")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NearTermModelError("Missing model artifact") from exc
    if resolved.parent != root or not resolved.is_file():
        raise NearTermModelError("Invalid model artifact path")
    return resolved


class NearTermRegistry:
    """Load trusted artifacts once, after compatibility and integrity checks."""

    def __init__(self, path: Path):
        root = Path(path).resolve(strict=True)
        if not root.is_dir():
            raise NearTermModelError("Model root is not a directory")
        summary_path = _artifact(root, "summary.json")
        model_path = _artifact(root, "model.joblib")
        forecast_path = _artifact(root, "forecast-2026-q3-q4.csv")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or (
            summary.get("artifact_kind") != near_term_forecast.KIND
            or summary.get("release_status") != "experimental_research_only"
            or summary.get("origin") != "2026Q2"
        ):
            raise NearTermModelError("Invalid research summary")
        try:
            self.snapshot_at = date.fromisoformat(summary["as_of"])
        except (KeyError, TypeError, ValueError) as exc:
            raise NearTermModelError("Invalid research snapshot date") from exc
        self.summary = summary

        model_bytes = model_path.read_bytes()
        forecast_bytes = forecast_path.read_bytes()
        for key, content in (("model_sha256", model_bytes), ("forecast_sha256", forecast_bytes)):
            digest = summary.get(key)
            if not isinstance(digest, str) or not _HASH.fullmatch(digest):
                raise NearTermModelError("Invalid artifact digest")
            if hashlib.sha256(content).hexdigest() != digest:
                raise NearTermModelError("Artifact digest mismatch")
        code_paths = {
            "forecast_2026.py": forecast_2026.__file__,
            "training.py": training.__file__,
            "ingest.py": ingest.__file__,
            "data.py": data.__file__,
            "periods.py": periods.__file__,
            "retrospective_all_industries.py": retrospective_all_industries.__file__,
        }
        dependencies = summary.get("dependency_code_sha256")
        if not isinstance(dependencies, dict):
            raise NearTermModelError("Missing dependency hashes")
        if summary.get("runner_sha256") != file_sha256(Path(near_term_forecast.__file__)):
            raise NearTermModelError("Prediction code does not match model")
        if summary.get("feature_code_sha256") != file_sha256(Path(features.__file__)):
            raise NearTermModelError("Feature code does not match model")
        for name, code_path in code_paths.items():
            if dependencies.get(name) != file_sha256(Path(code_path)):
                raise NearTermModelError(f"Dependency differs: {name}")
        versions = summary.get("versions")
        if not isinstance(versions, dict):
            raise NearTermModelError("Missing runtime versions")
        for name, current in package_versions().items():
            expected = versions.get(name)
            if name == "python":
                valid = (
                    isinstance(expected, str) and expected.split(".")[:2] == current.split(".")[:2]
                )
            else:
                valid = expected == current
            if not valid:
                raise NearTermModelError(f"Runtime version mismatch: {name}")

        # Scope membership is derived only from the generated forecast file,
        # not from caller-provided data. This is not provenance verification.
        self.scopes: dict[tuple[str, str, str], dict[str, Any]] = {}
        with io.StringIO(forecast_bytes.decode("utf-8-sig"), newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "region_code_system",
                "region_code",
                "industry_code",
                "base_period",
                "target_period",
                "scope_seen_in_training",
                "industry_test_rows",
                "warning",
            }
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise NearTermModelError("Invalid scope file columns")
            for row in reader:
                key = (row["region_code"], row["industry_code"], row["target_period"])
                if (
                    row["region_code_system"] != "seoul-adstrd"
                    or not _SCOPE.fullmatch(key[0])
                    or not _CODE.fullmatch(key[1])
                    or key[2] not in {"2026Q3", "2026Q4"}
                    or row["base_period"] != "2026Q2"
                    or row["scope_seen_in_training"] not in {"True", "False"}
                    or key in self.scopes
                ):
                    raise NearTermModelError("Invalid or duplicate scope")
                count = int(row["industry_test_rows"])
                if count < 0:
                    raise NearTermModelError("Invalid industry test count")
                self.scopes[key] = {
                    "scope_seen_in_training": row["scope_seen_in_training"] == "True",
                    "industry_test_rows": count,
                    "artifact_warnings": [w for w in row["warning"].split(";") if w],
                }
        if not self.scopes:
            raise NearTermModelError("No available forecast scope")

        # Only load files trusted by the operator. Hashes detect corruption,
        # they do not protect against malicious pickle execution.
        bundle = joblib.load(io.BytesIO(model_bytes))
        if not isinstance(bundle, dict) or bundle.get("artifact_kind") != near_term_forecast.KIND:
            raise NearTermModelError("Invalid model bundle")
        models = bundle.get("models")
        if not isinstance(models, dict) or set(models) != {"1", "2"}:
            raise NearTermModelError("Invalid horizon models")
        results = summary.get("results")
        if not isinstance(results, dict):
            raise NearTermModelError("Missing model selection metadata")
        for horizon, target in (("1", "2026Q3"), ("2", "2026Q4")):
            item, result = models[horizon], results.get(horizon)
            if (
                not isinstance(item, dict)
                or item.get("features") != near_term_forecast.FEATURES
                or not hasattr(item.get("model"), "predict")
                or not isinstance(result, dict)
                or item.get("model_name") != result.get("selection")
                or result.get("target_period") != target
            ):
                raise NearTermModelError("Incompatible horizon model")
            metrics = result.get("common_2025_test", {}).get("new", {})
            if (
                not isinstance(metrics, dict)
                or not isinstance(metrics.get("rows"), int)
                or isinstance(metrics.get("rows"), bool)
                or metrics["rows"] < 1
                or not all(
                    isinstance(metrics.get(key), (int, float))
                    and not isinstance(metrics[key], bool)
                    and math.isfinite(metrics[key])
                    and metrics[key] >= 0
                    for key in ("mae_pp", "median_absolute_error_pp")
                )
            ):
                raise NearTermModelError("Invalid retrospective metrics")
        self.models = models

    def predict(self, payload: NearTermRequest) -> dict[str, Any]:
        today = datetime.now(ZoneInfo("Asia/Seoul")).date()
        supported_target = payload.target_period in {"2026Q3", "2026Q4"}
        comparison_period = (
            periods.shift_quarter(payload.target_period, -4) if supported_target else None
        )
        estimate_kind = None
        if supported_target:
            estimate_kind = (
                "completed_quarter_unobserved_estimate"
                if periods.quarter_end(payload.target_period) <= min(payload.as_of, today)
                else "in_progress_or_future_quarter_estimate"
            )
        result = {
            "mode": "experimental_research_only",
            "status": "unavailable",
            "estimated_yoy_growth": None,
            "reason": None,
            "region_code": payload.region_code,
            "region_code_system": payload.region_code_system,
            "industry_code": payload.industry_code,
            "base_period": payload.base_period,
            "target_period": payload.target_period,
            "comparison_period": comparison_period,
            "estimate_kind": estimate_kind,
            "as_of": payload.as_of.isoformat(),
            "generated_at": datetime.now(UTC).isoformat(),
            "snapshot_available_at": self.snapshot_at.isoformat(),
            "model_sha256": self.summary["model_sha256"],
            "model_created_at": self.summary.get("created_at"),
            "scope_seen_in_training": None,
            "industry_test_rows": None,
            "model_choice": None,
            "retrospective_test": None,
            "region_scope_verified": False,
            "input_values_verified": False,
            "warnings": [
                "Research-only estimates from revised snapshots; historical publication "
                "versions and prospective performance are unverified.",
                "The caller supplies the history; input values and provenance are unverified.",
                "H10-to-Seoul code mapping and geographic applicability are not verified.",
            ],
        }

        def unavailable(reason: str) -> dict[str, Any]:
            result["reason"] = reason
            return result

        if payload.source_kind != "official" or payload.region_code_system != "seoul-adstrd":
            return unavailable("unsupported_source_or_region_code_system")
        if not _SCOPE.fullmatch(payload.region_code) or not _CODE.fullmatch(payload.industry_code):
            return unavailable("unsupported_region_or_industry_code")
        if payload.base_period != "2026Q2" or payload.target_period not in {"2026Q3", "2026Q4"}:
            return unavailable("unsupported_forecast_period")
        if payload.as_of < self.snapshot_at:
            return unavailable("as_of_before_model_snapshot")
        if payload.as_of > today:
            return unavailable("as_of_in_future")
        scope = self.scopes.get((payload.region_code, payload.industry_code, payload.target_period))
        if scope is None:
            return unavailable("unsupported_or_insufficient_scope")
        result["scope_seen_in_training"] = scope["scope_seen_in_training"]
        result["industry_test_rows"] = scope["industry_test_rows"]
        if not scope["scope_seen_in_training"]:
            result["warnings"].append("scope_absent_from_training")
        if scope["industry_test_rows"] < 30:
            result["warnings"].append("low_industry_test_sample")
        comparison = next((row for row in payload.history if row.period == comparison_period), None)
        if comparison is None or comparison.sales_amount is None or comparison.sales_amount <= 0:
            return unavailable("missing_or_nonpositive_comparison_sales")
        if comparison.sales_amount < 1_000_000:
            result["warnings"].append("comparison_sales_below_1m_krw")
        try:
            values = feature_record(payload.history, payload.base_period, payload.as_of)
        except FeatureUnavailable as exc:
            return unavailable(exc.reason)
        horizon = "1" if payload.target_period == "2026Q3" else "2"
        frame = pd.DataFrame(
            [
                {
                    **values,
                    "comparison_sales": comparison.sales_amount,
                    "target_quarter": int(payload.target_period[-1]),
                    "industry_code": payload.industry_code,
                }
            ]
        )
        prediction = near_term_forecast.predict(self.models[horizon]["model"], frame)
        result.update(
            status="available",
            estimated_yoy_growth=float(prediction[0]),
            model_choice=self.models[horizon]["model_name"],
            retrospective_test={
                "target_period": "2025Q3" if horizon == "1" else "2025Q4",
                **{
                    key: self.summary["results"][horizon]["common_2025_test"]["new"][key]
                    for key in ("rows", "mae_pp", "median_absolute_error_pp")
                },
            },
        )
        return result


def create_app(settings: NearTermSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        config = settings or NearTermSettings.from_env()
        if not config.api_key or not config.api_key.isascii():
            raise ValueError("PICKPLACE_NEAR_TERM_API_KEY must be nonempty ASCII")
        application.state.settings = config
        application.state.registry = None
        try:
            application.state.registry = NearTermRegistry(config.model_root)
        except Exception:
            logger.exception("Near-term research registry is unavailable")
        yield
        application.state.registry = None

    application = FastAPI(
        title="PickPlace Near-Term Research API",
        description="Experimental 2026 snapshot only. Not approved for live decisions.",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
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

    def registry(request: Request) -> NearTermRegistry:
        value = request.app.state.registry
        if value is None:
            raise HTTPException(status_code=503, detail="near_term_model_unavailable")
        return value

    @application.get("/health")
    def health(request: Request) -> dict:
        ready = request.app.state.registry is not None
        return {
            "status": "ok" if ready else "unavailable",
            "mode": "experimental_research_only",
            "model_ready": ready,
        }

    @application.get("/research/near-term/model-info", dependencies=[Depends(authorize)])
    def model_info(request: Request) -> dict:
        model = registry(request)
        return {
            "mode": "experimental_research_only",
            "base_period": "2026Q2",
            "target_periods": ["2026Q3", "2026Q4"],
            "snapshot_available_at": model.snapshot_at.isoformat(),
            "model_sha256": model.summary["model_sha256"],
            "model_created_at": model.summary.get("created_at"),
            "warnings": model.summary.get("limits", []),
        }

    @application.post("/research/near-term/predict", dependencies=[Depends(authorize)])
    def predict(payload: NearTermRequest, request: Request) -> dict:
        model = registry(request)
        try:
            return model.predict(payload)
        except Exception as exc:
            logger.exception("Near-term prediction failed")
            raise HTTPException(status_code=503, detail="near_term_prediction_unavailable") from exc

    return application


app = create_app()
