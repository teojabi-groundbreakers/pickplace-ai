"""Load only trusted, locally configured artifacts with compatibility checks.

The checksum catches corruption; it does not make an untrusted pickle safe.
"""

import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pickplace_ai.features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION


def package_versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scikit-learn", "joblib"):
        result[package] = importlib.metadata.version(package)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactError(ValueError):
    """An artifact cannot be used without retraining or environment repair."""


@dataclass
class ModelBundle:
    pipeline: Any
    metadata: dict[str, Any]

    def predict(self, features: dict[str, float]) -> float:
        if set(features) != set(FEATURE_COLUMNS):
            raise ArtifactError("Inference feature names do not match the saved feature schema")
        try:
            frame = pd.DataFrame([features], columns=FEATURE_COLUMNS, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ArtifactError("Inference features must be numeric or missing") from exc
        if np.isinf(frame.to_numpy()).any():
            raise ArtifactError("Inference features cannot contain infinite values")
        values = np.asarray(self.pipeline.predict(frame), dtype=float).reshape(-1)
        if len(values) != 1 or not math.isfinite(float(values[0])):
            raise ArtifactError("Model did not return one finite growth prediction")
        # The training evaluation applies this same physical lower bound.
        return max(-1.0, float(values[0]))

    def explain(self, features: dict[str, float], max_items: int = 3) -> list[dict[str, Any]]:
        """Single-feature sensitivity to TRAIN medians, not SHAP or causal effects.

        Changing one derived feature can create an inconsistent combination with
        correlated features. Deltas do not add up to the prediction.
        """
        original = self.predict(features)
        references = self.metadata.get("reference_features", {})
        if not isinstance(references, dict) or max_items <= 0:
            return []
        results = []
        for name in FEATURE_COLUMNS:
            observed = features[name]
            reference = references.get(name)
            if observed is None or reference is None:
                continue
            if not math.isfinite(float(observed)) or not math.isfinite(float(reference)):
                continue
            if math.isclose(float(observed), float(reference), rel_tol=1e-12, abs_tol=1e-12):
                continue
            changed = {**features, name: float(reference)}
            difference = original - self.predict(changed)
            if abs(difference) <= 1e-12:
                continue
            results.append(
                {
                    "method": "single_feature_reference_sensitivity",
                    "feature": name,
                    "observed_value": float(observed),
                    "reference_value": float(reference),
                    "delta_growth": float(difference),
                    "direction": "positive" if difference > 0 else "negative",
                }
            )
        return sorted(results, key=lambda row: abs(row["delta_growth"]), reverse=True)[:max_items]


def load_bundle(path: Path | str) -> ModelBundle:
    """Read a bundle from an operator-controlled directory, never a request path."""
    directory = Path(path).resolve(strict=True)
    if not directory.is_dir():
        raise ArtifactError("Model path must be a directory")
    metadata_path = directory / "metadata.json"
    model_path = directory / "model.joblib"
    for artifact in (metadata_path, model_path):
        if artifact.resolve(strict=True).parent != directory:
            raise ArtifactError("Artifact files must remain inside the configured directory")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError("Cannot read model metadata") from exc
    if not isinstance(metadata, dict):
        raise ArtifactError("Model metadata must be an object")
    required = {
        "model_version",
        "source_kind",
        "release_status",
        "feature_schema_version",
        "feature_columns",
        "supported_scopes",
        "versions",
        "model_sha256",
        "selected_model",
    }
    if required - metadata.keys():
        raise ArtifactError("Required model metadata is missing")
    if metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ArtifactError("Model feature schema version is incompatible")
    if metadata["feature_columns"] != list(FEATURE_COLUMNS):
        raise ArtifactError("Model feature columns or order are incompatible")
    if metadata["source_kind"] not in ("official", "synthetic"):
        raise ArtifactError("Unknown artifact data provenance")
    if metadata["release_status"] not in ("experimental", "validated"):
        raise ArtifactError("Unknown model release status")
    if not isinstance(metadata["supported_scopes"], list) or not metadata["supported_scopes"]:
        raise ArtifactError("Model must declare its supported scopes")
    scope_keys = {"region_code", "region_code_system", "industry_code"}
    for scope in metadata["supported_scopes"]:
        if not isinstance(scope, dict) or set(scope) != scope_keys:
            raise ArtifactError("Invalid model scope metadata")
        if any(not isinstance(value, str) or not value for value in scope.values()):
            raise ArtifactError("Scope metadata values must be nonempty strings")
    saved_versions = metadata["versions"]
    if not isinstance(saved_versions, dict):
        raise ArtifactError("Model must declare package versions")
    for name, version in package_versions().items():
        if name == "python":
            if str(saved_versions.get(name, "")).split(".")[:2] != version.split(".")[:2]:
                raise ArtifactError("Python major/minor version differs from the training runtime")
        elif saved_versions.get(name) != version:
            raise ArtifactError(f"{name} version differs from the training runtime")
    if metadata["selected_model"] == "lightgbm":
        try:
            installed = importlib.metadata.version("lightgbm")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ArtifactError("LightGBM model requires the recorded LightGBM runtime") from exc
        if saved_versions.get("lightgbm") != installed:
            raise ArtifactError("LightGBM version differs from the training runtime")
    if metadata["model_sha256"] != file_sha256(model_path):
        raise ArtifactError("Model checksum mismatch")
    # Only operator-controlled, trusted files may reach this pickle-based loader.
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ArtifactError("Invalid model payload")
    if not callable(getattr(payload["pipeline"], "predict", None)):
        raise ArtifactError("Saved pipeline cannot predict")
    return ModelBundle(pipeline=payload["pipeline"], metadata=metadata)
