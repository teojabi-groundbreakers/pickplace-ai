"""Research-only access to trusted, operator-supplied retrospective artifacts.

The checksum is an integrity check; it does not make untrusted pickle files safe.
Never use a request-provided path or serve these artifacts as live forecasts.
"""

import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from pickplace_ai import features
from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.features import FEATURE_COLUMNS

SUMMARY_KIND = "all_industries_retrospective_research_v1"
MODEL_KIND = "retrospective_snapshot_experiment_v1"
RESEARCH_STATUS = "retrospective_research_only"
CODE = re.compile(r"CS[0-9]{6}\Z")


class ResearchModelError(ValueError):
    """Artifact configuration, compatibility, or integrity failure."""


class UnsupportedIndustry(ResearchModelError):
    """The industry is not an explicitly completed research artifact."""


@dataclass(frozen=True)
class ResearchModel:
    estimator: Any
    metadata: dict[str, Any]
    entry: dict[str, Any]
    variant: str

    def predict(self, inputs: dict[str, float]) -> float:
        if set(inputs) != set(FEATURE_COLUMNS):
            raise ResearchModelError("Feature schema mismatch")
        matrix = pd.DataFrame([inputs], columns=FEATURE_COLUMNS, dtype=float)
        if np.isinf(matrix.to_numpy()).any():
            raise ResearchModelError("Infinite input features")
        prediction = np.asarray(self.estimator.predict(matrix), dtype=float)
        if prediction.shape != (1,) or not np.isfinite(prediction[0]):
            raise ResearchModelError("Model returned an invalid prediction")
        return max(-1.0, float(prediction[0]))


class ResearchRegistry:
    """Whitelisted local artifacts, loaded only from a trusted read-only mount."""

    def __init__(self, root: Path, max_cache: int = 4):
        if max_cache < 1:
            raise ValueError("max_cache must be positive")
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ResearchModelError("Model root is not a directory")
        summary_path = self._inside(self.root / "summary.json", self.root)
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ResearchModelError("Cannot read research summary") from exc
        if not isinstance(summary, dict):
            raise ResearchModelError("Research summary is not an object")
        if (
            summary.get("artifact_kind") != SUMMARY_KIND
            or summary.get("release_status") != RESEARCH_STATUS
        ):
            raise ResearchModelError("Research summary has the wrong artifact type or status")
        if summary.get("source_kind") != "official":
            raise ResearchModelError("Research summary is not official data")
        try:
            self.snapshot_at = date.fromisoformat(summary["snapshot_available_at"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ResearchModelError("Invalid snapshot date") from exc
        self.input_sha256 = summary.get("input_sha256")
        if not isinstance(self.input_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.input_sha256
        ):
            raise ResearchModelError("Invalid snapshot digest")
        self.entries: dict[str, dict[str, Any]] = {}
        industries = summary.get("industries")
        if not isinstance(industries, list):
            raise ResearchModelError("Research industries must be a list")
        seen: set[str] = set()
        for item in industries:
            if not isinstance(item, dict):
                raise ResearchModelError("Invalid industry summary")
            code = item.get("industry_code")
            if not isinstance(code, str) or CODE.fullmatch(code) is None or code in seen:
                raise ResearchModelError("Invalid or duplicate industry code")
            seen.add(code)
            if item.get("status") == "completed" and item.get("artifact_status") == "complete":
                split = item.get("quarter_splits")
                warnings = item.get("warnings", [])
                if (
                    not isinstance(split, dict)
                    or not isinstance(split.get("test"), list)
                    or not split["test"]
                    or not all(isinstance(period, str) for period in split["test"])
                    or not isinstance(warnings, list)
                    or not all(isinstance(warning, str) for warning in warnings)
                ):
                    raise ResearchModelError("Invalid industry test periods or warnings")
                self.entries[code] = item
        self.max_cache = max_cache
        self._cache: OrderedDict[tuple[str, str, str], Any] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _inside(path: Path, expected_parent: Path) -> Path:
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ResearchModelError("Research artifact does not exist") from exc
        if path.is_symlink() or resolved.parent != expected_parent:
            raise ResearchModelError("Research artifact path must stay in its configured directory")
        return resolved

    def list_industries(self) -> list[dict[str, Any]]:
        return [
            {
                "industry_code": code,
                "industry_name": self.entries[code].get("industry_name"),
                "warnings": self.entries[code].get("warnings", []),
                "test_periods": self.entries[code].get("quarter_splits", {}).get("test", []),
            }
            for code in sorted(self.entries)
        ]

    def load(self, code: str, variant: str) -> ResearchModel:
        if code not in self.entries:
            raise UnsupportedIndustry("Industry is not available")
        if variant not in {"selected", "learned"}:
            raise ResearchModelError("Invalid model variant")
        industries = self._inside(self.root / "industries", self.root)
        directory = self._inside(industries / code, industries)
        if not directory.is_dir():
            raise ResearchModelError("Industry model path is not a directory")
        metadata_path = self._inside(directory / "metadata.json", directory)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ResearchModelError("Cannot read research model metadata") from exc
        if not isinstance(metadata, dict):
            raise ResearchModelError("Model metadata must be an object")
        if (
            metadata.get("artifact_kind") != MODEL_KIND
            or metadata.get("release_status") != RESEARCH_STATUS
        ):
            raise ResearchModelError("Model artifact is not research-only")
        if metadata.get("industry_code") != code or metadata.get("source_kind") != "official":
            raise ResearchModelError("Model industry or source mismatch")
        if metadata.get("snapshot_available_at") != self.snapshot_at.isoformat():
            raise ResearchModelError("Model snapshot does not match summary")
        if metadata.get("all_partition_information_time") != self.snapshot_at.isoformat():
            raise ResearchModelError("Model information time does not match snapshot")
        if metadata.get("all_industries_input_sha256") != self.input_sha256:
            raise ResearchModelError("Model input digest does not match summary")
        if metadata.get("feature_columns") != FEATURE_COLUMNS:
            raise ResearchModelError("Feature schema mismatch")
        if metadata.get("feature_code_sha256") != file_sha256(Path(features.__file__)):
            raise ResearchModelError("Feature code differs from the training code")
        current = package_versions()
        saved = metadata.get("versions")
        if not isinstance(saved, dict):
            raise ResearchModelError("Model library versions are missing")
        for name, version in current.items():
            if name == "python":
                matches = str(saved.get(name, "")).split(".")[:2] == version.split(".")[:2]
            else:
                matches = saved.get(name) == version
            if not matches:
                raise ResearchModelError(f"Model runtime version mismatch: {name}")
        source_manifest = metadata.get("raw_source_manifest")
        if (
            not isinstance(source_manifest, dict)
            or source_manifest.get("region_code_system") != "seoul-adstrd"
            or source_manifest.get("industry_code") != code
        ):
            raise ResearchModelError("Model region code system is not supported")
        if (
            metadata.get("target") != "sales[t+1] / sales[t-3] - 1"
            or metadata.get("prediction_floor") != -1.0
        ):
            raise ResearchModelError("Model prediction target is incompatible")
        split = metadata.get("quarter_splits")
        if not isinstance(split, dict):
            raise ResearchModelError("Model test periods are missing")
        periods = split.get("test")
        entry = self.entries[code]
        entry_split = entry.get("quarter_splits")
        if (
            not isinstance(entry_split, dict)
            or not isinstance(periods, list)
            or not periods
            or periods != entry_split.get("test")
        ):
            raise ResearchModelError("Test periods do not match summary")
        filename = "model.joblib" if variant == "selected" else "learned-model.joblib"
        digest_key = "model_sha256" if variant == "selected" else "learned_model_sha256"
        selected_key = "selection" if variant == "selected" else "learned_model_selection"
        if metadata.get(selected_key) != entry.get(selected_key):
            raise ResearchModelError("Model selection does not match summary")
        path = self._inside(directory / filename, directory)
        expected_digest = metadata.get(digest_key)
        if not isinstance(expected_digest, str) or file_sha256(path) != expected_digest:
            raise ResearchModelError("Model checksum mismatch")
        key = (code, variant, expected_digest)
        with self._lock:
            bundle = self._cache.get(key)
            if bundle is None:
                try:
                    bundle = joblib.load(path)
                except Exception as exc:
                    raise ResearchModelError("Cannot load trusted research model") from exc
                if (
                    not isinstance(bundle, dict)
                    or bundle.get("artifact_kind") != MODEL_KIND
                    or bundle.get("selection") != metadata.get(selected_key)
                    or not callable(getattr(bundle.get("pipeline"), "predict", None))
                ):
                    raise ResearchModelError("Invalid research model payload")
                self._cache[key] = bundle
                while len(self._cache) > self.max_cache:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(key)
        return ResearchModel(bundle["pipeline"], metadata, entry, variant)
