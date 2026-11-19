"""Explore training-only Seoul label exclusions without filtering evaluation rows.

Thresholds are fixed training-data percentiles, not proof that any observation is
erroneous. The study is retrospective; no existing model or API is replaced.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

import pickplace_ai.features as features
import pickplace_ai.unified_features as unified_features
from pickplace_ai.artifacts import package_versions
from scripts import train_gyeonggi_research as base
from scripts import train_unified_extended_history as history
from scripts import train_unified_growth as unified

KIND = "seoul_training_outlier_exclusion_study_v1"


def masks_for_training_target(values: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Predeclared one-percent cutoffs are fit solely on Seoul training labels."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all() or (arr < -1).any():
        raise ValueError("Expected finite, one-dimensional valid growth targets")
    low, high = np.quantile(arr, [0.01, 0.99])
    return {"p01": float(low), "p99": float(high)}, {
        "all": np.ones(len(arr), dtype=bool),
        "exclude_above_p99": arr <= high,
        "exclude_outside_p01_p99": (arr >= low) & (arr <= high),
    }


def train(
    seoul: Path,
    seoul_manifest: Path,
    gyeonggi: Path,
    gyeonggi_manifest: Path,
    output: Path,
    snapshot: date,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output already exists")
    inputs = {
        "seoul": seoul,
        "seoul_manifest": seoul_manifest,
        "gyeonggi": gyeonggi,
        "gyeonggi_manifest": gyeonggi_manifest,
    }
    hashes = {name: base.sha(path) for name, path in inputs.items()}
    sm = json.loads(seoul_manifest.read_text())
    gm = json.loads(gyeonggi_manifest.read_text())
    if sm.get("output_sha256") != hashes["seoul"]:
        raise ValueError("Seoul source checksum mismatch")
    if (
        gm.get("purpose") != "review_only_not_training_or_prediction"
        or gm.get("candidate_sha256") != hashes["gyeonggi"]
    ):
        raise ValueError("Gyeonggi source checksum mismatch")
    try:
        seoul_known = max(
            date.fromisoformat(sm["historical_available_at"]),
            *(unified.acquisition_day(item["acquired_at_utc"]) for item in sm["api_sources"]),
        )
        gyeonggi_known = unified.acquisition_day(gm["generated_at"])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("Missing acquisition provenance") from exc
    if snapshot < max(seoul_known, gyeonggi_known):
        raise ValueError("Snapshot precedes observed data")
    if snapshot > datetime.now(ZoneInfo("Asia/Seoul")).date():
        raise ValueError("Snapshot is in the future")
    sq = unified.seoul_quarters(seoul)
    monthly = pd.read_csv(
        gyeonggi, dtype={"기준년월": str, "행정동코드": str, "중분류업종코드": str}
    )
    gq = base.make_quarters(monthly)
    periods = history.target_periods("2022Q2", "2025Q2")
    if periods != history.before_embargo(periods, "2025Q4"):
        raise ValueError("Training target crosses embargo")
    st = pd.concat(
        [unified.make_frame(sq, period, "seoul") for period in periods], ignore_index=True
    )
    gt = unified.make_frame(gq, "2025Q2", "gyeonggi")
    # Validation frames are never passed to masks_for_training_target.
    val_s = unified.make_frame(sq, "2025Q4", "seoul")
    val_g = unified.make_frame(gq, "2025Q4", "gyeonggi")
    thresholds, masks = masks_for_training_target(st.target.to_numpy())
    models = {}
    comparisons = {}
    for name, mask in masks.items():
        selected = st.loc[mask].copy()
        models[name] = history.fit_joint(selected, gt)
        comparisons[name] = {
            "seoul_train_rows": len(selected),
            "seoul_excluded_rows": int((~mask).sum()),
            "gyeonggi_train_rows": len(gt),
            "gyeonggi_excluded_rows": 0,
            "seoul_validation": history.score(models[name], val_s),
            "gyeonggi_validation": history.score(models[name], val_g),
            "validation_rows_excluded": 0,
        }
    metadata = {
        "artifact_kind": KIND,
        "purpose": "retrospective_research_only_not_service",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_review_date": snapshot.isoformat(),
        "source_known_by": {
            "seoul": seoul_known.isoformat(),
            "gyeonggi": gyeonggi_known.isoformat(),
        },
        "source_date_basis": {
            "seoul": "latest historical or API acquisition date, Asia/Seoul",
            "gyeonggi": "candidate review generation date; prior acquisition time unknown",
        },
        "source_code_systems": unified.SOURCE_CODE_SYSTEMS,
        "source_hashes": hashes,
        "code_sha256": {
            "study": base.sha(Path(__file__)),
            "history": base.sha(Path(history.__file__)),
            "original_unified": base.sha(Path(unified.__file__)),
            "unified_features": base.sha(Path(unified_features.__file__)),
            "common_features": base.sha(Path(features.__file__)),
            "base_trainer": base.sha(Path(base.__file__)),
        },
        "seoul_train_target_periods": periods,
        "gyeonggi_train_target_periods": ["2025Q2"],
        "validation_target_period": "2025Q4",
        "embargo_target_period": "2025Q3",
        "threshold_quantiles": [0.01, 0.99],
        "thresholds": thresholds,
        "threshold_definition": "Seoul training targets only; exact cutoff values retained",
        "validation_filter": "none",
        "training_source_weights": "each source has 50 percent of total weight after exclusion",
        "comparisons": comparisons,
        "versions": package_versions(),
        "limitations": [
            "Unusual growth is not proof of erroneous observations or failed businesses.",
            "Evaluation labels and all validation rows remain intact; no test data is used.",
            "2025Q4 validation was previously inspected and is not an independent test.",
            "Source measurement definitions and historical publication dates are unverified.",
            "Models are research comparison artifacts, not deployed or production approved.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seoul-outlier-study-", dir=output.parent) as tmp:
        stage = Path(tmp) / "output"
        stage.mkdir()
        for name, model in models.items():
            joblib.dump(unified.bundle(model), stage / f"{name}.joblib")
        metadata["model_sha256"] = {p.name: base.sha(p) for p in stage.iterdir()}
        (stage / "summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if hashes != {name: base.sha(path) for name, path in inputs.items()}:
            raise ValueError("Inputs changed during study")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output already exists")
        os.rename(stage, output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seoul", "seoul-manifest", "gyeonggi", "gyeonggi-manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--snapshot", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    result = train(
        args.seoul,
        args.seoul_manifest,
        args.gyeonggi,
        args.gyeonggi_manifest,
        args.output,
        args.snapshot,
    )
    print(json.dumps({k: result[k] for k in ("thresholds", "comparisons")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
