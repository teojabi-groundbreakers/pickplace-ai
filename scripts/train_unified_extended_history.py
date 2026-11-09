"""Compare a longer Seoul history under the existing joint research contract.

This is an exploratory current-snapshot simulation. It never overwrites the
existing model and does not claim historical publication dates are known.
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
import pickplace_ai.unified_features as shared
from pickplace_ai.artifacts import package_versions
from scripts import train_gyeonggi_research as base
from scripts import train_unified_growth as unified

KIND = "seoul_extended_history_unified_research_v1"
BASE_TARGET = "2025Q2"
VALIDATION_TARGET = "2025Q4"
HISTORICAL_TARGETS = ("2024Q2", "2024Q4", "2025Q2")
FINAL_GYEONGGI_TARGETS = ("2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2")


def target_periods(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="Q")]


def before_embargo(periods: list[str], target: str) -> list[str]:
    """Labels stop at least one full target quarter before evaluation target."""
    origin = pd.Period(target, freq="Q")
    return [p for p in periods if pd.Period(p, freq="Q") <= origin - 2]


def fit_joint(seoul: pd.DataFrame, gyeonggi: pd.DataFrame):
    if seoul.empty or gyeonggi.empty:
        raise ValueError("Both domains require nonempty data")
    frames = pd.concat([seoul, gyeonggi], ignore_index=True)
    if set(frames.source) != {"seoul", "gyeonggi"}:
        raise ValueError("Unexpected sources")
    counts = frames.source.value_counts().to_dict()
    weights = np.array([1.0 / counts[s] for s in frames.source])
    weights *= len(weights) / weights.sum()
    model = unified.model().fit(
        frames[list(shared.UNIFIED_RESEARCH_FEATURES)], frames.target, sample_weight=weights
    )
    return model


def score(model, frame: pd.DataFrame) -> dict:
    prediction = unified.predict_unified_model(
        unified.bundle(model), frame[list(shared.UNIFIED_RESEARCH_FEATURES)]
    )
    return unified.metric(frame.target.to_numpy(), prediction)


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
    sm, gm = json.loads(seoul_manifest.read_text()), json.loads(gyeonggi_manifest.read_text())
    if sm.get("output_sha256") != hashes["seoul"]:
        raise ValueError("Seoul checksum mismatch")
    if (
        gm.get("candidate_sha256") != hashes["gyeonggi"]
        or gm.get("purpose") != "review_only_not_training_or_prediction"
    ):
        raise ValueError("Gyeonggi checksum mismatch")
    try:
        seoul_known = max(
            date.fromisoformat(sm["historical_available_at"]),
            *(unified.acquisition_day(x["acquired_at_utc"]) for x in sm["api_sources"]),
        )
        gyeonggi_known = unified.acquisition_day(gm["generated_at"])
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("Missing provenance") from exc
    if (
        snapshot < max(seoul_known, gyeonggi_known)
        or snapshot > datetime.now(ZoneInfo("Asia/Seoul")).date()
    ):
        raise ValueError("Invalid snapshot date")
    sq = unified.seoul_quarters(seoul)
    monthly = pd.read_csv(
        gyeonggi, dtype={"기준년월": str, "행정동코드": str, "중분류업종코드": str}
    )
    gq = base.make_quarters(monthly)
    seoul_periods = target_periods("2022Q2", "2026Q2")
    seoul_frames = {p: unified.make_frame(sq, p, "seoul") for p in seoul_periods}
    gyeonggi_frames = {p: unified.make_frame(gq, p, "gyeonggi") for p in FINAL_GYEONGGI_TARGETS}
    eligible = before_embargo(seoul_periods, VALIDATION_TARGET)
    if eligible[-1] != BASE_TARGET:
        raise ValueError("Unexpected validation cutoff")
    short = seoul_frames[BASE_TARGET]
    extended = pd.concat([seoul_frames[p] for p in eligible], ignore_index=True)
    gg_train = gyeonggi_frames[BASE_TARGET]
    sv, gv = seoul_frames[VALIDATION_TARGET], gyeonggi_frames[VALIDATION_TARGET]
    candidates = {
        "baseline_joint": fit_joint(short, gg_train),
        "extended_joint": fit_joint(extended, gg_train),
    }
    validation = {
        name: {"seoul": score(model, sv), "gyeonggi": score(model, gv)}
        for name, model in candidates.items()
    }
    rolling = []
    for target in HISTORICAL_TARGETS:
        prior = before_embargo(seoul_periods, target)
        if not prior:
            raise ValueError("No rolling history")
        old_frame = seoul_frames[prior[-1]]
        all_frames = pd.concat([seoul_frames[p] for p in prior], ignore_index=True)
        old_model = unified.model().fit(
            old_frame[list(shared.UNIFIED_RESEARCH_FEATURES)], old_frame.target
        )
        long_model = unified.model().fit(
            all_frames[list(shared.UNIFIED_RESEARCH_FEATURES)], all_frames.target
        )
        rolling.append(
            {
                "target": target,
                "embargo": str(pd.Period(target, freq="Q") - 1),
                "old_training_period": prior[-1],
                "extended_training_periods": prior,
                "old": score(old_model, seoul_frames[target]),
                "extended": score(long_model, seoul_frames[target]),
            }
        )
    # Train distinct final candidate, after all comparisons, on labels through Q2.
    final_s = pd.concat([seoul_frames[p] for p in seoul_periods], ignore_index=True)
    final_g = pd.concat([gyeonggi_frames[p] for p in FINAL_GYEONGGI_TARGETS], ignore_index=True)
    final_model = fit_joint(final_s, final_g)
    scopes = {
        "seoul": sorted(
            {tuple(map(str, row)) for row in final_s[base.SCOPE].itertuples(index=False, name=None)}
        ),
        "gyeonggi": sorted(
            {tuple(map(str, row)) for row in final_g[base.SCOPE].itertuples(index=False, name=None)}
        ),
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
        "source_hashes": hashes,
        "code_sha256": {
            "trainer": base.sha(Path(__file__)),
            "original_unified": base.sha(Path(unified.__file__)),
            "unified_features": base.sha(Path(shared.__file__)),
            "common_features": base.sha(Path(features.__file__)),
            "base_trainer": base.sha(Path(base.__file__)),
        },
        "features": list(shared.UNIFIED_RESEARCH_FEATURES),
        "target": "quarter_sales[t+1]/quarter_sales[t-3]-1",
        "validation_target": VALIDATION_TARGET,
        "validation_embargo_target": "2025Q3",
        "baseline_training_periods": {"seoul": [BASE_TARGET], "gyeonggi": [BASE_TARGET]},
        "extended_training_periods": {"seoul": eligible, "gyeonggi": [BASE_TARGET]},
        "validation_rows": {"seoul": len(sv), "gyeonggi": len(gv)},
        "validation_metrics": validation,
        "rolling_seoul_only": rolling,
        "final_training_periods": {
            "seoul": seoul_periods,
            "gyeonggi": list(FINAL_GYEONGGI_TARGETS),
        },
        "final_training_rows": {"seoul": len(final_s), "gyeonggi": len(final_g)},
        "source_code_systems": unified.SOURCE_CODE_SYSTEMS,
        "source_scopes": scopes,
        "versions": package_versions(),
        "limitations": [
            "Different outcomes retained with source indicator and balanced source weights.",
            "Previously examined 2025Q4 validation is exploratory, not an independent holdout.",
            "No 2026Q2 metrics are computed; final model is refit through 2026Q2.",
            "Publication and region/industry equivalence unverified; no code mapping inferred.",
            "Historical current-snapshot simulation is not a real-time backtest.",
            "Not production approved or automatically connected to existing API.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extended-history-", dir=output.parent) as tmp:
        dest = Path(tmp) / "output"
        dest.mkdir()
        for name, model in (
            ("evaluation-extended.joblib", candidates["extended_joint"]),
            ("final-model.joblib", final_model),
        ):
            joblib.dump(unified.bundle(model), dest / name)
        metadata["model_sha256"] = {p.name: base.sha(p) for p in dest.iterdir()}
        (dest / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if hashes != {name: base.sha(path) for name, path in inputs.items()}:
            raise ValueError("Inputs changed during training")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Output already exists")
        os.rename(dest, output)
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
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("validation_metrics", "rolling_seoul_only", "final_training_rows")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
