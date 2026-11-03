"""Exploratory snapshot experiment; never a point-in-time or production model.

The observation available_at dates are not changed.  Historical features and
labels are constructed from the currently available, possibly revised snapshot.
Quarter-order holdouts therefore measure retrospective fit, not historical
live forecasts. This artifact deliberately does not satisfy the API contract.
"""

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import pickplace_ai
from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.data import read_observations, validate_observations
from pickplace_ai.features import FEATURE_COLUMNS, FeatureUnavailable, feature_record
from pickplace_ai.periods import quarter_end, quarter_index, shift_quarter
from pickplace_ai.training import RecentGrowthRegressor

ARTIFACT_KIND = "retrospective_snapshot_experiment_v1"


def make_frame(observations, snapshot_at):
    """Construct labels from snapshot; never pretend it was published earlier."""
    validate_observations(observations)
    if any(row.available_at != snapshot_at for row in observations):
        raise ValueError("Every observation must preserve the specified snapshot availability date")
    if any(quarter_end(row.period) > snapshot_at for row in observations):
        raise ValueError("Snapshot cannot contain an unfinished quarter")
    grouped = defaultdict(list)
    for row in observations:
        grouped[(row.region_code_system, row.region_code, row.industry_code)].append(row)
    result = []
    for scope, history in grouped.items():
        by_period = {row.period: row for row in history}
        for current in history:
            target_period = shift_quarter(current.period, 1)
            target = by_period.get(target_period)
            comparison = by_period.get(shift_quarter(current.period, -3))
            if target is None or comparison is None or target.sales_amount is None:
                continue
            if comparison.sales_amount is None or comparison.sales_amount <= 0:
                continue
            try:
                # feature_record excludes later periods and never uses target rows.
                features = feature_record(history, current.period, snapshot_at)
            except FeatureUnavailable:
                continue
            label = target.sales_amount / comparison.sales_amount - 1
            if not np.isfinite(label):
                raise ValueError("Non-finite target")
            result.append(
                {
                    **features,
                    "target": label,
                    "region_code_system": scope[0],
                    "region_code": scope[1],
                    "industry_code": scope[2],
                    "base_period": current.period,
                    "target_period": target_period,
                    "snapshot_available_at": snapshot_at.isoformat(),
                }
            )
    frame = pd.DataFrame(result)
    if frame.empty:
        raise ValueError("No retrospective samples; check history and snapshot")
    return frame.sort_values(
        ["target_period", "region_code_system", "region_code", "industry_code"]
    ).reset_index(drop=True)


def split_frame(frame):
    """Split target quarters with one full embargo quarter at each boundary."""
    periods = sorted(frame.target_period.unique(), key=quarter_index)
    if len(periods) < 8:
        raise ValueError("Need at least 8 eligible target quarters for splits and embargoes")
    if any(quarter_index(b) - quarter_index(a) != 1 for a, b in pairwise(periods)):
        raise ValueError("Target quarters must be contiguous before splitting")
    period_sets = {
        "train": periods[:-6],
        "validation": periods[-5:-3],
        "test": periods[-2:],
    }
    embargoes = [periods[-6], periods[-3]]
    partitions = {
        name: frame.loc[frame.target_period.isin(value)].copy()
        for name, value in period_sets.items()
    }
    if any(part.empty for part in partitions.values()):
        raise ValueError("Empty partition")
    return partitions, period_sets, embargoes


def predict_bundle(bundle, X):
    """Apply the exact research-evaluation prediction floor to a trusted bundle."""
    if bundle.get("artifact_kind") != ARTIFACT_KIND:
        raise ValueError("Not a retrospective experiment bundle")
    if list(X.columns) != FEATURE_COLUMNS:
        raise ValueError("Feature columns and order must match the model schema")
    pred = np.asarray(bundle["pipeline"].predict(X), dtype=float)
    if pred.ndim != 1 or len(pred) != len(X) or not np.isfinite(pred).all():
        raise ValueError("Invalid predictions")
    return np.maximum(pred, -1.0)


def _metrics(y, pred):
    pred = np.maximum(np.asarray(pred, dtype=float), -1.0)
    if len(y) != len(pred) or not np.isfinite(pred).all():
        raise ValueError("Invalid predictions")
    mae = float(mean_absolute_error(y, pred))
    return {
        "mae": mae,
        "mae_pp": 100 * mae,
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }


def run(
    input_path,
    output_path,
    snapshot_at,
    source_kind="official",
    manifest_path=None,
    raw_root=None,
):
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    observations = read_observations(input_path)
    if {row.source_kind for row in observations} != {source_kind}:
        raise ValueError("Dataset source kind does not match the explicit experiment source")
    source_manifest = None
    if source_kind == "official":
        if manifest_path is None or raw_root is None:
            raise ValueError("Official runs require a source manifest and raw source root")
        source_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if source_manifest.get("snapshot_available_at") != snapshot_at.isoformat():
            raise ValueError("Source manifest snapshot date does not match")
        source_files = source_manifest.get("source_files", [])
        if not source_files:
            raise ValueError("Source manifest has no files")
        for item in source_files:
            name = item["name"]
            category = item["category"]
            if Path(name).name != name or category not in {"sales", "stores"}:
                raise ValueError("Invalid source manifest path")
            path = Path(raw_root) / category / name
            if file_sha256(path) != item["sha256"]:
                raise ValueError(f"Source hash mismatch: {name}")
    frame = make_frame(observations, snapshot_at)
    split, quarter_sets, embargoes = split_frame(frame)
    train, validation, test = (split[name] for name in ("train", "validation", "test"))
    X_train, y_train = train[FEATURE_COLUMNS], train.target
    X_validation, y_validation = validation[FEATURE_COLUMNS], validation.target
    X_test, y_test = test[FEATURE_COLUMNS], test.target
    if np.isinf(frame[FEATURE_COLUMNS].to_numpy(float)).any():
        raise ValueError("Infinite feature values are not supported")
    candidates = {
        "zero_growth": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("model", DummyRegressor(strategy="constant", constant=0)),
            ]
        ),
        "recent_growth": RecentGrowthRegressor(),
        "ridge": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10)),
            ]
        ),
        "hist": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=100,
                        max_leaf_nodes=15,
                        learning_rate=0.05,
                        l2_regularization=1,
                        early_stopping=False,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    validation_results = {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        candidate_bundle = {"artifact_kind": ARTIFACT_KIND, "pipeline": estimator}
        validation_results[name] = _metrics(
            y_validation, predict_bundle(candidate_bundle, X_validation)
        )
    selected = min(validation_results, key=lambda name: validation_results[name]["mae"])
    learned_selected = min(("ridge", "hist"), key=lambda name: validation_results[name]["mae"])
    test_results = {}
    for name, estimator in candidates.items():
        candidate_bundle = {"artifact_kind": ARTIFACT_KIND, "pipeline": estimator}
        test_results[name] = _metrics(y_test, predict_bundle(candidate_bundle, X_test))
    output_path.mkdir(parents=True, exist_ok=False)
    bundle = {
        "artifact_kind": ARTIFACT_KIND,
        "selection": selected,
        "pipeline": candidates[selected],
    }
    joblib.dump(bundle, output_path / "model.joblib")
    learned_bundle = {
        "artifact_kind": ARTIFACT_KIND,
        "selection": learned_selected,
        "pipeline": candidates[learned_selected],
    }
    joblib.dump(learned_bundle, output_path / "learned-model.joblib")
    source_hash = file_sha256(input_path)
    package_root = Path(pickplace_ai.__file__).resolve().parents[2]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=package_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "artifact_kind": ARTIFACT_KIND,
        "release_status": "retrospective_research_only",
        "source_kind": source_kind,
        "input_file": input_path.name,
        "input_sha256": source_hash,
        "raw_source_manifest": source_manifest,
        "raw_source_manifest_sha256": file_sha256(Path(manifest_path)) if manifest_path else None,
        "git_commit": git_commit,
        "runner_sha256": file_sha256(Path(__file__)),
        "feature_code_sha256": file_sha256(package_root / "src/pickplace_ai/features.py"),
        "model_sha256": file_sha256(output_path / "model.joblib"),
        "learned_model_sha256": file_sha256(output_path / "learned-model.joblib"),
        "snapshot_available_at": snapshot_at.isoformat(),
        "all_partition_information_time": snapshot_at.isoformat(),
        "created_at": datetime.now(UTC).isoformat(),
        "target": "sales[t+1] / sales[t-3] - 1",
        "input_observations": len(observations),
        "eligible_rows": len(frame),
        "feature_columns": FEATURE_COLUMNS,
        "quarter_splits": quarter_sets,
        "embargo_quarters": embargoes,
        "rows_by_partition": {name: len(value) for name, value in split.items()},
        "selection": selected,
        "selection_rule": "minimum validation MAE, including zero and recent baselines",
        "learned_model_selection": learned_selected,
        "learned_selection_rule": "minimum validation MAE among ridge and hist only",
        "prediction_floor": -1.0,
        "prediction_method": (
            "Use retrospective_experiment.predict_bundle with the saved trusted joblib artifact"
        ),
        "seed": 42,
        "versions": package_versions(),
        "limitations": [
            "Retrospective evaluation on the current snapshot, "
            "not historical point-in-time backtesting.",
            "Prior vintages and past publication dates are unverified; "
            "observations may include revisions.",
            "One-quarter embargo is a conservative evaluation convention, "
            "not verified historical availability.",
            "Not approved for production, service predictions, "
            "or public claims of predictive accuracy.",
            "Quarterly estimated sales are an aggregate measure; "
            "not individual sales or business success.",
            "Population and area features are missing in the current sales/store adapter.",
        ],
    }
    evaluation = {
        "artifact_kind": ARTIFACT_KIND,
        "source_kind": source_kind,
        "snapshot_available_at": snapshot_at.isoformat(),
        "all_partition_information_time": snapshot_at.isoformat(),
        "selection": selected,
        "learned_model_selection": learned_selected,
        "validation": validation_results,
        "test": test_results,
        "units": {
            "mae": "growth ratio",
            "mae_pp": "percentage points",
            "rmse": "growth ratio",
        },
        "interpretation": "Retrospective diagnostic on revised data; not a real-time backtest.",
    }
    for name, value in [("metadata.json", metadata), ("evaluation.json", evaluation)]:
        (output_path / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if manifest_path:
        (output_path / "source-manifest.json").write_bytes(Path(manifest_path).read_bytes())
    (output_path / "MODEL_CARD.md").write_text(
        "# 서울 상권 후향 탐색 모델\n\n"
        "**연구 전용, 운영 사용 금지. 실시간 과거 백테스트나 서비스 성능이 아니다.**\n\n"
        f"- 원본 유형: {source_kind}\n- 스냅샷 확보일: {snapshot_at}\n"
        f"- 입력 SHA-256: `{source_hash}`\n- 선택 모델: {selected}\n"
        f"- 머신러닝 후보 중 선택: {learned_selected} (별도 learned-model.joblib)\n"
        + (
            "- 전체 검증 후보 중 기준 모델이 선택됐으며 "
            "머신러닝 후보를 강제로 우승자로 바꾸지 않았다.\n"
            if selected in {"zero_growth", "recent_growth"}
            else "- 전체 검증 후보 중 머신러닝 모델이 선택됐다.\n"
        )
        + f"- 샘플: {len(frame)}\n- 대상: 다음 분기 전년 동기 대비 추정매출 성장률\n"
        "- 현재 확보한 수정본만 이용하므로 과거 공개 시점과 과거 자료 버전은 검증되지 않았다.\n"
        "- 검증·시험 값은 학습 당시 실제로 확보 가능했던 자료를 재현한 성능이 아니다.\n"
        "- 구간별 분리와 1분기 간격은 시간순 진단일 뿐 공개일 검증을 대체하지 못한다.\n"
        "- API 모델 스키마를 충족하지 않으므로 기존 서비스에 연결할 수 없다.\n"
        "- 연구용 추론은 신뢰 가능한 joblib 파일을 불러온 뒤 "
        "retrospective_experiment.predict_bundle을 사용한다. "
        "이 함수는 평가와 동일하게 예측값을 -1 이상으로 제한한다.\n"
        + (
            "- 출처: 서울특별시·서울신용보증재단, 서울시 상권분석서비스. "
            "원본 링크와 개별 SHA-256은 source-manifest.json 참조.\n"
            if source_kind == "official"
            else "- 합성 데이터 실험이며 실제 상권 성능이 아니다.\n"
        )
        + "- 추후 당시 버전과 공개일을 확보해 전향적 평가가 필요하다.\n",
        encoding="utf-8",
    )
    return metadata, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-at", type=date.fromisoformat, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    args = parser.parse_args()
    metadata, _ = run(
        args.input,
        args.output,
        args.snapshot_at,
        manifest_path=args.manifest,
        raw_root=args.raw_root,
    )
    print(
        json.dumps(
            {
                "artifact_kind": metadata["artifact_kind"],
                "selection": metadata["selection"],
                "rows_by_partition": metadata["rows_by_partition"],
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
