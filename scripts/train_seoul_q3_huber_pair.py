"""학습·검증 표본을 고정한 서울 2026Q3 비교용 모델을 만든다."""

import argparse
import json
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import pandas as pd

from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.data import read_observations
from pickplace_ai.periods import quarter_index, shift_quarter
from scripts import evaluate_seoul_next_quarter_tail as tail
from scripts import evaluate_seoul_rolling as rolling
from scripts import near_term_forecast as near

BASE = "2026Q2"
TARGET = shift_quarter(BASE, 1)
LABEL_CUTOFF = shift_quarter(BASE, -1)
COMPARISON = shift_quarter(TARGET, -4)
CHALLENGER = "huber_asinh_under_100m"
CUTOFF_KRW = tail.LOW_SALES_CUTOFFS[CHALLENGER]
FROZEN_KIND = "seoul_q3_huber_frozen_predictions_v1"
KIND = "seoul_q3_huber_pair_research_v1"
PREDICTION_COLUMNS = [tail.RAW, CHALLENGER]


def split_training(frame):
    one_quarter = frame.loc[frame.horizon.eq(1)]
    training = one_quarter.loc[
        one_quarter.target_period.le(LABEL_CUTOFF) & one_quarter.target.notna()
    ]
    forecast = one_quarter.loc[
        one_quarter.base_period.eq(BASE)
        & one_quarter.target_period.eq(TARGET)
        & one_quarter.industry_code.isin(rolling.SERVICE_INDUSTRIES)
    ]
    if (
        training.empty
        or training.target_period.max() != LABEL_CUTOFF
        or forecast.empty
        or forecast.target.notna().any()
        or set(forecast.industry_code) != set(rolling.SERVICE_INDUSTRIES)
        or not forecast.comparison_period.eq(COMPARISON).all()
        or forecast.duplicated(tail.SCOPE).any()
    ):
        raise ValueError(
            "Expected fixed 2026Q1 labels and unique unobserved 2026Q3 forecast scopes"
        )
    return training, forecast


def predict_pair(models, forecast):
    columns = [*tail.SCOPE, "target_period", "comparison_period", "comparison_sales"]
    result = forecast[columns].copy()
    result[tail.RAW] = near.predict(models[tail.RAW], forecast)
    transformed = pd.Series(near.predict(models[tail.ASINH], forecast), index=forecast.index)
    result["uses_transformed_model"] = result.comparison_sales.lt(CUTOFF_KRW)
    result[CHALLENGER] = transformed.where(result.uses_transformed_model, result[tail.RAW])
    return result.sort_values(tail.SCOPE).reset_index(drop=True)


def require_matching_predictions(actual, expected):
    # 코드, 분모와 적용 범위는 정확히 일치하고, 예측값만 CSV 부동소수점 오차를 허용한다.
    features = [name for name in expected if name not in PREDICTION_COLUMNS]
    try:
        pd.testing.assert_frame_equal(actual[features], expected[features], check_exact=True)
        pd.testing.assert_frame_equal(
            actual[PREDICTION_COLUMNS],
            expected[PREDICTION_COLUMNS],
            check_exact=False,
            atol=1e-12,
            rtol=0,
        )
    except AssertionError as exc:
        raise ValueError("Saved model forecasts differ from the frozen comparison") from exc


def run(input_path, frozen_dir, output):
    input_path, frozen_dir, output = map(Path, (input_path, frozen_dir, output))
    if output.exists() or output.is_symlink():
        raise FileExistsError("Model output already exists")
    frozen_summary = frozen_dir / "summary.json"
    frozen_csv = frozen_dir / "private-frozen-predictions.csv"
    hashes = {
        "snapshot": file_sha256(input_path),
        "frozen_summary": file_sha256(frozen_summary),
        "frozen_predictions": file_sha256(frozen_csv),
    }
    frozen = json.loads(frozen_summary.read_text(encoding="utf-8"))
    versions = package_versions()
    code_hashes = {
        "runner": file_sha256(Path(tail.__file__)),
        "rolling": file_sha256(Path(rolling.__file__)),
        "near_term": file_sha256(Path(near.__file__)),
        "features": file_sha256(Path(near.feature_record.__code__.co_filename)),
    }
    expected_metadata = {
        "artifact_kind": FROZEN_KIND,
        "base_period": BASE,
        "target_period": TARGET,
        "comparison_period": COMPARISON,
        "training_label_cutoff": LABEL_CUTOFF,
        "prediction_columns": PREDICTION_COLUMNS,
        "comparison_sales_cutoff_krw_exclusive": CUTOFF_KRW,
        "snapshot_sha256": hashes["snapshot"],
        "prediction_sha256": hashes["frozen_predictions"],
        "versions": versions,
        "code_sha256": code_hashes,
    }
    if any(frozen.get(name) != value for name, value in expected_metadata.items()):
        raise ValueError("Frozen comparison, source, code, or runtime differs")

    as_of = date.fromisoformat(frozen["snapshot_analysis_date"])
    observations = read_observations(input_path)
    if max(row.period for row in observations) != BASE or any(
        row.available_at > as_of
        or row.source_kind != "official"
        or row.region_code_system != "seoul-adstrd"
        for row in observations
    ):
        raise ValueError("Use the verified official Seoul snapshot through 2026Q2")
    earliest = shift_quarter(min(row.period for row in observations), 4)
    origins = tuple(
        shift_quarter(earliest, offset)
        for offset in range(quarter_index(BASE) - quarter_index(earliest) + 1)
    )
    frame, _ = near.build_frames(observations, as_of, origins)
    training, forecast = split_training(frame)
    if (
        len(training) != frozen["training_rows"]
        or len(forecast) != frozen["frozen_prediction_rows"]
    ):
        raise ValueError("Training or forecast rows differ from the frozen comparison")

    inputs = near.matrix(training)
    labels = training.target.to_numpy(float)
    models = rolling.huber_target_candidates()
    for model in models.values():
        model.fit(inputs, labels)

    expected = pd.read_csv(frozen_csv, dtype={"region_code": str})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seoul-q3-pair-", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        staged.mkdir()
        model_path = staged / "model.joblib"
        joblib.dump({"artifact_kind": KIND, "models": models}, model_path, compress=3)
        # 이 실행에서 생성한 임시 파일만 다시 불러온다.
        loaded = joblib.load(model_path)
        reproduced = predict_pair(loaded["models"], forecast)
        require_matching_predictions(reproduced, expected)
        if any(
            file_sha256(path) != hashes[name]
            for name, path in (
                ("snapshot", input_path),
                ("frozen_summary", frozen_summary),
                ("frozen_predictions", frozen_csv),
            )
        ):
            raise ValueError("Input changed during model training")
        metadata = {
            **expected_metadata,
            "artifact_kind": KIND,
            "release_status": "research_only",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "snapshot_analysis_date": as_of.isoformat(),
            "source_sha256": hashes,
            "training_script_sha256": file_sha256(Path(__file__)),
            "model_sha256": file_sha256(model_path),
            "training_rows": len(training),
            "replayed_forecast_rows": len(reproduced),
            "asinh_routed_rows": int(reproduced.uses_transformed_model.sum()),
            "max_replay_difference_pp": {
                name: float((reproduced[name] - expected[name]).abs().max() * 100)
                for name in PREDICTION_COLUMNS
            },
            "selected_reference_model": tail.RAW,
            "frozen_at_utc": frozen["frozen_at_utc"],
        }
        (staged / "summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError("Model output already exists")
        staged.rename(output)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.input, args.frozen_dir, args.output)
    print(json.dumps({"output": str(args.output), "rows": result["replayed_forecast_rows"]}))


if __name__ == "__main__":
    main()
