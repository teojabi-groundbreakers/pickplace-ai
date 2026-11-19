"""합성 자료로 순차 평가의 시점·표본·선택 구간 분리를 검증한다."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from pickplace_ai.data import write_observations
from pickplace_ai.periods import shift_quarter
from pickplace_ai.schemas import Observation
from scripts import evaluate_seoul_rolling as rolling
from scripts import near_term_forecast as near


@pytest.fixture
def observations():
    return [
        Observation(
            region_code=f"1111051{region}",
            region_code_system="seoul-adstrd",
            industry_code=industry,
            period=shift_quarter("2021Q1", offset),
            available_at=date(2026, 11, 4),
            source_kind="official",
            source_dataset="fictional fixture, not actual Seoul data",
            sales_amount=float(1_000_000 + offset * 10_000 + region * 1_000),
            store_count=20,
        )
        for region in (5, 6)
        for industry in (*rolling.SERVICE_INDUSTRIES, "CS100001")
        for offset in range(22)
    ]


@pytest.fixture
def frame(observations):
    return near.build_frames(
        observations,
        date(2026, 11, 4),
        [shift_quarter("2022Q1", offset) for offset in range(16)],
    )[0]


def simple_candidates():
    return {
        "zero": DummyRegressor(strategy="constant", constant=0),
        "positive": DummyRegressor(strategy="constant", constant=0.1),
    }


def test_training_targets_respect_explicit_embargo_and_evaluation_scope(frame):
    train, test, eligible, cutoff = rolling.split_fold(frame, "2025Q1", 2, 1)
    assert cutoff == "2024Q4"
    assert train.target_period.max() <= cutoff
    assert not set(train.target_period) & set(test.target_period)
    assert set(test.industry_code) == set(rolling.SERVICE_INDUSTRIES)
    assert "CS100001" in set(train.industry_code)
    assert set(test.target_period) == {"2025Q3"}
    assert eligible == len(test)


def test_reporting_targets_do_not_select_candidate(frame):
    origins = ("2024Q1", "2024Q2", "2024Q3", "2024Q4")
    _, _, selected = rolling.evaluate(frame, origins, candidate_factory=simple_candidates)
    changed = frame.copy()
    changed.loc[changed.base_period.isin(origins[-2:]), "target"] = 100_000.0
    _, aggregate, after = rolling.evaluate(changed, origins, candidate_factory=simple_candidates)
    assert selected == after
    assert aggregate.loc[aggregate.phase == "reporting", "mae_pp"].min() > 1_000_000


def test_metrics_use_same_rows_and_preserve_missing_target_coverage(frame):
    changed = frame.copy()
    missing = (
        (changed.base_period == "2024Q4")
        & (changed.horizon == 1)
        & (changed.industry_code == "CS100010")
        & (changed.region_code == "11110515")
    )
    changed.loc[missing, "target"] = np.nan
    origins = ("2024Q1", "2024Q2", "2024Q3", "2024Q4")
    base_counts = {(origin, code): 2 for origin in origins for code in rolling.SERVICE_INDUSTRIES}
    folds, _, _ = rolling.evaluate(
        changed, origins, base_counts=base_counts, candidate_factory=simple_candidates
    )
    part = folds.loc[
        (folds.base_period == "2024Q4") & (folds.horizon == 1) & (folds.industry_code == "CS100010")
    ]
    assert set(part.rows) == {1}
    assert set(part.missing_target_rows) == {1}
    assert set(part.evaluation_coverage) == {0.5}
    assert part.model.nunique() == 2


@pytest.mark.parametrize("report_origins", [0, 4])
def test_requires_distinct_selection_and_reporting_origins(frame, report_origins):
    with pytest.raises(ValueError, match="selection and one reporting"):
        rolling.evaluate(
            frame,
            ("2024Q1", "2024Q2", "2024Q3", "2024Q4"),
            report_origins=report_origins,
        )


def test_duplicate_scopes_are_rejected(frame):
    with pytest.raises(ValueError, match="Duplicate scope"):
        rolling.evaluate(pd.concat([frame, frame.iloc[:1]], ignore_index=True))


def test_nonfinite_predictions_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        rolling.error_metrics([0.1], [np.inf])


def test_runner_preserves_input_and_does_not_overwrite_output(observations, tmp_path, monkeypatch):
    source = tmp_path / "fixture.csv"
    write_observations(observations, source)
    before = source.read_bytes()
    output = tmp_path / "result"
    monkeypatch.setattr(rolling, "candidates", simple_candidates)
    summary = rolling.run(
        source,
        output,
        date(2026, 11, 4),
        origins=("2024Q1", "2024Q2", "2024Q3", "2024Q4"),
    )
    assert source.read_bytes() == before
    assert summary["training_scope"] == "all"
    assert summary["selection_origins"] == ["2024Q1", "2024Q2"]
    assert summary["reporting_origins"] == ["2024Q3", "2024Q4"]
    assert set(path.name for path in output.iterdir()) == {
        "fold-metrics.csv",
        "aggregate-metrics.csv",
        "summary.json",
    }
    with pytest.raises(FileExistsError):
        rolling.run(source, output, date(2026, 11, 4))


def test_runner_refuses_data_acquired_after_analysis_date(observations, tmp_path):
    source = tmp_path / "fixture.csv"
    write_observations(observations, source)
    with pytest.raises(ValueError, match="not acquired"):
        rolling.run(source, tmp_path / "result", date(2026, 11, 3))


def test_runner_rejects_input_changed_during_evaluation(observations, tmp_path, monkeypatch):
    source = tmp_path / "fixture.csv"
    write_observations(observations, source)
    before = source.read_bytes()
    output = tmp_path / "result"
    evaluate = rolling.evaluate

    def evaluate_then_change_input(*args, **kwargs):
        result = evaluate(*args, **kwargs, candidate_factory=simple_candidates)
        source.write_bytes(before + b"\n")
        return result

    monkeypatch.setattr(rolling, "evaluate", evaluate_then_change_input)
    with pytest.raises(ValueError, match="Input changed during evaluation"):
        rolling.run(
            source,
            output,
            date(2026, 11, 4),
            origins=("2024Q1", "2024Q2", "2024Q3", "2024Q4"),
        )
    assert source.read_bytes() != before
    assert not output.exists()


def test_hist_candidate_trains_on_common_transformed_features(frame):
    train, test, _, _ = rolling.split_fold(frame, "2025Q1", 1)
    model = rolling.candidates()["hist_absolute"]
    model.fit(near.matrix(train), train.target)
    estimates = near.predict(model, test)
    assert len(estimates) == len(test)
    assert np.isfinite(estimates).all()
