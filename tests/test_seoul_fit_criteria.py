"""허구 관측으로 적합도 기준의 표본 내 비교와 자료 보존을 검증한다."""

import json
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pickplace_ai import scoring, seoul_stats_ingest
from pickplace_ai.artifacts import file_sha256
from pickplace_ai.schemas import Observation
from scripts import compare_seoul_fit_criteria as study


@pytest.fixture
def scored_frame():
    rows = []
    for factor, industry in enumerate(study.audit.SERVICE_CODES, start=1):
        for index in range(1, 5):
            observation = Observation(
                region_code=f"1111051{index}",
                region_code_system="seoul-adstrd",
                industry_code=industry,
                period="2026Q2",
                available_at=date(2026, 11, 9),
                source_kind="synthetic",
                source_dataset="fictional criterion fixture, not an actual Seoul region",
                store_count=index * 10 * factor,
                resident_population=1000 * index**2,
                area_km2=1.0,
            )
            fit = scoring.diagnose([observation], "2026Q2")["fit"]
            components = {value["name"]: value for value in fit["components"]}
            rows.append(
                {
                    "period": observation.period,
                    "region_code": observation.region_code,
                    "industry_code": industry,
                    "fit_score": fit["score"],
                    "residents_per_store": components["residents_per_store"]["value"],
                    "stores_per_km2": components["store_density"]["value"],
                }
            )
    return pd.DataFrame.from_records(rows)


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshot"
    path.mkdir()
    database = path / "snapshot.sqlite3"
    with sqlite3.connect(database) as connection:
        seoul_stats_ingest._schema(connection)
        for index in range(1, 5):
            region = f"1111051{index}"
            connection.execute(
                "INSERT INTO resident VALUES (?,?,?)", ("2026Q2", region, 1000 * index**2)
            )
            for factor, industry in enumerate(study.audit.SERVICE_CODES, start=1):
                connection.execute(
                    "INSERT INTO stores VALUES (?,?,?,?,?,?)",
                    ("2026Q2", region, industry, index * 10 * factor, 0, 0),
                )
    manifest = {
        "schema_version": 1,
        "mode": "seoul_descriptive_research_only",
        "created_at_utc": "2026-11-09T00:00:00+00:00",
        "database_sha256": file_sha256(database),
        "datasets": {name: {"periods": ["2026Q2"]} for name in ("stores", "resident")},
        "fixture": "constructed test database, not actual Seoul data",
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    area = tmp_path / "area.csv"
    pd.DataFrame(
        {"행정동_코드": [f"1111051{index}" for index in range(1, 5)], "영역_면적": [1e6] * 4}
    ).to_csv(area, index=False)
    return path, area


def test_compares_actual_scoring_with_linear_quantiles_on_common_scopes(scored_frame):
    criteria, components = study.compare(scored_frame)
    assert len(criteria) == 9 and len(components) == 6
    assert set(criteria.criteria) == set(study.CRITERIA)
    assert set(criteria.matched_rows) == {4}
    industry = study.audit.SERVICE_CODES[0]
    observed = scored_frame.loc[scored_frame.industry_code == industry]
    rows = criteria.loc[criteria.industry_code == industry].set_index("criteria")
    assert rows.loc["fixed_500_50", "score_median"] == observed.fit_score.median()
    assert rows.loc["industry_q75_q75", "residents_per_store_threshold"] == pytest.approx(325)
    assert rows.loc["industry_q75_q75", "density_threshold"] == pytest.approx(32.5)
    assert rows.loc["industry_q90_q90", "residents_per_store_threshold"] == pytest.approx(370)
    assert rows.loc["industry_q90_q90", "density_threshold"] == pytest.approx(37)
    assert not any("score" in column for column in components.columns)
    assert set(components.component) == {"residents_per_store", "stores_per_km2"}
    np.testing.assert_allclose(components.percentile_mean, 50)


def test_reference_quantiles_are_fit_after_matching_all_industries(scored_frame):
    missing = (scored_frame.industry_code == study.audit.SERVICE_CODES[-1]) & (
        scored_frame.region_code == "11110514"
    )
    criteria, _ = study.compare(scored_frame.loc[~missing])
    assert set(criteria.matched_rows) == {3}
    row = criteria.loc[
        (criteria.industry_code == study.audit.SERVICE_CODES[0])
        & (criteria.criteria == "industry_q75_q75")
    ].iloc[0]
    assert row.residents_per_store_threshold == pytest.approx(250)


def test_population_sensitivity_keeps_fitted_reference_thresholds(scored_frame):
    criteria, _ = study.compare(scored_frame)
    quantiles = criteria.loc[criteria.criteria != "fixed_500_50"]
    # 기준도 동시에 10% 늘렸다면 사라졌을 변화가 고정 기준에서는 남아야 한다.
    assert (quantiles.residents_x1_1_median_shift > 0).all()
    assert (quantiles.residents_x1_1_unchanged_rows < quantiles.matched_rows).all()


def test_order_independence_and_lexicographic_tie_resolution(scored_frame):
    expected, expected_components = study.compare(scored_frame)
    shuffled, shuffled_components = study.compare(scored_frame.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(expected, shuffled)
    pd.testing.assert_frame_equal(expected_components, shuffled_components)
    group = scored_frame.loc[scored_frame.industry_code == study.audit.SERVICE_CODES[0]]
    tied = pd.Series(50.0, index=group.index)
    assert study._top_regions(group.iloc[::-1], tied, 1) == {"11110511"}


def test_future_proxy_values_do_not_change_descriptive_criteria(scored_frame):
    expected = study.compare(scored_frame)
    changed = scored_frame.assign(next_sales_yoy=1e9, sales_per_store=1e15, risk_score=99)
    actual = study.compare(changed)
    for first, second in zip(expected, actual, strict=True):
        pd.testing.assert_frame_equal(first, second)


def test_duplicate_invalid_and_wrong_quarter_inputs_are_rejected(scored_frame):
    with pytest.raises(ValueError, match="Duplicate"):
        study.compare(pd.concat([scored_frame, scored_frame.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="2026Q2"):
        study.compare(scored_frame.assign(period="2025Q4"))
    invalid = scored_frame.copy()
    invalid.loc[invalid.index[0], "stores_per_km2"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        study.compare(invalid)


def test_changed_baseline_is_rejected_instead_of_relabelled(scored_frame):
    changed = scored_frame.copy()
    changed.loc[changed.index[0], "fit_score"] += 1
    with pytest.raises(ValueError, match="draft-v0.1"):
        study.compare(changed)


def test_run_emits_aggregate_only_and_preserves_input_files(snapshot, tmp_path):
    path, area = snapshot
    inputs = [path / "snapshot.sqlite3", path / "manifest.json", area]
    before = {value: file_sha256(value) for value in inputs}
    output = tmp_path / "result"
    summary = study.run(path, area, output)
    assert summary["matched_regions"] == 4
    assert summary["in_sample_comparison_not_validation"] is True
    assert summary["outcome_validated"] is False
    assert summary["area_effective_period_verified"] is False
    assert summary["percentile_components_combined"] is False
    assert {value: file_sha256(value) for value in inputs} == before
    assert set(p.name for p in output.iterdir()) == {
        "criteria-comparison.csv",
        "component-percentiles.csv",
        "summary.json",
    }
    for name, digest in summary["outputs_sha256"].items():
        assert file_sha256(output / name) == digest
        assert "region_code" not in pd.read_csv(output / name).columns
    with pytest.raises(FileExistsError):
        study.run(path, area, output)


@pytest.mark.parametrize("created_at", ["2026-11-09T00:00:00", "2026-06-29T00:00:00+00:00"])
def test_invalid_acquisition_date_cannot_publish(snapshot, tmp_path, created_at):
    path, area = snapshot
    manifest = path / "manifest.json"
    content = json.loads(manifest.read_text())
    content["created_at_utc"] = created_at
    manifest.write_text(json.dumps(content))
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="timezone|precedes"):
        study.run(path, area, output)
    assert not output.exists()


def test_input_change_during_comparison_prevents_publication(snapshot, tmp_path, monkeypatch):
    path, area = snapshot
    original = study.compare

    def changed(frame):
        result = original(frame)
        area.write_text(area.read_text() + "\n")
        return result

    monkeypatch.setattr(study, "compare", changed)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="Input changed"):
        study.run(path, area, output)
    assert not output.exists()


def test_snapshot_digest_mismatch_is_rejected(snapshot, tmp_path):
    path, area = snapshot
    (path / "snapshot.sqlite3").write_bytes(b"not the recorded database")
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="matching statistics"):
        study.run(path, area, output)
    assert not output.exists()
