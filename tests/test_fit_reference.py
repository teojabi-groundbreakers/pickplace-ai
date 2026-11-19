"""허구 참조표로 날짜가 맞는 상대 지표와 선택적 API 연결을 검증한다."""

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pickplace_ai import api, fit_reference
from pickplace_ai.artifacts import file_sha256
from pickplace_ai.schemas import PredictionRequest


def write_reference(root: Path, rows=None, **updates):
    root.mkdir()
    rows = rows or [
        ("11110511", "CS100010", 10, 1000, 1_000_000),
        ("11110512", "CS100010", 20, 6000, 2_000_000),
        ("11110513", "CS100005", 5, 100, 1_000_000),
    ]
    path = root / "reference.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fit_reference.ROW_FIELDS)
        writer.writerows(rows)
    manifest = {
        "kind": fit_reference.KIND,
        "region_code_system": "seoul-adstrd",
        "source_kind": "official",
        "reference_date": "2025-06-30",
        "available_at": "2026-11-20",
        "base_period": "2025Q2",
        "area_reference_date": "2025-06-30",
        "population_reference_date": "2025-06-30",
        "mapping_reference_date": "2025-06-30",
        "stores_period": "2025Q2",
        "row_count": len(rows),
        "reference_sha256": file_sha256(path),
        "sources": {
            key: {"url": f"https://fixture.invalid/{key}", "sha256": "a" * 64}
            for key in ("area", "population", "mapping", "stores")
        },
        "fixture": "constructed test table, not actual official observations",
    }
    manifest.update(updates)
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def payload(region="11110511", industry="CS100010", period="2025Q2", as_of="2026-11-20"):
    scope = {
        "region_code": region,
        "region_code_system": "seoul-adstrd",
        "industry_code": industry,
        "source_kind": "official",
    }
    return {
        **scope,
        "base_period": period,
        "as_of": as_of,
        "history": [
            {
                **scope,
                "period": period,
                "available_at": "2026-07-01" if period == "2026Q2" else "2025-07-01",
                "source_dataset": "fictional request deliberately unlike the stored reference",
                "store_count": 999,
                "resident_population": 999_999,
                "area_km2": 999,
            }
        ],
    }


def test_indicators_use_reference_values_and_per_industry_tied_ranks(tmp_path):
    model = fit_reference.FitReference(write_reference(tmp_path / "reference"))
    result = model.indicators(PredictionRequest.model_validate(payload()))
    assert result["store_count"] == 10
    assert result["resident_population"] == 1000
    assert result["area_km2"] == 1.0
    assert result["cohort_size"] == 2
    assert result["reference_date"] == "2025-06-30"
    assert result["reference_available_at"] == "2026-11-20"
    assert result["residents_per_store"]["value"] == 100
    assert result["residents_per_store"]["percentile"] == 0
    assert result["stores_per_km2"]["value"] == 10
    assert result["stores_per_km2"]["percentile"] == 50
    assert "score" not in result


def test_single_area_has_values_but_no_fabricated_percentile(tmp_path):
    model = fit_reference.FitReference(write_reference(tmp_path / "reference"))
    result = model.indicators(
        PredictionRequest.model_validate(payload(region="11110513", industry="CS100005"))
    )
    assert result["cohort_size"] == 1
    for field in fit_reference.UNITS:
        assert result[field]["status"] == "available"
        assert result[field]["percentile"] is None
        assert result[field]["percentile_reason"] == "insufficient_reference_areas"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"region_code": "11110599"}, "fit_scope_not_in_reference"),
        ({"industry_code": "CS100008"}, "fit_scope_not_in_reference"),
        ({"as_of": "2026-11-19"}, "fit_reference_after_as_of"),
        ({"base_period": "2026Q2"}, "fit_reference_period_mismatch"),
        ({"region_code_system": "MOIS_H10"}, "unsupported_fit_reference_scope"),
        ({"source_kind": "synthetic"}, "unsupported_fit_reference_scope"),
    ],
)
def test_unavailable_scope_period_and_knowledge_date_preserve_null(tmp_path, change, reason):
    model = fit_reference.FitReference(write_reference(tmp_path / "reference"))
    raw = payload(period=change.get("base_period", "2025Q2"))
    raw.update(change)
    for row in raw["history"]:
        for key in ("region_code", "industry_code", "region_code_system", "source_kind"):
            row[key] = raw[key]
    result = model.indicators(PredictionRequest.model_validate(raw))
    assert result["store_count"] is None and result["cohort_size"] is None
    for field in fit_reference.UNITS:
        assert result[field]["status"] == "unavailable"
        assert result[field]["value"] is None
        assert result[field]["percentile"] is None
        assert result[field]["reason"] == reason


@pytest.mark.parametrize(
    "change",
    [
        {"area_reference_date": "2026-06-30"},
        {"population_reference_date": "2025-05-31"},
        {"mapping_reference_date": "2025-07-01"},
        {"reference_date": "2025-06-29"},
        {"available_at": "2025-06-29"},
        {"stores_period": "2026Q2"},
        {"source_kind": "synthetic"},
    ],
)
def test_manifest_dates_and_source_kind_must_align(tmp_path, change):
    root = write_reference(tmp_path / "reference", **change)
    with pytest.raises(ValueError):
        fit_reference.FitReference(root)


def test_digest_duplicates_invalid_area_and_count_are_rejected(tmp_path):
    root = write_reference(tmp_path / "corrupt")
    path = root / "reference.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="digest"):
        fit_reference.FitReference(root)
    duplicate = ("11110511", "CS100010", 10, 1000, 1_000_000)
    root = write_reference(tmp_path / "duplicate", rows=[duplicate, duplicate])
    with pytest.raises(ValueError, match="duplicate"):
        fit_reference.FitReference(root)
    root = write_reference(tmp_path / "area", rows=[("11110511", "CS100010", 10, 1000, 0)])
    with pytest.raises(ValueError):
        fit_reference.FitReference(root)
    root = write_reference(tmp_path / "count", row_count=999)
    with pytest.raises(ValueError, match="count"):
        fit_reference.FitReference(root)


def test_api_optional_reference_provides_descriptive_values_without_model(tmp_path):
    root = write_reference(tmp_path / "reference")
    settings = api.Settings(model_dir=tmp_path / "absent-model", fit_reference_dir=root)
    with TestClient(api.create_app(settings)) as client:
        response = client.post("/predict", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["growth"]["estimated_yoy_growth"] is None
    assert body["fit_indicators"]["residents_per_store"]["value"] == 100
    assert body["fit_indicators"]["stores_per_km2"]["percentile"] == 50
    assert body["fit_indicators"]["store_count"] == 10


@pytest.mark.parametrize("configured", [False, True])
def test_api_missing_or_invalid_reference_does_not_fall_back_to_request(tmp_path, configured):
    settings = api.Settings(
        model_dir=tmp_path / "absent-model",
        fit_reference_dir=tmp_path / "absent-reference" if configured else None,
    )
    with TestClient(api.create_app(settings)) as client:
        response = client.post("/predict", json=payload())
    assert response.status_code == 200
    expected = "fit_reference_invalid" if configured else "fit_reference_not_configured"
    result = response.json()["fit_indicators"]
    assert result["resident_population"] is None
    for field in fit_reference.UNITS:
        assert result[field]["value"] is None and result[field]["percentile"] is None
        assert result[field]["reason"] == expected


def test_environment_reference_configuration_is_optional(tmp_path, monkeypatch):
    monkeypatch.delenv("PICKPLACE_SEOUL_FIT_REFERENCE_DIR", raising=False)
    assert api.Settings.from_env().fit_reference_dir is None
    monkeypatch.setenv("PICKPLACE_SEOUL_FIT_REFERENCE_DIR", str(tmp_path / "reference"))
    assert api.Settings.from_env().fit_reference_dir == tmp_path / "reference"
