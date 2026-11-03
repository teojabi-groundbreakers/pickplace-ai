import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from pickplace_ai import features
from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.features import FEATURE_COLUMNS
from pickplace_ai.research_api import ResearchSettings, create_app
from pickplace_ai.research_models import MODEL_KIND, RESEARCH_STATUS, SUMMARY_KIND, ResearchRegistry
from pickplace_ai.training import RecentGrowthRegressor


@pytest.fixture
def registry_dir(tmp_path):
    root = tmp_path / "research"
    folder = root / "industries" / "CS100010"
    folder.mkdir(parents=True)
    training = pd.DataFrame([{key: 0.1 for key in FEATURE_COLUMNS}])
    selected = RecentGrowthRegressor().fit(training)
    learned = DummyRegressor(strategy="constant", constant=0.2).fit(training, [0.0])
    for name, model, selection in [
        ("model.joblib", selected, "recent_growth"),
        ("learned-model.joblib", learned, "dummy"),
    ]:
        joblib.dump(
            {"artifact_kind": MODEL_KIND, "selection": selection, "pipeline": model}, folder / name
        )
    metadata = {
        "artifact_kind": MODEL_KIND,
        "release_status": RESEARCH_STATUS,
        "industry_code": "CS100010",
        "source_kind": "official",
        "snapshot_available_at": "2026-11-03",
        "all_partition_information_time": "2026-11-03",
        "all_industries_input_sha256": "a" * 64,
        "feature_columns": FEATURE_COLUMNS,
        "feature_code_sha256": file_sha256(Path(features.__file__)),
        "versions": package_versions(),
        "quarter_splits": {"test": ["2025Q4"]},
        "raw_source_manifest": {"region_code_system": "seoul-adstrd", "industry_code": "CS100010"},
        "target": "sales[t+1] / sales[t-3] - 1",
        "prediction_floor": -1.0,
        "selection": "recent_growth",
        "learned_model_selection": "dummy",
        "model_sha256": file_sha256(folder / "model.joblib"),
        "learned_model_sha256": file_sha256(folder / "learned-model.joblib"),
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    entry = {
        "industry_code": "CS100010",
        "industry_name": "가상의 테스트 업종",
        "status": "completed",
        "artifact_status": "complete",
        "quarter_splits": {"test": ["2025Q4"]},
        "selection": "recent_growth",
        "learned_model_selection": "dummy",
        "warnings": [],
    }
    summary = {
        "artifact_kind": SUMMARY_KIND,
        "release_status": RESEARCH_STATUS,
        "source_kind": "official",
        "snapshot_available_at": "2026-11-03",
        "input_sha256": "a" * 64,
        "industries": [entry],
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return root


@pytest.fixture
def payload():
    periods = ["2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3"]
    history = [
        {
            "region_code": "11110515",
            "region_code_system": "seoul-adstrd",
            "industry_code": "CS100010",
            "period": period,
            "available_at": "2026-11-03",
            "source_kind": "official",
            "source_dataset": "synthetic-test-fixture",
            "sales_unit": "KRW",
            "sales_amount": 100.0 + 10 * i,
            "store_count": 10,
            "opening_count": 1,
            "closing_count": 1,
        }
        for i, period in enumerate(periods)
    ]
    return {
        "schema_version": "1.0",
        "region_code": "11110515",
        "region_code_system": "seoul-adstrd",
        "industry_code": "CS100010",
        "base_period": "2025Q3",
        "as_of": "2026-11-03",
        "source_kind": "official",
        "history": history,
    }


@pytest.fixture
def client(registry_dir):
    with TestClient(create_app(ResearchSettings(registry_dir, "test-key"))) as client:
        yield client


def test_key_required_to_start(registry_dir):
    with pytest.raises(ValueError, match="API_KEY"):
        with TestClient(create_app(ResearchSettings(registry_dir, ""))):
            pass


def test_health_and_authentication(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["registry_ready"] is True
    assert health.json()["model_artifacts_verified"] is False
    assert health.json()["mode"] == "retrospective_research_only"
    assert client.get("/research/industries").status_code == 401
    assert client.get("/research/industries", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/research/industries", headers=[(b"X-API-Key", b"\xff")]).status_code == 401
    industries = client.get("/research/industries", headers={"X-API-Key": "test-key"}).json()
    assert [row["industry_code"] for row in industries["industries"]] == ["CS100010"]


@pytest.mark.parametrize(
    "variant,expected,selection", [("selected", 0.4, "recent_growth"), ("learned", 0.2, "dummy")]
)
def test_historical_replay_is_clearly_labelled(client, payload, variant, expected, selection):
    payload["model_variant"] = variant
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "retrospective_research_only"
    assert body["estimated_yoy_growth"] == pytest.approx(expected)
    assert body["target_period"] == "2025Q4"
    assert body["model_choice"] == selection
    assert len(body["model_sha256"]) == 64
    assert body["feature_schema_version"] == "1.0"
    assert body["region_scope_verified"] is False
    assert body["input_values_verified"] is False
    assert "not a live forecast" in body["warnings"][0]


def test_as_of_cannot_precede_snapshot(client, payload):
    payload["as_of"] = "2026-01-01"
    for row in payload["history"]:
        row["available_at"] = "2026-01-01"
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 422
    assert "precedes the research snapshot" in response.json()["detail"]


def test_request_cannot_backdate_snapshot_data(client, payload):
    payload["history"][0]["available_at"] = "2024-12-31"
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 422
    assert "history availability precedes" in response.json()["detail"]


def test_only_test_quarters_can_be_replayed(client, payload):
    payload["base_period"] = "2025Q2"
    payload["history"] = payload["history"][:-1]
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 422
    assert "outside" in response.json()["detail"]


def test_current_target_quarter_is_not_a_forecast(client, payload):
    payload["base_period"] = "2026Q3"
    payload["history"][-1]["period"] = "2026Q3"
    payload["history"][-1]["available_at"] = "2026-11-03"
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 422
    assert "Only completed historical" in response.json()["detail"]


def test_history_requires_five_adjacent_quarters(client, payload):
    payload["history"].pop(1)
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 422
    assert response.json()["detail"] == "insufficient_history"


def test_unknown_or_traversal_code_cannot_select_a_path(client, payload):
    for code in ("CS999999", "../../etc"):
        payload["industry_code"] = code
        for row in payload["history"]:
            row["industry_code"] = code
        response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
        assert response.status_code == 404


def test_wrong_source_and_code_system_rejected(client, payload):
    payload["region_code_system"] = "h10"
    for row in payload["history"]:
        row["region_code_system"] = "h10"
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 422


def test_obviously_non_seoul_region_is_rejected(client, payload):
    payload["region_code"] = "26110515"
    for row in payload["history"]:
        row["region_code"] = "26110515"
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 422


def test_corrupt_split_metadata_is_unavailable(client, registry_dir, payload):
    path = registry_dir / "industries" / "CS100010" / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata["quarter_splits"] = []
    path.write_text(json.dumps(metadata))
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 503


@pytest.mark.parametrize("kind", ["feature_code", "library_version", "target", "scope"])
def test_incompatible_model_metadata_is_unavailable(client, registry_dir, payload, kind):
    path = registry_dir / "industries" / "CS100010" / "metadata.json"
    metadata = json.loads(path.read_text())
    if kind == "feature_code":
        metadata["feature_code_sha256"] = "a" * 64
    elif kind == "library_version":
        metadata["versions"]["scikit-learn"] = "0.0.0"
    elif kind == "target":
        metadata["target"] = "unrelated"
    else:
        metadata["raw_source_manifest"]["industry_code"] = "CS999999"
    path.write_text(json.dumps(metadata))
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 503


def test_changed_model_is_detected_even_when_cached(client, registry_dir, payload):
    headers = {"X-API-Key": "test-key"}
    assert client.post("/research/predict", json=payload, headers=headers).status_code == 200
    (registry_dir / "industries" / "CS100010" / "model.joblib").write_bytes(b"modified")
    assert client.post("/research/predict", json=payload, headers=headers).status_code == 503


def test_symlink_model_is_rejected(client, registry_dir, payload, tmp_path):
    model = registry_dir / "industries" / "CS100010" / "model.joblib"
    outside = tmp_path / "outside.joblib"
    outside.write_bytes(model.read_bytes())
    model.unlink()
    model.symlink_to(outside)
    response = client.post("/research/predict", json=payload, headers={"X-API-Key": "test-key"})
    assert response.status_code == 503


def test_missing_registry_does_not_claim_ready(tmp_path):
    with TestClient(create_app(ResearchSettings(tmp_path / "missing", "key"))) as client:
        assert client.get("/health").json()["registry_ready"] is False
        assert client.get("/research/industries", headers={"X-API-Key": "key"}).status_code == 503


def test_corrupt_summary_does_not_claim_ready(registry_dir):
    path = registry_dir / "summary.json"
    data = json.loads(path.read_text())
    data["industries"][0]["quarter_splits"] = []
    path.write_text(json.dumps(data))
    with TestClient(create_app(ResearchSettings(registry_dir, "key"))) as client:
        assert client.get("/health").json()["registry_ready"] is False
        assert client.get("/research/industries", headers={"X-API-Key": "key"}).status_code == 503


def test_cache_is_bounded(registry_dir):
    registry = ResearchRegistry(registry_dir, max_cache=1)
    registry.load("CS100010", "selected")
    registry.load("CS100010", "learned")
    assert len(registry._cache) == 1
