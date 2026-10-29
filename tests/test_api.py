"""Behavioral contract tests with explicitly fake data and model bundles."""

from datetime import timedelta
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import pickplace_ai.api as api
from pickplace_ai.periods import quarter_end


def request_payload(source_kind="official"):
    periods = ["2024Q1", "2024Q2", "2024Q3", "2024Q4", "2025Q1"]
    return {
        "region_code": "9999999999",
        "region_code_system": "test-h10",
        "industry_code": "coffee",
        "base_period": "2025Q1",
        "as_of": "2025-04-15",
        "source_kind": source_kind,
        "history": [
            {
                "region_code": "9999999999",
                "region_code_system": "test-h10",
                "industry_code": "coffee",
                "period": period,
                "available_at": (quarter_end(period) + timedelta(days=7)).isoformat(),
                "source_kind": source_kind,
                "source_dataset": "constructed test fixture, not an actual region",
                "sales_amount": float(1000000 + index * 10000),
                "store_count": 20,
                "opening_count": 2,
                "closing_count": 1,
                "resident_population": 10000.0,
                "working_population": 5000.0,
                "floating_population": 30000.0,
                "area_km2": 2.0,
            }
            for index, period in enumerate(periods)
        ],
    }


def install_fake_bundle(monkeypatch, directory, **metadata_updates):
    directory.mkdir()
    metadata = {
        "model_version": "fake-model-v1",
        "source_kind": "official",
        "release_status": "validated",
        "split": {
            name: {
                "latest_as_of": "2025-04-01",
                "latest_label_available_at": "2025-04-01",
            }
            for name in ("train", "validation", "test")
        },
        "supported_scopes": [
            {
                "region_code": "9999999999",
                "region_code_system": "test-h10",
                "industry_code": "coffee",
            }
        ],
    }
    metadata.update(metadata_updates)
    bundle = Mock()
    bundle.metadata = metadata
    bundle.predict.return_value = 0.05
    loader = Mock(return_value=bundle)
    monkeypatch.setattr(api, "load_bundle", loader)
    return bundle, loader


def test_missing_artifact_preserves_basic_diagnosis(tmp_path):
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "absent"))) as client:
        health = client.get("/health").json()
        assert health["basic_diagnosis_ready"] is True
        assert health["growth_model_ready"] is False
        response = client.post("/predict", json=request_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["growth"]["status"] == "unavailable"
    assert body["growth"]["estimated_yoy_growth"] is None
    assert body["growth"]["explanations"] == []
    assert body["growth"]["reason"] == "model_not_found"
    assert body["source_kind"] == "official"
    assert "confidence" not in body
    assert body["risk"]["status"] == "available"
    assert body["policy_version"].startswith("draft")


def test_authentication_is_enforced_when_configured(tmp_path):
    config = api.Settings(model_dir=tmp_path, api_key="test-only-secret")
    with TestClient(api.create_app(config)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model-info").status_code == 401
        assert client.post("/predict", json=request_payload()).status_code == 401
        assert (
            client.get("/model-info", headers={"X-API-Key": "test-only-secret"}).status_code == 200
        )


def test_production_requires_a_key():
    with pytest.raises(ValueError, match="PICKPLACE_API_KEY"):
        api.create_app(api.Settings(require_api_key=True))


def test_synthetic_data_requires_explicit_flag(tmp_path):
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "absent"))) as client:
        response = client.post("/predict", json=request_payload("synthetic"))
    assert response.status_code == 403


@pytest.mark.parametrize("change", ["future_publication", "wrong_scope", "duplicate", "nan"])
def test_invalid_history_is_a_validation_error(tmp_path, change):
    payload = request_payload()
    if change == "future_publication":
        payload["history"][-1]["available_at"] = "2025-05-01"
    elif change == "wrong_scope":
        payload["history"][0]["region_code"] = "8888888888"
    elif change == "duplicate":
        payload["history"].append(payload["history"][0])
    else:
        payload["history"][0]["sales_amount"] = "NaN"
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "absent"))) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_is_loaded_once_and_returns_growth(monkeypatch, tmp_path):
    bundle, loader = install_fake_bundle(monkeypatch, tmp_path / "model")
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "model"))) as client:
        for _ in range(2):
            response = client.post("/predict", json=request_payload())
            assert response.status_code == 200
            growth = response.json()["growth"]
            assert growth["estimated_yoy_growth"] == 0.05
            assert growth["target_period"] == "2025Q2"
            assert growth["comparison_period"] == "2024Q2"
    assert loader.call_count == 1
    assert bundle.predict.call_count == 2


@pytest.mark.parametrize(
    ("metadata", "reason"),
    [
        ({"source_kind": "synthetic"}, "source_kind_mismatch"),
        ({"supported_scopes": []}, "unsupported_scope"),
    ],
)
def test_model_provenance_and_scope_are_not_silently_substituted(
    monkeypatch, tmp_path, metadata, reason
):
    bundle, _ = install_fake_bundle(monkeypatch, tmp_path / "model", **metadata)
    config = api.Settings(model_dir=tmp_path / "model", allow_synthetic=True)
    with TestClient(api.create_app(config)) as client:
        response = client.post("/predict", json=request_payload())
    assert response.json()["growth"]["reason"] == reason
    assert response.json()["growth"]["estimated_yoy_growth"] is None
    bundle.predict.assert_not_called()


def test_experimental_models_are_blocked_by_default(monkeypatch, tmp_path):
    bundle, _ = install_fake_bundle(monkeypatch, tmp_path / "model", release_status="experimental")
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "model"))) as client:
        response = client.post("/predict", json=request_payload())
        assert client.get("/health").json()["growth_model_ready"] is False
    assert response.json()["growth"]["reason"] == "experimental_model_disabled"
    bundle.predict.assert_not_called()


def test_ended_target_is_not_presented_as_a_future_forecast(monkeypatch, tmp_path):
    bundle, _ = install_fake_bundle(monkeypatch, tmp_path / "model")
    payload = request_payload()
    payload["as_of"] = "2025-07-01"
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "model"))) as client:
        response = client.post("/predict", json=payload)
    assert response.json()["growth"]["reason"] == "target_not_future"
    bundle.predict.assert_not_called()


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_future_model_information_is_not_used_for_historical_requests(
    monkeypatch, tmp_path, partition
):
    bundle, _ = install_fake_bundle(monkeypatch, tmp_path / "model")
    bundle.metadata["split"][partition]["latest_label_available_at"] = "2025-05-01"
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "model"))) as client:
        response = client.post("/predict", json=request_payload())
    assert response.json()["growth"]["reason"] == "model_information_after_as_of"
    assert response.json()["growth"]["estimated_yoy_growth"] is None
    bundle.predict.assert_not_called()


def test_missing_model_temporal_metadata_is_not_assumed_safe(monkeypatch, tmp_path):
    bundle, _ = install_fake_bundle(monkeypatch, tmp_path / "model")
    del bundle.metadata["split"]
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "model"))) as client:
        response = client.post("/predict", json=request_payload())
    assert response.json()["growth"]["reason"] == "invalid_model_temporal_metadata"
    bundle.predict.assert_not_called()


def test_failed_model_load_is_explicit_and_does_not_crash_basic_api(monkeypatch, tmp_path):
    directory = tmp_path / "broken-model"
    directory.mkdir()
    monkeypatch.setattr(api, "load_bundle", Mock(side_effect=ValueError("checksum mismatch")))
    with TestClient(api.create_app(api.Settings(model_dir=directory))) as client:
        response = client.post("/predict", json=request_payload())
    assert response.status_code == 200
    assert response.json()["growth"]["reason"] == "model_load_failed"


def test_non_finite_model_prediction_is_not_json_null_disguised_as_success(monkeypatch, tmp_path):
    bundle, _ = install_fake_bundle(monkeypatch, tmp_path / "model")
    bundle.predict.return_value = float("nan")
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "model"))) as client:
        response = client.post("/predict", json=request_payload())
    assert response.status_code == 200
    assert response.json()["growth"]["reason"] == "prediction_out_of_bounds"
    assert response.json()["growth"]["estimated_yoy_growth"] is None


def test_growth_sensitivity_is_returned_with_an_interpretation_warning(monkeypatch, tmp_path):
    bundle, _ = install_fake_bundle(monkeypatch, tmp_path / "model")
    expected = [
        {
            "method": "single_feature_reference_sensitivity",
            "feature": "sales_yoy",
            "observed_value": 0.04,
            "reference_value": 0.01,
            "delta_growth": 0.02,
        }
    ]
    bundle.explain.return_value = expected
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "model"))) as client:
        response = client.post("/predict", json=request_payload())
    body = response.json()
    assert body["growth"]["status"] == "available"
    assert body["growth"]["explanations"] == expected
    assert "single_feature_reference_sensitivity_is_not_shap_causal_or_additive" in body["warnings"]


@pytest.mark.parametrize("failure", ["absent", "exception", "invalid_number"])
def test_explanation_failure_keeps_the_valid_growth_prediction(monkeypatch, tmp_path, failure):
    bundle, _ = install_fake_bundle(monkeypatch, tmp_path / "model")
    if failure == "absent":
        bundle.explain = None
    elif failure == "exception":
        bundle.explain.side_effect = ValueError("explanation failed")
    else:
        bundle.explain.return_value = [
            {
                "method": "single_feature_reference_sensitivity",
                "feature": "sales_yoy",
                "observed_value": float("nan"),
                "reference_value": 0.01,
                "delta_growth": 0.02,
            }
        ]
    with TestClient(api.create_app(api.Settings(model_dir=tmp_path / "model"))) as client:
        response = client.post("/predict", json=request_payload())
    body = response.json()
    assert response.status_code == 200
    assert body["growth"]["status"] == "available"
    assert body["growth"]["reason"] is None
    assert body["growth"]["estimated_yoy_growth"] == 0.05
    assert body["growth"]["explanations"] == []
    assert "growth_explanations_unavailable" in body["warnings"]
