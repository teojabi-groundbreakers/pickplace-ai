import csv
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from pickplace_ai import data, features, forecast_2026, ingest, periods, training
from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.near_term_api import NearTermRegistry, NearTermSettings, create_app
from scripts import near_term_forecast, retrospective_all_industries


@pytest.fixture
def artifacts(tmp_path):
    model = DummyRegressor(strategy="constant", constant=0.2)
    model.fit(pd.DataFrame([[0]], columns=["x"]), [0])
    joblib.dump(
        {
            "artifact_kind": near_term_forecast.KIND,
            "models": {
                h: {"model_name": "dummy", "model": model, "features": near_term_forecast.FEATURES}
                for h in ("1", "2")
            },
        },
        tmp_path / "model.joblib",
    )
    fields = [
        "region_code_system",
        "region_code",
        "industry_code",
        "base_period",
        "target_period",
        "scope_seen_in_training",
        "industry_test_rows",
        "warning",
    ]
    with (tmp_path / "forecast-2026-q3-q4.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target in ("2026Q3", "2026Q4"):
            writer.writerow(
                {
                    "region_code_system": "seoul-adstrd",
                    "region_code": "11110515",
                    "industry_code": "CS100010",
                    "base_period": "2026Q2",
                    "target_period": target,
                    "scope_seen_in_training": "True",
                    "industry_test_rows": "50",
                    "warning": "",
                }
            )
    dependencies = {
        "forecast_2026.py": forecast_2026.__file__,
        "training.py": training.__file__,
        "ingest.py": ingest.__file__,
        "data.py": data.__file__,
        "periods.py": periods.__file__,
        "retrospective_all_industries.py": retrospective_all_industries.__file__,
    }
    summary = {
        "artifact_kind": near_term_forecast.KIND,
        "release_status": "experimental_research_only",
        "as_of": "2026-11-04",
        "created_at": "2026-11-04T12:00:00Z",
        "origin": "2026Q2",
        "model_sha256": file_sha256(tmp_path / "model.joblib"),
        "forecast_sha256": file_sha256(tmp_path / "forecast-2026-q3-q4.csv"),
        "feature_code_sha256": file_sha256(Path(features.__file__)),
        "runner_sha256": file_sha256(Path(near_term_forecast.__file__)),
        "dependency_code_sha256": {k: file_sha256(Path(v)) for k, v in dependencies.items()},
        "versions": package_versions(),
        "results": {
            h: {
                "target_period": target,
                "selection": "dummy",
                "common_2025_test": {
                    "new": {
                        "rows": 42,
                        "mae_pp": 12.0,
                        "median_absolute_error_pp": 7.0,
                    }
                },
            }
            for h, target in (("1", "2026Q3"), ("2", "2026Q4"))
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return tmp_path


@pytest.fixture
def payload():
    rows = []
    for period in ("2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"):
        rows.append(
            {
                "region_code": "11110515",
                "region_code_system": "seoul-adstrd",
                "industry_code": "CS100010",
                "period": period,
                "available_at": "2026-11-03" if period.startswith("2025") else "2026-11-04",
                "source_kind": "official",
                "source_dataset": "fixture",
                "sales_amount": 1_000_000,
                "store_count": 10,
            }
        )
    return {
        "region_code": "11110515",
        "region_code_system": "seoul-adstrd",
        "industry_code": "CS100010",
        "base_period": "2026Q2",
        "target_period": "2026Q3",
        "as_of": "2026-11-04",
        "source_kind": "official",
        "history": rows,
    }


@pytest.fixture
def client(artifacts):
    with TestClient(create_app(NearTermSettings(artifacts, "test-secret"))) as test_client:
        yield test_client


def test_authenticated_prediction_contains_research_limits(client, payload):
    endpoint = "/research/near-term/predict"
    assert client.get("/health").json()["model_ready"] is True
    assert client.post(endpoint, json=payload).status_code == 401
    response = client.post(endpoint, json=payload, headers={"X-API-Key": "test-secret"})
    assert response.status_code == 200
    body = response.json()
    assert body["estimated_yoy_growth"] == pytest.approx(0.2)
    assert body["estimate_kind"] == "completed_quarter_unobserved_estimate"
    assert body["input_values_verified"] is False
    assert body["region_scope_verified"] is False
    assert body["scope_seen_in_training"] is True
    assert body["retrospective_test"]["mae_pp"] == 12
    assert body["mode"] == "experimental_research_only"
    payload["target_period"] = "2026Q4"
    response = client.post(endpoint, json=payload, headers={"X-API-Key": "test-secret"})
    assert response.json()["estimate_kind"] == "in_progress_or_future_quarter_estimate"


def test_unavailable_values_do_not_become_zero(client, payload):
    endpoint = "/research/near-term/predict"
    headers = {"X-API-Key": "test-secret"}
    payload["history"] = [r for r in payload["history"] if r["period"] != "2025Q2"]
    result = client.post(endpoint, json=payload, headers=headers).json()
    assert result["estimated_yoy_growth"] is None
    assert result["reason"] == "insufficient_history"
    assert result["comparison_period"] == "2025Q3"
    assert result["estimate_kind"] == "completed_quarter_unobserved_estimate"
    payload["target_period"] = "2027Q1"
    result = client.post(endpoint, json=payload, headers=headers).json()
    assert result["reason"] == "unsupported_forecast_period"
    assert result["estimated_yoy_growth"] is None


def test_comparison_zero_and_wrong_scope_are_unavailable(client, payload):
    endpoint = "/research/near-term/predict"
    headers = {"X-API-Key": "test-secret"}
    payload["history"][1]["sales_amount"] = 0
    result = client.post(endpoint, json=payload, headers=headers).json()
    assert result["reason"] == "missing_or_nonpositive_comparison_sales"
    assert result["estimated_yoy_growth"] is None
    payload["region_code"] = "11110516"
    for row in payload["history"]:
        row["region_code"] = payload["region_code"]
    result = client.post(endpoint, json=payload, headers=headers).json()
    assert result["reason"] == "unsupported_or_insufficient_scope"


def test_wrong_h10_code_cannot_be_silently_converted(client, payload):
    payload["region_code"] = "1111051500"
    for row in payload["history"]:
        row["region_code"] = payload["region_code"]
    result = client.post(
        "/research/near-term/predict", json=payload, headers={"X-API-Key": "test-secret"}
    ).json()
    assert result["reason"] == "unsupported_region_or_industry_code"
    assert result["estimated_yoy_growth"] is None


def test_hash_or_version_mismatch_fails_closed(artifacts):
    (artifacts / "model.joblib").write_bytes(b"tampered")
    with TestClient(create_app(NearTermSettings(artifacts, "test-secret"))) as client:
        assert client.get("/health").json()["model_ready"] is False
        assert (
            client.get(
                "/research/near-term/model-info", headers={"X-API-Key": "test-secret"}
            ).status_code
            == 503
        )


def test_snapshot_before_acquisition_is_unavailable(client, payload):
    payload["as_of"] = "2026-11-03"
    for row in payload["history"]:
        row["available_at"] = "2026-11-03"
    response = client.post(
        "/research/near-term/predict", json=payload, headers={"X-API-Key": "test-secret"}
    )
    assert response.json()["reason"] == "as_of_before_model_snapshot"


def test_future_as_of_is_unavailable(client, payload):
    payload["as_of"] = "2099-01-01"
    response = client.post(
        "/research/near-term/predict", json=payload, headers={"X-API-Key": "test-secret"}
    )
    assert response.json()["reason"] == "as_of_in_future"
    assert response.json()["estimated_yoy_growth"] is None


def test_runtime_version_mismatch_is_rejected(artifacts):
    summary_path = artifacts / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["versions"]["scikit-learn"] = "0.0.0"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with TestClient(create_app(NearTermSettings(artifacts, "test-secret"))) as client:
        assert client.get("/health").json()["model_ready"] is False


def test_loaded_pickle_bytes_are_the_verified_bytes(artifacts, monkeypatch):
    import pickplace_ai.near_term_api as api

    actual_load = api.joblib.load

    def swap_after_verification(stream):
        (artifacts / "model.joblib").write_bytes(b"changed after checksum")
        return actual_load(stream)

    monkeypatch.setattr(api.joblib, "load", swap_after_verification)
    registry = NearTermRegistry(artifacts)
    assert registry.models["1"]["model_name"] == "dummy"
