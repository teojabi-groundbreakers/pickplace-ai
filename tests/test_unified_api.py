"""Use only fictional scopes and inputs to test the research API."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from pickplace_ai import features, unified_features
from pickplace_ai.artifacts import file_sha256, package_versions
from pickplace_ai.unified_api import Registry, create_app
from scripts import train_gyeonggi_research, train_unified_growth


@pytest.fixture
def root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(
        "pickplace_ai.seoul_guard_policy.policy_compatible", lambda meta, digest: True
    )
    cols = list(unified_features.UNIFIED_RESEARCH_FEATURES)
    model = DummyRegressor(strategy="constant", constant=0.2)
    model.fit(pd.DataFrame([[1.0] * len(cols)], columns=cols), np.array([0.0]))
    bundle = {"artifact_kind": train_unified_growth.KIND, "model": model, "feature_columns": cols}
    model_path = tmp_path / "final-model.joblib"
    joblib.dump(bundle, model_path)
    meta = {
        "artifact_kind": train_unified_growth.KIND,
        "purpose": "retrospective_research_only_not_service",
        "created_at_utc": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "source_code_systems": {
            "seoul": "seoul-adstrd",
            "gyeonggi": "gyeonggi-admdong-10-unverified",
        },
        "source_known_by": {"seoul": "2026-11-04", "gyeonggi": "2026-11-09"},
        "target": "quarter_sales[t+1]/quarter_sales[t-3]-1",
        "final_training_target_periods": ["2025Q2", "2026Q2"],
        "source_scopes": {
            "seoul": [["11110515", "CS100010"]],
            "gyeonggi": [["4111111111", "Q13"]],
        },
        "versions": package_versions(),
        "code_sha256": {
            "trainer": file_sha256(Path(train_unified_growth.__file__)),
            "feature_code": file_sha256(Path(features.__file__)),
            "unified_feature_code": file_sha256(Path(unified_features.__file__)),
            "base_trainer": file_sha256(Path(train_gyeonggi_research.__file__)),
        },
        "model_sha256": {"final-model.joblib": hashlib.sha256(model_path.read_bytes()).hexdigest()},
    }
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return tmp_path


def payload(source="seoul"):
    return {
        "source": source,
        "region_code_system": "seoul-adstrd"
        if source == "seoul"
        else "gyeonggi-admdong-10-unverified",
        "region_code": "11110515" if source == "seoul" else "4111111111",
        "industry_code": "CS100010" if source == "seoul" else "Q13",
        "sales_unit": "KRW" if source == "seoul" else "source_unit_unverified",
        "base_period": "2026Q2",
        "target_period": "2026Q3",
        "as_of": "2026-11-09",
        "data_available_by": "2026-11-09",
        "history": [
            {"period": "2025Q3", "sales_amount": 20_000_000},
            {"period": "2025Q4", "sales_amount": 22_000_000},
            {"period": "2026Q1", "sales_amount": 23_000_000},
            {"period": "2026Q2", "sales_amount": 24_000_000},
        ],
    }


def test_both_sources_and_null_scores(root):
    with TestClient(create_app(root, "test-key")) as client:
        assert client.get("/health").json()["model_ready"] is True
        assert client.post("/research/unified/predict", json=payload()).status_code == 401
        for source in ("seoul", "gyeonggi"):
            response = client.post(
                "/research/unified/predict", json=payload(source), headers={"X-API-Key": "test-key"}
            )
            assert response.status_code == 200
            body = response.json()
            assert body["status"] == "available" and body["estimated_yoy_growth"] == pytest.approx(
                0.2
            )
            assert body["input_values_verified"] is False
            assert body["region_scope_verified"] is False
            assert body["sales_unit_verified"] is False
            assert body["estimate_kind"] == "completed_quarter_unobserved_estimate"
            assert all(v is None for v in body["scores"].values())
            assert all(body["score_unavailable_reasons"].values())
            for key in (
                "factors",
                "statistics",
                "floating_population",
                "competition",
                "map",
                "summary",
                "recommendations",
                "prediction_interval",
                "retrospective_test",
            ):
                assert body[key] is None
                assert body["unavailable_reasons"][key]


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ({"target_period": "2026Q4"}, "unsupported_forecast_period"),
        ({"base_period": "2026Q1", "target_period": "2026Q2"}, "unsupported_forecast_period"),
        ({"as_of": "2026-11-08"}, "as_of_before_model_snapshot"),
        ({"data_available_by": None}, "input_known_by_missing"),
        ({"data_available_by": "2026-11-10"}, "input_known_after_as_of"),
        ({"data_available_by": "2026-06-29"}, "input_known_before_base_period_end"),
        ({"industry_code": "CS100011"}, "unsupported_or_insufficient_scope"),
        ({"region_code_system": "MOIS_H10"}, "unsupported_source_or_region_code_system"),
        ({"sales_unit": "source_unit_unverified"}, "unsupported_or_unverified_sales_unit"),
        ({"history": []}, "insufficient_history"),
        ({"history": [{"period": "2026Q3", "sales_amount": 10}]}, "history_after_base_period"),
    ],
)
def test_unavailable_has_null_and_reason(root, mutation, reason):
    request = payload()
    request.update(mutation)
    with TestClient(create_app(root, "test-key")) as client:
        body = client.post(
            "/research/unified/predict", json=request, headers={"X-API-Key": "test-key"}
        ).json()
        assert body["status"] == "unavailable" and body["estimated_yoy_growth"] is None
        assert body["reason"] == reason
        assert body["risk_details"] == []


def test_missing_duplicate_and_zero_are_not_coerced(root):
    request = payload()
    with TestClient(create_app(root, "test-key")) as client:
        request["history"][0]["sales_amount"] = None
        response = client.post(
            "/research/unified/predict", json=request, headers={"X-API-Key": "test-key"}
        )
        assert response.json()["reason"] == "missing_sales"
        request["history"][0]["sales_amount"] = 0
        response = client.post(
            "/research/unified/predict", json=request, headers={"X-API-Key": "test-key"}
        )
        assert response.json()["reason"] == "invalid_sales_denominator"
        request["history"][0]["sales_amount"] = 10
        request["history"].append(request["history"][0])
        response = client.post(
            "/research/unified/predict", json=request, headers={"X-API-Key": "test-key"}
        )
        assert response.json()["reason"] == "duplicate_history_period"


def test_invalid_artifact_has_null_and_reason(root):
    (root / "final-model.joblib").write_bytes(b"corrupt")
    with TestClient(create_app(root, "test-key")) as client:
        assert client.get("/health").json()["model_ready"] is False
        response = client.post(
            "/research/unified/predict", json=payload(), headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 503
        assert response.json()["reason"] == "unified_model_unavailable"
        assert response.json()["estimated_yoy_growth"] is None
        assert response.json()["risk_details"] == []


def test_rejects_mismatched_code_hash(root):
    file = root / "metadata.json"
    metadata = json.loads(file.read_text())
    metadata["code_sha256"]["base_trainer"] = "0" * 64
    file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="does not match"):
        Registry(root)


def test_invalid_quarter_year_is_validation_error(root):
    request = payload()
    request["target_period"] = "0001Q1"
    with TestClient(create_app(root, "test-key")) as client:
        response = client.post(
            "/research/unified/predict", json=request, headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 422


def test_extreme_finite_sales_returns_unavailable_not_server_error(root):
    request = payload()
    request["history"][0]["sales_amount"] = 1e-320
    request["history"][-1]["sales_amount"] = 1e300
    with TestClient(create_app(root, "test-key")) as client:
        response = client.post(
            "/research/unified/predict", json=request, headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200
        assert response.json()["reason"] == "invalid_or_nonfinite_features"
        assert response.json()["estimated_yoy_growth"] is None


def test_seoul_guard_withholds_each_signal_and_does_not_call_model(root, monkeypatch):
    def should_not_predict(*args, **kwargs):
        raise AssertionError("model must not be called when Seoul input is flagged")

    monkeypatch.setattr(train_unified_growth, "predict_unified_model", should_not_predict)
    with TestClient(create_app(root, "test-key")) as client:
        cases = [
            ([1_000_000] * 4, ["low_reference_sales", "low_latest_sales"]),
            ([10_000_000, 11_000_000, 11_000_000, 11_000_000], ["low_reference_sales"]),
            ([20_000_000, 15_000_000, 10_000_000, 8_000_000], ["low_latest_sales"]),
            ([100_000_000, 20_000_000, 20_000_000, 20_000_000], ["rapid_sales_decrease"]),
            ([20_000_000, 80_000_000, 80_000_000, 80_000_000], ["rapid_sales_increase"]),
            (
                [20_000_000, 2_000_000, 2_000_000, 2_000_000],
                ["low_latest_sales", "rapid_sales_decrease"],
            ),
            (
                [1_000_000, 20_000_000, 2_000_000, 2_000_000],
                [
                    "low_reference_sales",
                    "low_latest_sales",
                    "rapid_sales_increase",
                    "rapid_sales_decrease",
                ],
            ),
        ]
        for values, signals in cases:
            request = payload()
            for row, value in zip(request["history"], values, strict=True):
                row["sales_amount"] = value
            response = client.post(
                "/research/unified/predict", json=request, headers={"X-API-Key": "test-key"}
            )
            body = response.json()
            assert response.status_code == 200
            assert body["status"] == "unavailable" and body["estimated_yoy_growth"] is None
            assert body["reason"] == "seoul_sales_history_unstable"
            assert body["risk_signals"] == signals
            assert [d["signal"] for d in body["risk_details"]] == signals
            assert all(
                d["to_period"] in {row["period"] for row in request["history"]}
                for d in body["risk_details"]
            )
            assert body["reason_message"] and len(body["risk_signal_messages"]) == len(signals)
            assert body["input_values_verified"] is False


def test_policy_mismatch_withholds_seoul_but_does_not_change_gyeonggi(root, monkeypatch):
    monkeypatch.setattr(
        "pickplace_ai.seoul_guard_policy.policy_compatible", lambda meta, digest: False
    )
    with TestClient(create_app(root, "test-key")) as client:
        seoul = client.post(
            "/research/unified/predict", json=payload(), headers={"X-API-Key": "test-key"}
        ).json()
        assert seoul["reason"] == "seoul_guard_policy_mismatch"
        assert seoul["risk_details"] == []
        assert seoul["estimated_yoy_growth"] is None and seoul["reason_message"]
        gyeonggi = client.post(
            "/research/unified/predict",
            json=payload("gyeonggi"),
            headers={"X-API-Key": "test-key"},
        ).json()
        assert gyeonggi["status"] == "available"
        assert gyeonggi["risk_policy_status"] == "not_applicable"
        assert gyeonggi["risk_details"] == []


def test_guard_order_and_zero_latest_and_gyeonggi_explicit_nonapplication(root):
    with TestClient(create_app(root, "test-key")) as client:
        request = payload()
        request["history"].reverse()
        good = client.post(
            "/research/unified/predict", json=request, headers={"X-API-Key": "test-key"}
        ).json()
        assert good["status"] == "available" and good["risk_signals"] == []
        assert good["risk_details"] == []
        request["history"][0]["sales_amount"] = 0
        zero = client.post(
            "/research/unified/predict", json=request, headers={"X-API-Key": "test-key"}
        ).json()
        assert zero["estimated_yoy_growth"] is None
        assert "low_latest_sales" in zero["risk_signals"]
        assert "rapid_sales_decrease" in zero["risk_signals"]
        assert zero["risk_policy_prospectively_validated"] is False
        gyeonggi_request = payload("gyeonggi")
        for row in gyeonggi_request["history"]:
            row["sales_amount"] = 10
        gyeonggi = client.post(
            "/research/unified/predict", json=gyeonggi_request, headers={"X-API-Key": "test-key"}
        ).json()
        assert gyeonggi["status"] == "available"
        assert gyeonggi["risk_policy_status"] == "not_applicable"
        assert gyeonggi["risk_policy_reason"] == "gyeonggi_risk_policy_unverified"
        assert gyeonggi["risk_policy_reason_message"]
        health = client.get("/health").json()
        assert health["model_ready"] is True and health["seoul_risk_policy_ready"] is True
