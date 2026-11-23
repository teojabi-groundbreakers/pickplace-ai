"""합성 관측과 가상 모델을 사용한 기본 /predict 분기 전환 검증."""

import json
from dataclasses import replace
from datetime import date
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from pickplace_ai import api, near_term_api
from pickplace_ai import seoul_growth_fallback as fallback
from pickplace_ai.artifacts import file_sha256
from pickplace_ai.data import write_observations
from pickplace_ai.schemas import Observation


@pytest.fixture
def configured_fallback(monkeypatch, tmp_path):
    scope = {
        "region_code": "11199901",
        "region_code_system": "seoul-adstrd",
        "industry_code": "CS100010",
    }
    base = {
        **scope,
        "base_period": "2026Q3",
        "as_of": "2026-11-16",
        "source_kind": "official",
        "history": [],
    }
    rows = [
        Observation(
            **scope,
            period=period,
            available_at=date(2026, 11, 4),
            source_kind="official",
            source_dataset="constructed example",
            sales_amount=250_000_000,
        )
        for period in sorted(fallback.REQUIRED_PERIODS)
    ]
    source = {"kind": "sales", "period": "2026Q2", "sha256": "a", "metadata_sha256": "b"}
    registry = Mock(snapshot_at=date(2026, 11, 13))
    registry.summary = {"historical_input_sha256": "history", "new_sources": [source]}
    registry.predict.return_value = {
        "status": "available",
        "estimated_yoy_growth": 0.125,
        "model_sha256": "test-model",
        "mode": "experimental_research_only",
    }
    monkeypatch.setattr(near_term_api, "NearTermRegistry", Mock(return_value=registry))
    snapshot = tmp_path / "snapshot.csv"
    write_observations(rows, snapshot)
    snapshot.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "output_sha256": file_sha256(snapshot),
                "rows": len(rows),
                "region_code_system": "seoul-adstrd",
                "sales_basis": "as_published",
                "historical_input_sha256": "history",
                "api_sources": [source],
            }
        ),
        encoding="utf-8",
    )
    settings = api.Settings(
        model_dir=tmp_path / "primary",
        allow_experimental=True,
        seoul_fallback_model_dir=tmp_path / "fallback-model",
        seoul_fallback_snapshot=snapshot,
    )
    return registry, base, rows, settings


def _q3_row(rows, sales):
    return rows[-1].model_dump(mode="json") | {"period": "2026Q3", "sales_amount": sales}


@pytest.mark.parametrize("q3_present", [False, True])
def test_missing_q3_uses_local_q2_without_changing_indicator_period(
    configured_fallback, q3_present
):
    registry, payload, rows, settings = configured_fallback
    if q3_present:
        payload["history"] = [row.model_dump(mode="json") for row in rows[1:]] + [
            _q3_row(rows, None)
        ]
    with TestClient(api.create_app(settings)) as client:
        assert client.get("/health").json()["seoul_growth_fallback_ready"] is True
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["base_period"] == "2026Q3"
    assert result["risk_indicators"]["sales_yoy_change"]["base_period"] == "2026Q3"
    growth = result["growth"]
    assert growth["estimated_yoy_growth"] == 0.125
    assert growth["base_period"] == "2026Q2"
    assert growth["target_period"] == "2026Q4"
    assert growth["comparison_period"] == "2025Q4"
    assert growth["horizon_quarters"] == 2
    assert growth["fallback_reason"] == ("missing_sales" if q3_present else "insufficient_history")
    assert growth["release_status"] == "experimental_research_only"
    request = registry.predict.call_args.args[0]
    assert request.base_period == "2026Q2" and request.target_period == "2026Q4"
    assert request.history == rows


def test_complete_q3_uses_primary_model(configured_fallback, monkeypatch):
    registry, payload, rows, settings = configured_fallback
    payload["history"] = [row.model_dump(mode="json") for row in rows[1:]] + [
        _q3_row(rows, 275_000_000)
    ]
    settings.model_dir.mkdir()
    timing = {"latest_as_of": "2026-11-10", "latest_label_available_at": "2026-11-10"}
    scope = {key: payload[key] for key in ("region_code", "region_code_system", "industry_code")}
    bundle = Mock()
    bundle.metadata = {
        "model_version": "primary",
        "source_kind": "official",
        "release_status": "validated",
        "supported_scopes": [scope],
        "split": dict.fromkeys(("train", "validation", "test"), timing),
    }
    bundle.predict.return_value = 0.25
    bundle.explain.return_value = []
    monkeypatch.setattr(api, "load_bundle", Mock(return_value=bundle))
    with TestClient(api.create_app(settings)) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    growth = response.json()["growth"]
    assert growth["base_period"] == "2026Q3" and growth["target_period"] == "2026Q4"
    assert growth["horizon_quarters"] == 1 and growth["fallback_reason"] is None
    assert growth["estimated_yoy_growth"] == 0.25
    registry.predict.assert_not_called()


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"region_code": "11199902"}, "fallback_insufficient_history"),
        ({"as_of": "2026-10-30"}, "fallback_insufficient_history"),
        ({"as_of": "2027-01-01"}, "target_not_future"),
    ],
)
def test_fallback_unavailable_is_null(configured_fallback, changes, reason):
    registry, payload, _, settings = configured_fallback
    with TestClient(api.create_app(settings)) as client:
        response = client.post("/predict", json={**payload, **changes})
    assert response.status_code == 200
    growth = response.json()["growth"]
    assert growth["base_period"] is None and growth["target_period"] == "2026Q4"
    assert growth["status"] == "unavailable" and growth["estimated_yoy_growth"] is None
    assert growth["reason"] == reason
    registry.predict.assert_not_called()


def test_snapshot_integrity_failure_preserves_basic_diagnosis(configured_fallback):
    registry, payload, _, settings = configured_fallback
    path = settings.seoul_fallback_snapshot
    path.write_bytes(path.read_bytes() + b"\n")
    with TestClient(api.create_app(settings)) as client:
        assert client.get("/health").json()["seoul_growth_fallback_ready"] is False
        response = client.post("/predict", json=payload)
    result = response.json()
    assert result["growth"]["reason"] == "fallback_load_failed"
    assert result["risk"]["reason"] == "missing_base_period"
    registry.predict.assert_not_called()


def test_experimental_fallback_requires_explicit_opt_in(configured_fallback):
    registry, payload, _, settings = configured_fallback
    with TestClient(api.create_app(replace(settings, allow_experimental=False))) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200
    growth = response.json()["growth"]
    assert growth["estimated_yoy_growth"] is None
    assert growth["reason"] == "fallback_experimental_model_disabled"
    registry.predict.assert_not_called()
