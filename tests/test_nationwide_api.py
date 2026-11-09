"""Research API tests use fictional snapshot rows only."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pickplace_ai.nationwide_api import NationwideSettings, create_app


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    stores = tmp_path / "stores.csv"
    population = tmp_path / "population.csv"
    with stores.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "region_code_system",
                "region_code",
                "region_name",
                "sido_name",
                "sigungu_name",
                "industry_code_system",
                "industry_code",
                "industry_name",
                "store_count",
            ]
        )
        writer.writerow(
            [
                "sbiz-admin-8",
                "11110515",
                "가상동",
                "가상시",
                "가상구",
                "sbiz-commercial-2026-09",
                "I21007",
                "가상업종",
                "5",
            ]
        )
    with population.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["region_code_system", "region_code", "region_name", "resident_population"])
        writer.writerow(["mois-resident-admin-10", "1111051500", "가상동", "500"])
        writer.writerow(["mois-resident-admin-10", "1111051600", "가상동2", ""])
    manifest = {
        "schema_version": 1,
        "generated_at": "2026-11-09T10:00:00+00:00",
        "crosswalk_verified": False,
        "stores": {
            "reference_label": "202609",
            "observed_date": None,
            "available_at": None,
            "row_count": 5,
            "group_count": 1,
            "output_sha256": hashlib.sha256(stores.read_bytes()).hexdigest(),
        },
        "population": {
            "reference_month": "2026-10",
            "observed_date": "2026-10-31",
            "available_at": None,
            "row_count": 2,
            "output_sha256": hashlib.sha256(population.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def client(path: Path) -> TestClient:
    return TestClient(create_app(NationwideSettings(data_dir=path, api_key="test-key")))


def test_auth_and_separate_dated_sources(sample: Path) -> None:
    with client(sample) as app:
        assert app.get("/health").json()["snapshot_ready"] is True
        assert app.get("/research/nationwide/metadata").status_code == 401
        headers = {"X-API-Key": "test-key"}
        stores = app.get(
            "/research/nationwide/stores",
            params={"region_code": "11110515", "industry_code": "I21007"},
            headers=headers,
        ).json()
        population = app.get(
            "/research/nationwide/population",
            params={"region_code": "1111051500"},
            headers=headers,
        ).json()
        assert stores["store_count"] == 5
        assert stores["source"]["observed_date"] is None
        assert stores["source"]["reference_label"] == "202609"
        assert stores["source"]["reference_label_verified"] is False
        assert stores["source"]["national_coverage_verified"] is False
        assert population["resident_population"] == 500
        assert population["source"]["observed_date"] == "2026-10-31"
        assert stores["source"]["region_code_system"] != population["source"]["region_code_system"]
        assert "estimated_yoy_growth" not in stores
        assert "resident_population" not in stores
        assert "store_count" not in population
        assert (
            app.get("/research/nationwide/metadata", headers=headers).json()["crosswalk_verified"]
            is False
        )


def test_missing_values_are_not_zero_and_inputs_are_validated(sample: Path) -> None:
    with client(sample) as app:
        headers = {"X-API-Key": "test-key"}
        missing_scope = app.get(
            "/research/nationwide/stores",
            params={"region_code": "11110515", "industry_code": "I21008"},
            headers=headers,
        ).json()
        assert missing_scope["store_count"] is None
        assert missing_scope["reason"] == "scope_not_in_snapshot"
        missing_value = app.get(
            "/research/nationwide/population",
            params={"region_code": "1111051600"},
            headers=headers,
        ).json()
        assert missing_value["resident_population"] is None
        assert missing_value["reason"] == "population_missing"
        missing_region = app.get(
            "/research/nationwide/population",
            params={"region_code": "1111051700"},
            headers=headers,
        ).json()
        assert missing_region["reason"] == "region_not_in_snapshot"
        assert (
            app.get(
                "/research/nationwide/stores",
                params={"region_code": "1111051500", "industry_code": "I21007"},
                headers=headers,
            ).status_code
            == 422
        )


def test_tampered_artifact_fails_closed(sample: Path) -> None:
    with (sample / "stores.csv").open("a", encoding="utf-8") as file:
        file.write("tampered\n")
    with client(sample) as app:
        assert app.get("/health").json()["snapshot_ready"] is False
        assert (
            app.get("/research/nationwide/metadata", headers={"X-API-Key": "test-key"}).status_code
            == 503
        )


def test_duplicate_scope_is_rejected_even_with_matching_hash(sample: Path) -> None:
    path = sample / "stores.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    manifest_path = sample / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stores"]["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["stores"]["group_count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with client(sample) as app:
        assert app.get("/health").json()["snapshot_ready"] is False


@pytest.mark.parametrize(
    "filename,change",
    [("stores.csv", "extra"), ("population.csv", "extra"), ("population.csv", "missing")],
)
def test_malformed_csv_rows_fail_closed(sample: Path, filename: str, change: str) -> None:
    path = sample / filename
    lines = path.read_text(encoding="utf-8").splitlines()
    if change == "extra":
        lines[1] += ",unexpected"
    else:
        # A missing field is different from the explicit empty population cell.
        assert lines[2].endswith(",")
        lines[2] = lines[2][:-1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_path = sample / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[filename.split(".")[0]]["output_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with client(sample) as app:
        assert app.get("/health").json()["snapshot_ready"] is False
        assert (
            app.get("/research/nationwide/metadata", headers={"X-API-Key": "test-key"}).status_code
            == 503
        )


def test_key_is_mandatory(sample: Path) -> None:
    with (
        pytest.raises(ValueError, match="PICKPLACE_NATIONWIDE_API_KEY"),
        TestClient(create_app(NationwideSettings(data_dir=sample, api_key=""))),
    ):
        pass
