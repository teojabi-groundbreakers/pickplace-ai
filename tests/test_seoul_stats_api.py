"""Research endpoint tests use fictional source data only."""

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pickplace_ai.seoul_stats_api import MODE, SeoulStatsSettings, create_app


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    dbpath = tmp_path / "snapshot.sqlite3"
    with closing(sqlite3.connect(dbpath)) as db:
        db.executescript("""
            CREATE TABLE sales (period TEXT NOT NULL, region_code TEXT NOT NULL,
                industry_code TEXT NOT NULL, sales_amount INTEGER,
                PRIMARY KEY(period, region_code, industry_code)) WITHOUT ROWID;
            CREATE TABLE stores (period TEXT NOT NULL, region_code TEXT NOT NULL,
                industry_code TEXT NOT NULL, store_count INTEGER, opening_count INTEGER,
                closing_count INTEGER,
                PRIMARY KEY(period, region_code, industry_code)) WITHOUT ROWID;
            CREATE TABLE resident (period TEXT NOT NULL, region_code TEXT NOT NULL,
                total INTEGER, PRIMARY KEY(period, region_code)) WITHOUT ROWID;
            CREATE TABLE floating (period TEXT NOT NULL, region_code TEXT NOT NULL,
                total INTEGER, PRIMARY KEY(period, region_code)) WITHOUT ROWID;
            INSERT INTO sales VALUES ('2025Q1', '11110515', 'CS100001', 0);
            INSERT INTO stores VALUES ('2025Q1', '11110515', 'CS100001', 4, 1, NULL);
            INSERT INTO stores VALUES ('2025Q1', '11110515', 'CS100002', 3, 0, 1);
            INSERT INTO resident VALUES ('2025Q1', '11110515', 500);
            INSERT INTO floating VALUES ('2025Q2', '11110515', NULL);
            """)
        db.commit()
    sources = {}
    for table, rows, periods in (
        ("sales", 1, ["2025Q1"]),
        ("stores", 2, ["2025Q1"]),
        ("resident", 1, ["2025Q1"]),
        ("floating", 1, ["2025Q2"]),
    ):
        sources[table] = {
            "row_count": rows,
            "periods": periods,
            "source_files": [{"name": "fictional.csv", "sha256": "0" * 64}],
            "publication_date_verified": False,
        }
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at_utc": "2026-11-09T10:00:00+00:00",
                "mode": MODE,
                "database_sha256": hashlib.sha256(dbpath.read_bytes()).hexdigest(),
                "datasets": sources,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def client(path: Path) -> TestClient:
    return TestClient(create_app(SeoulStatsSettings(data_dir=path, api_key="secret")))


QUERY = {
    "region_code": "11110515",
    "industry_code": "CS100001",
    "start_period": "2025Q1",
    "end_period": "2025Q2",
}
HEADERS = {"X-API-Key": "secret"}


def test_quarterly_snapshot_auth_and_independent_sources(sample: Path) -> None:
    with client(sample) as app:
        assert app.get("/health").json()["snapshot_ready"] is True
        assert app.get("/research/seoul/indicators", params=QUERY).status_code == 401
        response = app.get("/research/seoul/indicators", params=QUERY, headers=HEADERS)
        assert response.status_code == 200
        result = response.json()
        assert result["period_granularity"] == "quarter"
        assert result["snapshot_created_at_utc"] == "2026-11-09T10:00:00+00:00"
        assert result["publication_date_verified"] is False
        assert result["source_columns"]["sales.sales_amount"] == "당월_매출_금액"
        first, second = result["observations"]
        assert first["period"] == "2025Q1"
        assert first["sales"]["values"]["sales_amount"]["value"] == 0
        assert first["stores"]["status"] == "partial"
        assert first["stores"]["values"]["closing_count"]["reason"] == "source_value_missing"
        assert first["resident"]["values"]["total"]["value"] == 500
        assert first["floating"]["status"] == "unavailable"
        assert first["floating"]["reason"] == "scope_not_in_snapshot"
        assert second["floating"]["status"] == "unavailable"
        assert second["floating"]["values"]["total"]["reason"] == "source_value_missing"
        assert result["unavailable_metrics"]["monthly_series"]["value"] is None
        assert "latest_snapshot_not_proof_of_historical_publication" in result["warnings"]
        assert app.get("/research/seoul/metadata", headers=HEADERS).status_code == 200


def test_stores_only_industry_survives_and_population_is_not_industry_specific(
    sample: Path,
) -> None:
    with client(sample) as app:
        response = app.get(
            "/research/seoul/indicators",
            headers=HEADERS,
            params={**QUERY, "industry_code": "CS100002"},
        ).json()
        row = response["observations"][0]
        assert row["sales"]["values"]["sales_amount"]["value"] is None
        assert row["stores"]["values"]["store_count"]["value"] == 3
        assert row["resident"]["values"]["total"]["value"] == 500


@pytest.mark.parametrize(
    "bad",
    [
        {"region_code": "1111051500"},
        {"region_code": "21110515"},
        {"industry_code": "coffee"},
        {"start_period": "2025-01"},
        {"start_period": "2025Q3", "end_period": "2025Q1"},
        {"start_period": "2010Q1", "end_period": "2025Q1"},
    ],
)
def test_invalid_scope_and_period_fail(sample: Path, bad: dict) -> None:
    with client(sample) as app:
        assert (
            app.get(
                "/research/seoul/indicators", params={**QUERY, **bad}, headers=HEADERS
            ).status_code
            == 422
        )


def test_modified_or_symlinked_snapshot_fails_closed(sample: Path, tmp_path: Path) -> None:
    database = sample / "snapshot.sqlite3"
    with database.open("ab") as stream:
        stream.write(b"tampered")
    with client(sample) as app:
        assert app.get("/health").json()["snapshot_ready"] is False
        assert (
            app.get("/research/seoul/indicators", params=QUERY, headers=HEADERS).status_code == 503
        )
    database.unlink()
    database.symlink_to(tmp_path / "elsewhere")
    with client(sample) as app:
        assert app.get("/health").json()["snapshot_ready"] is False


def test_wrong_count_fails_even_with_valid_database_digest(sample: Path) -> None:
    path = sample / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["datasets"]["sales"]["row_count"] = 2
    path.write_text(json.dumps(manifest))
    with client(sample) as app:
        assert app.get("/health").json()["snapshot_ready"] is False


def test_key_mandatory(sample: Path) -> None:
    with (
        pytest.raises(ValueError, match="PICKPLACE_SEOUL_STATS_API_KEY"),
        TestClient(create_app(SeoulStatsSettings(data_dir=sample, api_key=""))),
    ):
        pass


def test_snapshot_mutated_after_start_fails_closed(sample: Path) -> None:
    with client(sample) as app:
        path = sample / "snapshot.sqlite3"
        with path.open("ab") as stream:
            stream.write(b"mutation")
        response = app.get("/research/seoul/indicators", params=QUERY, headers=HEADERS)
        assert response.status_code == 503
