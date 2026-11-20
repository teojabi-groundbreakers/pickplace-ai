"""참조표 생성 테스트는 실제 지역의 점포·인구를 사용하지 않는다."""

import csv
import json
import sqlite3
from contextlib import closing

import pytest

from pickplace_ai.artifacts import file_sha256
from pickplace_ai.fit_reference import FitReference
from pickplace_ai.seoul_stats_ingest import MODE
from scripts.build_seoul_fit_reference import build


@pytest.fixture
def inputs(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    database = snapshot / "snapshot.sqlite3"
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "CREATE TABLE stores (period TEXT, region_code TEXT, "
            "industry_code TEXT, store_count INT)"
        )
        connection.executemany(
            "INSERT INTO stores VALUES (?, ?, ?, ?)",
            [
                ("2025Q2", "11110515", "CS100010", 10),
                ("2025Q2", "11110530", "CS100010", 4),
                ("2025Q2", "11110540", "CS100010", 0),
                ("2025Q1", "11110515", "CS100010", 100),
            ],
        )
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "mode": MODE,
                "database_sha256": file_sha256(database),
                "created_at_utc": "2026-11-19T16:00:00+00:00",
                "datasets": {"stores": {"periods": ["2025Q2"]}},
            }
        )
    )
    entries = {
        "area": ["reference_date,sgis_code,area_m2", "2025-06-30,11010720,2500000"],
        "population": [
            "행정구역,2025년06월_총인구수",
            "서울특별시 (1100000000),250",
            "서울특별시 종로구 (1111000000),250",
            "서울특별시 종로구 청운효자동 (1111051500),250",
        ],
        "mapping": [
            "reference_date,sgis_code,h10_code,seoul_code",
            "2025-06-30,11010720,1111051500,11110515",
        ],
    }
    paths = {}
    metadata = {"reference_date": "2025-06-30", "sources": {}}
    for name, lines in entries.items():
        path = tmp_path / f"{name}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[name] = path
        metadata["sources"][name] = {
            "url": f"https://example.test/{name}",
            "sha256": file_sha256(path),
            "reference_date": "2025-06-30",
            "acquired_at": "2026-11-19",
        }
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(metadata))
    return (
        snapshot,
        paths["area"],
        paths["population"],
        paths["mapping"],
        evidence,
        tmp_path / "out",
    )


def test_build_joins_exact_scopes_and_records_every_exclusion(inputs):
    result = build(*inputs)
    assert result["cohort_sizes"] == {"CS100010": 1}
    assert result["available_at"] == "2026-11-20"
    assert result["candidate_rows"] == 3
    assert result["exclusions"] == {
        "nonpositive_or_missing_store_count": 1,
        "region_mapping_missing": 1,
    }
    output = inputs[-1]
    reference = FitReference(output)
    assert reference.manifest["row_count"] == 1
    with (output / "reference.csv").open(newline="") as handle:
        assert [*csv.reader(handle)][1] == ["11110515", "CS100010", "10", "250", "2500000.0"]
    with pytest.raises(FileExistsError):
        build(*inputs)


def test_build_rejects_mismatched_date_and_tampered_input(inputs):
    _, area, _, _, evidence, output = inputs
    area.write_text("reference_date,sgis_code,area_m2\n2026-06-30,11010720,2500000\n")
    with pytest.raises(ValueError, match="recorded SHA-256"):
        build(*inputs)
    metadata = json.loads(evidence.read_text())
    metadata["sources"]["area"]["sha256"] = file_sha256(area)
    evidence.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="reference date"):
        build(*inputs)
    assert not output.exists()


def test_build_rejects_ambiguous_code_mapping(inputs):
    _, _, _, mapping, evidence, output = inputs
    with mapping.open("a") as handle:
        handle.write("2025-06-30,11010720,1111053000,11110530\n")
    metadata = json.loads(evidence.read_text())
    metadata["sources"]["mapping"]["sha256"] = file_sha256(mapping)
    evidence.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="one-to-one"):
        build(*inputs)
    assert not output.exists()
