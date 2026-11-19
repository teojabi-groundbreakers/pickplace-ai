"""Prevent future or unverified zoning from entering a base-quarter feature frame."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from pickplace_ai import seoul_zoning_time as gate


def zoning(
    version_id="20250805",
    *,
    first_available_at="2025-08-06",
    geometry_valid_at="2025-08-05",
    **changes,
):
    value = {
        "version_id": version_id,
        "archive_sha256": "a" * 64,
        "first_available_at": first_available_at,
        "geometry_valid_at": geometry_valid_at,
        "historical_validity_verified": True,
        "training_eligible": True,
        "availability_evidence": "https://data.seoul.go.kr/example-availability",
        "geometry_validity_evidence": "docs/verified-zoning-geometry.md",
    }
    value.update(changes)
    return value


def boundary(
    version_id="boundary-2025",
    *,
    first_available_at="2025-01-02",
    geometry_valid_from="2025-01-01",
    geometry_valid_through="2025-12-31",
    **changes,
):
    value = {
        "version_id": version_id,
        "archive_sha256": "b" * 64,
        "first_available_at": first_available_at,
        "geometry_valid_from": geometry_valid_from,
        "geometry_valid_through": geometry_valid_through,
        "historical_validity_verified": True,
        "training_eligible": True,
        "availability_evidence": "https://data.seoul.go.kr/example-boundary-availability",
        "geometry_validity_evidence": "docs/verified-boundary-interval.md",
    }
    value.update(changes)
    return value


def test_base_quarter_cutoff_crosses_year_boundary():
    assert gate.base_quarter_cutoff("2026Q1") == ("2025Q4", gate.date(2025, 12, 31))
    assert gate.base_quarter_cutoff("2025Q4") == ("2025Q3", gate.date(2025, 9, 30))


@pytest.mark.parametrize("bad", ["2025Q0", "2025Q5", "2025-04", "2025Q4junk", None])
def test_invalid_target_period_rejected(bad):
    with pytest.raises(ValueError, match="target_period"):
        gate.select_temporal_sources(bad, [], [])


def test_latest_verified_source_known_by_base_cutoff_selected_without_mutation():
    versions = [
        zoning("20250320", first_available_at="2025-03-21", geometry_valid_at="2025-03-20"),
        zoning(),
        zoning("20260304", first_available_at="2026-03-05", geometry_valid_at="2026-03-04"),
    ]
    boundaries = [
        boundary("boundary-old", geometry_valid_from="2024-01-01"),
        boundary("boundary-new", geometry_valid_from="2025-01-01"),
    ]
    original = deepcopy((versions, boundaries))
    result = gate.select_temporal_sources("2025Q4", versions, boundaries)
    assert result["temporal_status"] == "passed"
    assert result["reason"] is None
    assert (result["base_period"], result["base_cutoff"]) == ("2025Q3", "2025-09-30")
    assert result["zoning"]["version_id"] == "20250805"
    assert result["boundary"]["version_id"] == "boundary-new"
    assert result["rejected_versions"]["zoning"] == [
        {"version_id": "20260304", "reason": "not_available_by_cutoff"}
    ]
    assert (versions, boundaries) == original


def test_source_not_available_by_base_cutoff_is_blocked():
    result = gate.select_temporal_sources(
        "2025Q4",
        [zoning(first_available_at="2025-10-01", geometry_valid_at="2025-08-05")],
        [boundary()],
    )
    assert result["temporal_status"] == "blocked"
    assert result["reason"] == "zoning_not_available_by_cutoff"
    assert result["zoning"] is None


def test_future_geometry_is_blocked_even_when_availability_missing():
    result = gate.select_temporal_sources(
        "2025Q4",
        [zoning(first_available_at=None, geometry_valid_at="2025-10-01")],
        [boundary()],
    )
    assert result["reason"] == "zoning_geometry_after_cutoff"


def test_boundary_must_have_verified_interval_covering_cutoff():
    for candidate, expected in [
        (
            boundary(geometry_valid_through=None),
            "boundary_validity_interval_unverified",
        ),
        (
            boundary(geometry_valid_through="2025-09-29"),
            "boundary_geometry_not_valid_at_cutoff",
        ),
        (boundary(geometry_valid_from="2025-10-01"), "boundary_geometry_after_cutoff"),
    ]:
        result = gate.select_temporal_sources("2025Q4", [zoning()], [candidate])
        assert result["temporal_status"] == "blocked"
        assert result["reason"] == expected
        assert result["zoning"] is None
        assert result["boundary"] is None


def test_current_raw_sidecars_and_filename_dates_cannot_pass_as_evidence():
    raw_zoning = zoning(
        first_available_at=None,
        geometry_valid_at=None,
        historical_validity_verified=False,
        training_eligible=False,
        source_listing_modified_date="2025-08-05",
        CREATE_DAT="20250805",
    )
    raw_boundary = boundary(
        first_available_at=None,
        geometry_valid_from=None,
        geometry_valid_through=None,
        historical_validity_verified=False,
        training_eligible=False,
        source_listing_modified_date="2023-10-31",
    )
    result = gate.select_temporal_sources("2025Q4", [raw_zoning], [raw_boundary])
    assert result["temporal_status"] == "blocked"
    assert result["reason"] == "zoning_provenance_unverified"
    assert result["zoning"] is None and result["boundary"] is None
    assert result["rejected_versions"]["boundary"][0]["reason"] == "provenance_unverified"


def test_actual_ignored_raw_sidecars_are_blocked():
    root = Path(__file__).resolve().parents[1]
    zoning_path = (
        root / "data/raw/seoul/zoning/urban_area/source_versions/20250805/source.metadata.json"
    )
    boundary_path = (
        root / "data/raw/seoul/boundaries/official_polygon/imported_20261112/source.metadata.json"
    )
    if not zoning_path.exists() or not boundary_path.exists():
        pytest.skip("Locally imported raw sidecars are absent")
    zoning_sidecar = json.loads(zoning_path.read_text(encoding="utf-8"))
    boundary_sidecar = json.loads(boundary_path.read_text(encoding="utf-8"))
    assert zoning_sidecar["first_available_at"] is None
    assert boundary_sidecar["historical_validity_verified"] is False
    result = gate.select_temporal_sources("2025Q4", [zoning_sidecar], [boundary_sidecar])
    assert result["temporal_status"] == "blocked"
    assert result["reason"] == "zoning_metadata_invalid"
    assert result["zoning"] is None and result["boundary"] is None
    assert result["rejected_versions"]["boundary"][0]["reason"] == "metadata_invalid"


def test_dates_are_required_even_if_flags_true_and_filenames_look_old():
    candidate = zoning(
        first_available_at=None,
        geometry_valid_at=None,
        source_listing_modified_date="2023-10-24",
        CREATE_DAT="18991230",
    )
    result = gate.select_temporal_sources("2025Q4", [candidate], [boundary()])
    assert result["reason"] == "zoning_availability_date_unverified"


def test_evidence_references_are_required_for_both_source_types():
    no_zone_evidence = zoning(availability_evidence=None)
    result = gate.select_temporal_sources("2025Q4", [no_zone_evidence], [boundary()])
    assert result["reason"] == "zoning_availability_evidence_missing"
    no_boundary_evidence = boundary(geometry_validity_evidence=".env")
    result = gate.select_temporal_sources("2025Q4", [zoning()], [no_boundary_evidence])
    assert result["reason"] == "boundary_geometry_evidence_missing"
    key_bearing_url = zoning(availability_evidence="https://api.vworld.kr/example?key=private")
    result = gate.select_temporal_sources("2025Q4", [key_bearing_url], [boundary()])
    assert result["reason"] == "zoning_availability_evidence_missing"


def test_overlapping_verified_boundaries_choose_latest_start():
    result = gate.select_temporal_sources(
        "2025Q4",
        [zoning()],
        [
            boundary("b1", geometry_valid_from="2023-10-20"),
            boundary("b2", geometry_valid_from="2025-06-01"),
        ],
    )
    assert result["temporal_status"] == "passed"
    assert result["boundary"]["version_id"] == "b2"


def test_ambiguous_or_duplicate_versions_are_blocked():
    result = gate.select_temporal_sources("2025Q4", [zoning("a"), zoning("b")], [boundary()])
    assert result["reason"] == "zoning_ambiguous_latest_version"
    result = gate.select_temporal_sources("2025Q4", [zoning("a"), zoning("a")], [boundary()])
    assert result["reason"] == "zoning_duplicate_version_id"


def test_zoning_explicit_validity_interval_must_cover_cutoff():
    valid = zoning(
        geometry_valid_at=None,
        geometry_valid_from="2025-07-01",
        geometry_valid_through="2025-12-31",
    )
    result = gate.select_temporal_sources("2025Q4", [valid], [boundary()])
    assert result["temporal_status"] == "passed"
    assert result["zoning"]["geometry_valid_from"] == "2025-07-01"
    expired = {**valid, "geometry_valid_through": "2025-09-29"}
    result = gate.select_temporal_sources("2025Q4", [expired], [boundary()])
    assert result["reason"] == "zoning_geometry_not_valid_at_cutoff"
