"""Pinned supersession keeps unverified 2026 conflicts visible."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from pickplace_ai.seoul_zoning_conflicts import SourceZone
from pickplace_ai.seoul_zoning_research import SpatialFeature, ZoningPreparationError, area_shares
from pickplace_ai.seoul_zoning_supersession import (
    NEW_GREEN_DECISION_SN,
    NEW_GREEN_NOTICE_SN,
    NEW_GREEN_PRESENT_SN,
    PinnedSource,
    simulate_pinned_supersession,
)

box = pytest.importorskip("shapely.geometry", reason="Optional GIS dependency").box


def _zone(
    index: int, category: str, geometry, *, current: bool = True, source_id: str | None = None
) -> SourceZone:
    version = "20260911" if current else "20250805"
    record = {
        "PRESENT_SN": source_id or f"legacy-{index}",
        "NTFC_SN": "",
        "WTNNC_SN": "",
        "DGM_NM": "주거지역" if category != "green" else "자연녹지지역",
        "LCLAS_CL": "UQA100",
        "MLSFC_CL": "UQA130",
        "SCLAS_CL": "",
        "ATRB_SE": "UQA130",
    }
    if category == "green":
        record.update(
            PRESENT_SN=NEW_GREEN_PRESENT_SN,
            NTFC_SN=NEW_GREEN_NOTICE_SN,
            WTNNC_SN=NEW_GREEN_DECISION_SN,
            LCLAS_CL="UQA400",
            MLSFC_CL="UQA430",
            SCLAS_CL="",
            ATRB_SE="UQA430",
        )
    return SourceZone(
        source_row_index=index,
        feature_key=f"{version}:{index}:{hashlib.sha256(geometry.wkb).hexdigest()}",
        record=record,
        category=category,
        reason="code_hierarchy_mismatch" if category == "unknown" else None,
        geometry=geometry,
    )


def test_only_exact_prior_known_polygons_are_clipped() -> None:
    prior_residential = _zone(
        0, "residential", box(0, 0, 6, 10), current=False, source_id="legacy-a"
    )
    current_residential = _zone(4, "residential", box(0, 0, 6, 10), source_id="legacy-a")
    changed_residential = _zone(5, "residential", box(6, 0, 10, 10))
    current_unknown = _zone(6, "unknown", box(5, 0, 6, 10))
    new_green = _zone(7, "green", box(4, 0, 8, 10))
    pin = PinnedSource("synthetic-archive", 7, hashlib.sha256(new_green.geometry.wkb).hexdigest())
    current = [current_residential, changed_residential, current_unknown, new_green]

    simulated, audit = simulate_pinned_supersession([prior_residential], current, pin)

    by_index = {zone.source_row_index: zone for zone in simulated}
    assert by_index[4].geometry.area == pytest.approx(40)
    assert by_index[5].geometry.equals(changed_residential.geometry)
    assert by_index[6].geometry.equals(current_unknown.geometry)
    assert by_index[7].geometry.equals(new_green.geometry)
    assert by_index[6].geometry.intersection(by_index[7].geometry).area == pytest.approx(10)
    assert audit["exact_prior_rows_clipped"] == 1
    assert audit["clipped_area_union_m2"] == pytest.approx(20)
    assert audit["new_green_known_overlap_before_m2"] == pytest.approx(40)
    assert audit["new_green_known_overlap_unresolved_m2"] == pytest.approx(20)
    boundary = {"12345678": box(0, 0, 10, 10)}
    strict_share = area_shares(
        boundary, [SpatialFeature(zone.category, zone.geometry, zone.reason) for zone in current]
    )[0]
    scenario_share = area_shares(
        boundary, [SpatialFeature(zone.category, zone.geometry, zone.reason) for zone in simulated]
    )[0]
    assert strict_share["overlap_ambiguity_share"] == pytest.approx(0.4)
    assert scenario_share["overlap_ambiguity_share"] == pytest.approx(0.3)
    assert scenario_share["overlap_unknown_involved_share"] == pytest.approx(0.1)
    assert scenario_share["training_eligible"] is False


def test_changed_pinned_source_fails_closed() -> None:
    new_green = _zone(7, "green", box(4, 0, 8, 10))
    wrong_pin = PinnedSource("synthetic-archive", 7, "incorrect-geometry-hash")
    with pytest.raises(ZoningPreparationError, match="changed"):
        simulate_pinned_supersession([], [new_green], wrong_pin)


def test_changed_notice_or_decision_is_not_treated_as_prior_record() -> None:
    prior = _zone(0, "residential", box(0, 0, 6, 10), current=False, source_id="legacy-a")
    current = _zone(4, "residential", box(0, 0, 6, 10), source_id="legacy-a")
    new_green = _zone(7, "green", box(4, 0, 8, 10))
    pin = PinnedSource("synthetic-archive", 7, hashlib.sha256(new_green.geometry.wkb).hexdigest())

    for changed_field in ("NTFC_SN", "WTNNC_SN"):
        changed = replace(current, record={**current.record, changed_field: "new-decision"})
        simulated, audit = simulate_pinned_supersession([prior], [changed, new_green], pin)
        assert simulated[0].geometry.equals(current.geometry)
        assert audit["exact_prior_rows_clipped"] == 0
        assert audit["new_green_known_overlap_unresolved_m2"] == pytest.approx(20)
