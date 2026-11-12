"""Pair provenance must explain, without double-counting, zoning ambiguity."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from pickplace_ai.seoul_zoning_conflicts import SourceZone, diagnose_region, load_source_zones
from pickplace_ai.seoul_zoning_research import ZoningPreparationError

shapely_geometry = pytest.importorskip("shapely.geometry", reason="Optional GIS dependency")
box = shapely_geometry.box


def _zone(index: int, category: str, geometry, reason: str | None = None) -> SourceZone:
    return SourceZone(
        source_row_index=index,
        feature_key=f"20260911:{index}:geometry-{index}",
        record={
            "PRESENT_SN": "reused-source-id",
            "NTFC_SN": "notice-1",
            "WTNNC_SN": "decision-1",
            "DGM_NM": category,
            "LCLAS_CL": "UQA100",
            "MLSFC_CL": "",
            "SCLAS_CL": "",
            "ATRB_SE": "",
        },
        category=category,
        reason=reason,
        geometry=geometry,
        review_candidate_class="residential" if category == "unknown" else None,
    )


def test_pair_intersections_are_not_additive_and_reused_ids_keep_row_lineage() -> None:
    boundary = box(0, 0, 10, 10)
    zones = [
        _zone(0, "residential", box(0, 0, 8, 10)),
        _zone(1, "commercial", box(4, 0, 10, 10)),
        _zone(2, "unknown", box(5, 0, 7, 10), "code_hierarchy_mismatch"),
        _zone(3, "residential", box(0, 0, 8, 10)),
    ]
    pairs, summary = diagnose_region("20260911", "12345678", "테스트동", boundary, zones)
    assert summary["overlap_ambiguity_share"] == pytest.approx(0.4)
    assert summary["training_eligible"] is False
    assert len(pairs) == 5  # The two residential records overlap, but form no pair together.
    assert sum(row["pair_intersection_share"] for row in pairs) > 1.0
    assert {row["pair_type"] for row in pairs} == {"cross_known_class", "unknown_involved"}
    assert any(
        row["right_source_row_index"] == 2 and row["right_reason"] == "code_hierarchy_mismatch"
        for row in pairs
    )
    assert any(row["right_review_candidate_class"] == "residential" for row in pairs)
    assert all(row["left_feature_key"] != row["right_feature_key"] for row in pairs)
    assert {row["left_PRESENT_SN"] for row in pairs} == {"reused-source-id"}


def test_minimum_pair_area_filters_rows_but_not_unioned_ambiguity() -> None:
    boundary = box(0, 0, 10, 10)
    zones = [
        _zone(0, "residential", box(0, 0, 6, 10)),
        _zone(1, "commercial", box(5, 0, 10, 10)),
    ]
    pairs, summary = diagnose_region(
        "20260911", "12345678", "테스트동", boundary, zones, min_pair_area_m2=11
    )
    assert pairs == []
    assert summary["overlap_ambiguity_share"] == pytest.approx(0.1)
    assert summary["pair_count"] == 0


def test_missing_source_identifiers_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    fields = [
        (name, "C", 40, 0)
        for name in ("DeletionFlag", "DGM_NM", "LCLAS_CL", "MLSFC_CL", "SCLAS_CL", "ATRB_SE")
    ]
    fake_reader = SimpleNamespace(fields=fields)
    monkeypatch.setattr(
        "pickplace_ai.seoul_zoning_conflicts._read_shapefile",
        lambda *args: nullcontext(fake_reader),
    )
    with pytest.raises(ZoningPreparationError, match="identifier fields"):
        load_source_zones(Path("unused.zip"), None, "20260911")


@pytest.mark.parametrize("invalid_limit", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_minimum_pair_area_is_rejected(invalid_limit: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        diagnose_region(
            "20260911",
            "12345678",
            "테스트동",
            box(0, 0, 10, 10),
            [],
            min_pair_area_m2=invalid_limit,
        )
