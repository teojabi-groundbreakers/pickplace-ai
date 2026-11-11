"""Small geometric and codebook checks for the optional zoning preparation path."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from pickplace_ai.seoul_zoning_research import (
    CODE_SYSTEM,
    Codebook,
    SpatialFeature,
    ZoningPreparationError,
    _checked_members,
    _transform_polygon,
    area_shares,
    classify_record,
    prepare_version,
)

shapely_geometry = pytest.importorskip("shapely.geometry", reason="Optional GIS dependency")
Polygon = shapely_geometry.Polygon
box = shapely_geometry.box


def test_cross_class_overlap_and_unmapped_area_are_exclusive() -> None:
    rows = area_shares(
        {"12345678": box(0, 0, 10, 10)},
        [
            SpatialFeature("residential", box(0, 0, 6, 10)),
            SpatialFeature("commercial", box(4, 0, 8, 10)),
            SpatialFeature("unknown", box(8, 0, 9, 10), "unmapped_or_other_class"),
        ],
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["region_code_system"] == CODE_SYSTEM
    assert row["residential_share"] == pytest.approx(0.4)
    assert row["commercial_share"] == pytest.approx(0.2)
    assert row["industrial_share"] == 0
    assert row["green_share"] == 0
    assert row["unknown_share"] == pytest.approx(0.3)
    assert row["uncovered_share"] == pytest.approx(0.1)
    assert row["overlap_ambiguity_share"] == pytest.approx(0.2)
    assert sum(
        row[f"{name}_share"]
        for name in ("residential", "commercial", "industrial", "green", "unknown", "uncovered")
    ) == pytest.approx(1)
    assert row["training_eligible"] is False


def test_strict_codebook_and_name_check() -> None:
    book = Codebook(
        frozenset({("UQA100", "UQA120", "UQA121", "UQA121")}),
        {"UQA100": "주거지역"},
        "hash",
    )
    record = {
        "LCLAS_CL": "UQA100",
        "MLSFC_CL": "UQA120",
        "SCLAS_CL": "UQA121",
        "ATRB_SE": "UQA121",
        "DGM_NM": "제1종일반주거지역",
    }
    assert classify_record(record, book) == ("residential", None)
    assert classify_record({**record, "MLSFC_CL": "UQA100"}, book) == (
        "unknown",
        "code_hierarchy_mismatch",
    )
    assert classify_record({**record, "DGM_NM": "상업지역"}, book) == (
        "unknown",
        "name_category_mismatch",
    )
    assert classify_record({**record, "LCLAS_CL": "UQA999"}, book) == (
        "unknown",
        "unmapped_or_other_class",
    )


def test_zip_directory_is_allowed_but_traversal_is_rejected() -> None:
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        archive.writestr("폴더/", "")
        archive.writestr("폴더/source.prj", "crs")
    with zipfile.ZipFile(io.BytesIO(blob.getvalue())) as archive:
        assert _checked_members(archive, {".prj"}) == {".prj": b"crs"}
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        archive.writestr("../source.prj", "crs")
    with zipfile.ZipFile(io.BytesIO(blob.getvalue())) as archive:
        with pytest.raises(ZoningPreparationError, match="Unsafe"):
            _checked_members(archive, {".prj"})


def test_missing_source_sidecar_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ZoningPreparationError, match="metadata"):
        prepare_version("20260911", tmp_path, tmp_path / "boundary.zip", tmp_path / "out")


def test_geometry_repair_requires_spatial_preservation() -> None:
    class SourceShape:
        def __init__(self, polygon: Polygon) -> None:
            self.__geo_interface__ = polygon.__geo_interface__

    class IdentityProjection:
        def transform(self, x, y):
            return x, y

    ring_touch = Polygon([(0, 0), (2, 0), (2, 2), (1, 1), (0, 2), (0, 0), (1, 1)])
    repaired, was_invalid, safe = _transform_polygon(SourceShape(ring_touch), IdentityProjection())
    assert repaired is not None and repaired.is_valid
    assert was_invalid and safe

    bow_tie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    repaired, was_invalid, safe = _transform_polygon(SourceShape(bow_tie), IdentityProjection())
    assert repaired is not None and repaired.is_valid
    assert was_invalid and not safe
