"""Small geometric and codebook checks for the optional zoning preparation path."""

from __future__ import annotations

import io
import json
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
    load_zone_geometries,
    prepare_version,
    repeated_midcode_review_candidate,
    sha256,
)

shapely_geometry = pytest.importorskip("shapely.geometry", reason="Optional GIS dependency")
shapefile = pytest.importorskip("shapefile", reason="Optional GIS dependency")
pyproj = pytest.importorskip("pyproj", reason="Optional GIS dependency")
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
    assert row["overlap_known_only_share"] == pytest.approx(0.2)
    assert row["overlap_unknown_involved_share"] == 0
    assert sum(
        row[f"{name}_share"]
        for name in ("residential", "commercial", "industrial", "green", "unknown", "uncovered")
    ) == pytest.approx(1)
    assert row["training_eligible"] is False


def test_overlap_breakdown_does_not_double_count_triple_overlap() -> None:
    row = area_shares(
        {"12345678": box(0, 0, 10, 10)},
        [
            SpatialFeature("residential", box(0, 0, 7, 10)),
            SpatialFeature("commercial", box(4, 0, 9, 10)),
            SpatialFeature("unknown", box(5, 0, 6, 10), "code_hierarchy_mismatch"),
        ],
    )[0]
    assert row["overlap_ambiguity_share"] == pytest.approx(0.3)
    assert row["overlap_unknown_involved_share"] == pytest.approx(0.1)
    assert row["overlap_known_only_share"] == pytest.approx(0.2)
    assert row["unknown_share"] == pytest.approx(0.3)


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


def test_repeated_midcode_is_review_hint_only() -> None:
    book = Codebook(
        frozenset({("UQA100", "UQA130", "", "UQA130")}),
        {"UQA100": "주거지역"},
        "hash",
    )
    record = {
        "LCLAS_CL": "UQA100",
        "MLSFC_CL": "UQA130",
        "SCLAS_CL": "UQA130",
        "ATRB_SE": "UQA130",
        "DGM_NM": "준주거지역",
    }
    assert classify_record(record, book) == ("unknown", "code_hierarchy_mismatch")
    assert repeated_midcode_review_candidate(record, book) == "residential"
    assert repeated_midcode_review_candidate({**record, "DGM_NM": "상업지역"}, book) is None
    assert repeated_midcode_review_candidate({**record, "MLSFC_CL": "UQA100"}, book) is None
    assert repeated_midcode_review_candidate({**record, "SCLAS_CL": ""}, book) is None


def test_alias_scenario_is_separate_from_strict_output(tmp_path: Path) -> None:
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    with shapefile.Writer(
        shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON, encoding="cp949"
    ) as writer:
        for field in ("LCLAS_CL", "MLSFC_CL", "SCLAS_CL", "ATRB_SE", "DGM_NM"):
            writer.field(field, "C", size=40)
        ring = [
            (200000, 450000),
            (200000, 450100),
            (200100, 450100),
            (200100, 450000),
            (200000, 450000),
        ]
        for small in ("", "UQA130"):
            writer.poly([ring])
            writer.record("UQA100", "UQA130", small, "UQA130", "준주거지역")
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("zone.shp", shp.getvalue())
        zipped.writestr("zone.shx", shx.getvalue())
        zipped.writestr("zone.dbf", dbf.getvalue())
        zipped.writestr("zone.prj", pyproj.CRS.from_epsg(5174).to_wkt())
    book = Codebook(
        frozenset({("UQA100", "UQA130", "", "UQA130")}),
        {"UQA100": "주거지역"},
        "hash",
    )
    strict, strict_quality = load_zone_geometries(archive, book)
    simulated, simulated_quality = load_zone_geometries(
        archive, book, classification_scenario="repeated_midcode_alias"
    )
    assert [feature.category for feature in strict] == ["residential", "unknown"]
    assert [feature.category for feature in simulated] == ["residential", "residential"]
    assert strict_quality["reviewable_repeated_midcode_rows"] == 1
    assert simulated_quality["simulated_repeated_midcode_alias_rows"] == 1
    boundary = {"12345678": strict[0].geometry}
    assert area_shares(boundary, strict)[0]["overlap_ambiguity_share"] == pytest.approx(1)
    assert area_shares(boundary, simulated)[0]["overlap_ambiguity_share"] == 0
    assert area_shares(boundary, simulated)[0]["training_eligible"] is False


def test_alias_scenario_rejects_an_unreviewed_source_archive(tmp_path: Path) -> None:
    source = tmp_path / "20260911" / "source.zip"
    source.parent.mkdir()
    source.write_bytes(b"different source archive")
    (source.parent / "source.metadata.json").write_text(
        json.dumps({"archive_sha256": sha256(source)}), encoding="utf-8"
    )
    with pytest.raises(ZoningPreparationError, match="not reviewed"):
        prepare_version(
            "20260911",
            tmp_path,
            tmp_path / "missing_boundary.zip",
            tmp_path / "output",
            classification_scenario="repeated_midcode_alias",
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
