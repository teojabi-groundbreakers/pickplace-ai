"""경기 지리 집계 테스트는 작은 합성 도형·점포만 사용한다."""

import csv
import io
import zipfile

import pytest

from scripts import gyeonggi_fit_geometry as geo

pyproj = pytest.importorskip("pyproj")
shapefile = pytest.importorskip("shapefile")
shapely = pytest.importorskip("shapely")

A = "4111051000"
B = "4111052000"


def write_stores(path, entries):
    converter = pyproj.Transformer.from_crs(5179, 4326, always_xy=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(geo.DEFAULT_COLUMNS.values())
        for identifier, code, industry, x, y in entries:
            lon, lat = converter.transform(x, y)
            writer.writerow([identifier, code, industry, lon, lat])


def test_pip_assigns_exact_geometry_without_converting_source8(tmp_path):
    boundaries = {
        A: shapely.box(950000, 1950000, 951000, 1951000),
        B: shapely.box(952000, 1950000, 953000, 1951000),
    }
    path = tmp_path / "stores.csv"
    write_stores(
        path,
        [
            ("S1", "99999999", "I21201", 950200, 1950200),
            ("S2", "99999999", "I20101", 950300, 1950300),
            ("S3", "99999999", "I21201", 952200, 1950200),
            ("S4", "41171582", "I21201", 950400, 1950400),
        ],
    )
    counts, audit = geo.locate_stores(
        path, boundaries, {A}, batch_size=2, excluded_source_codes={"41171582"}
    )
    assert counts == {(A, "I21201"): 1, (A, "I20101"): 1}
    assert audit["input_rows"] == audit["unique_store_ids"] == 4
    assert audit["exclusions"] == {"excluded_source_code": 1, "outside_reporting_scope": 1}
    assert audit["excluded_source_rows_by_code"] == {"41171582": 1}
    assert audit["source_to_geometry_pairs"] == [
        {"source_code": "99999999", "geometry_h10": A, "rows": 2},
        {"source_code": "99999999", "geometry_h10": B, "rows": 1},
    ]


@pytest.mark.parametrize(
    "code,identifier,match",
    [
        ("41110510", "S1", "Duplicate original store"),
        ("4111051000", "S2", "eight-digit"),
    ],
)
def test_rejects_duplicate_ids_and_inferred_source_codes(tmp_path, code, identifier, match):
    path = tmp_path / "stores.csv"
    write_stores(
        path,
        [
            ("S1", "41110510", "I21201", 950200, 1950200),
            (identifier, code, "I21201", 950300, 1950300),
        ],
    )
    with pytest.raises(ValueError, match=match):
        geo.locate_stores(path, {A: shapely.box(950000, 1950000, 951000, 1951000)}, {A})


def test_pip_uses_full_source_and_rejects_overlap_with_unreported_scope(tmp_path):
    path = tmp_path / "stores.csv"
    write_stores(path, [("S1", "41110510", "I21201", 950500, 1950500)])
    polygon = shapely.box(950000, 1950000, 951000, 1951000)
    counts, audit = geo.locate_stores(path, {A: polygon, B: polygon}, {A})
    assert not counts
    assert audit["exclusions"] == {"overlapping_boundaries": 1}


def test_invalid_bbox_records_valid_adjacent_hit_for_review(tmp_path):
    path = tmp_path / "stores.csv"
    write_stores(path, [("S1", "41110510", "I21201", 950200, 1950200)])
    invalid = shapely.Polygon(
        [
            (950000, 1950000),
            (951000, 1951000),
            (950000, 1951000),
            (951000, 1950000),
            (950000, 1950000),
        ]
    )
    counts, audit = geo.locate_stores(
        path, {A: shapely.box(950000, 1950000, 951000, 1951000), B: invalid}, {A}
    )
    assert not counts
    assert audit["exclusions"] == {"invalid_source_geometry_bbox": 1}
    assert audit["invalid_bbox_valid_hit_rows"] == 1
    assert audit["invalid_bbox_valid_hit_counts"] == {A: 1}


def make_archive(path, *, prj=None, deleted=False):
    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    with shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shapefile.POLYGON) as writer:
        writer.field("EMD_CD", "C", size=10)
        writer.field("EMD_KOR_NM", "C", size=40)
        writer.poly(
            [
                [
                    (950000, 1950000),
                    (950000, 1951000),
                    (951000, 1951000),
                    (951000, 1950000),
                    (950000, 1950000),
                ]
            ]
        )
        writer.record(A, "synthetic")
    attributes = bytearray(dbf.getvalue())
    if deleted:
        attributes[int.from_bytes(attributes[8:10], "little")] = ord("*")
    with zipfile.ZipFile(path, "w") as archive:
        for ext, content in [
            ("shp", shp.getvalue()),
            ("shx", shx.getvalue()),
            ("dbf", bytes(attributes)),
        ]:
            archive.writestr(f"41000/TL_SCCO_GEMD.{ext}", content)
        if prj:
            archive.writestr("41000/TL_SCCO_GEMD.prj", pyproj.CRS.from_user_input(prj).to_wkt())


def test_inspects_documented_crs_and_all_original_h10(tmp_path):
    path = tmp_path / "boundary.zip"
    make_archive(path)
    boundaries, audit = geo.inspect_boundaries(path)
    assert set(boundaries) == {A}
    assert boundaries[A].area == pytest.approx(1_000_000)
    assert audit["original_prj_present"] is False
    assert audit["coordinate_reference_system"] == "EPSG:5179"
    assert not audit["invalid"]


@pytest.mark.parametrize(
    "prj,deleted,match",
    [
        ("EPSG:4326", False, "PRJ conflicts"),
        (None, True, "Deleted DBF"),
    ],
)
def test_rejects_conflicting_prj_and_deleted_dbf(tmp_path, prj, deleted, match):
    path = tmp_path / "boundary.zip"
    make_archive(path, prj=prj, deleted=deleted)
    with pytest.raises(ValueError, match=match):
        geo.inspect_boundaries(path)
