"""GEMD 정규화 검증은 작은 합성 도형만 사용한다."""

import csv
import json
from datetime import date

import pytest

from pickplace_ai.artifacts import file_sha256
from scripts import normalize_seoul_gemd_areas as gemd
from scripts.build_seoul_fit_reference import _read_areas

shapefile = pytest.importorskip("shapefile")
pyproj = pytest.importorskip("pyproj")
pytest.importorskip("shapely")

REFERENCE = "2026-06-30"
FIRST = "1111051500"
LAST = "1111053000"
MISSING = "1111054000"
UNMAPPED = "1111055000"
OUTSIDE = "2611051000"


def rectangle(x=950000, y=1950000, width=1000, height=2000):
    return [[x, y], [x, y + height], [x + width, y + height], [x + width, y], [x, y]]


def write_shapes(folder, rows, crs="EPSG:5179"):
    path = folder / "TL_SCCO_GEMD.shp"
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        writer.field("EMD_CD", "C", size=10)
        writer.field("EMD_KOR_NM", "C", size=40)
        for code, polygon in rows:
            if polygon is None:
                writer.null()
            else:
                writer.poly([polygon])
            writer.record(code, "synthetic")
    path.with_suffix(".prj").write_text(pyproj.CRS.from_user_input(crs).to_wkt())
    return path


def save_metadata(path, shapefile_path, bridge, **updates):
    metadata = {
        "reference_date": REFERENCE,
        "reference_date_verified": True,
        "source_url": "https://business.juso.go.kr/jst/jstAddressDownload?menu=24",
        "reference_date_basis": {
            "url": "https://business.juso.go.kr/synthetic-official-basis",
            "description": "Synthetic date evidence only; no actual source approval.",
        },
        "acquired_at": "2026-11-20",
        "components": {
            suffix: file_sha256(shapefile_path.with_suffix(suffix))
            for suffix in gemd.REQUIRED_COMPONENTS
        },
        "bridge": {
            "url": "https://kssc.mods.go.kr/synthetic-official-bridge",
            "sha256": file_sha256(bridge),
            "reference_date": REFERENCE,
        },
    }
    metadata.update(updates)
    path.write_text(json.dumps(metadata))
    return metadata


@pytest.fixture
def inputs(tmp_path):
    source = write_shapes(
        tmp_path,
        [
            (FIRST, rectangle()),
            (OUTSIDE, rectangle()),
            (UNMAPPED, rectangle()),
            (LAST, rectangle(width=500, height=1000)),
        ],
    )
    bridge = tmp_path / "bridge.csv"
    bridge.write_text(
        "reference_date,sgis_code,h10_code\n"
        f"{REFERENCE},11010720,{FIRST}\n"
        f"{REFERENCE},11010530,{LAST}\n"
        f"{REFERENCE},11010540,{MISSING}\n"
    )
    metadata = tmp_path / "metadata.json"
    save_metadata(metadata, source, bridge)
    return source, metadata, bridge, tmp_path / "normalized"


def test_normalizes_every_record_and_preserves_existing_builder_contract(inputs):
    source, metadata, bridge, output = inputs
    result = gemd.normalize(*inputs)
    assert result["input_record_count"] == 4
    assert result["seoul_record_count"] == 3
    assert result["row_count"] == 2
    assert result["exclusions"] == {
        "geometry_missing": 1,
        "official_bridge_missing": 1,
        "outside_seoul": 1,
    }
    assert _read_areas(output / "areas.csv", date(2026, 6, 30)) == {
        "11010720": pytest.approx(2_000_000),
        "11010530": pytest.approx(500_000),
    }
    assert result["area_source"] == {
        "url": json.loads(metadata.read_text())["source_url"],
        "sha256": file_sha256(output / "areas.csv"),
        "reference_date": REFERENCE,
        "acquired_at": "2026-11-20",
    }
    assert result["input_sha256"][".shp"] == file_sha256(source)
    assert result["input_sha256"]["bridge"] == file_sha256(bridge)
    assert result["bridge_identity"]["row_count"] == 3
    with (output / "excluded.csv").open(newline="") as handle:
        excluded = list(csv.DictReader(handle))
    assert {row["h10_code"] for row in excluded} == {OUTSIDE, UNMAPPED, MISSING}
    assert all("seoul_code" not in row for row in excluded)
    assert json.loads((output / "manifest.json").read_text()) == result
    with pytest.raises(FileExistsError):
        gemd.normalize(*inputs)


def test_projects_geographic_prj_to_metres(tmp_path):
    polygon = rectangle(126.97, 37.56, 0.01, 0.01)
    source = write_shapes(tmp_path, [(FIRST, polygon)], "EPSG:4326")
    bridge = tmp_path / "bridge.csv"
    bridge.write_text(f"reference_date,sgis_code,h10_code\n{REFERENCE},11010720,{FIRST}\n")
    metadata = tmp_path / "metadata.json"
    save_metadata(metadata, source, bridge)
    output = tmp_path / "normalized"
    result = gemd.normalize(source, metadata, bridge, output)
    area = _read_areas(output / "areas.csv", date(2026, 6, 30))["11010720"]
    assert 900_000 < area < 1_100_000
    assert result["area_projection"] == "EPSG:5179"
    assert result["area_unit"] == "m2"


@pytest.mark.parametrize("suffix", [".dbf", ".shx", ".prj"])
def test_requires_actual_sidecar_components(inputs, suffix):
    inputs[0].with_suffix(suffix).unlink()
    with pytest.raises(ValueError, match="requires SHP, DBF, SHX"):
        gemd.normalize(*inputs)
    assert not inputs[-1].exists()


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"reference_date_verified": False}, "verified reference date"),
        ({"source_url": "https://example.test/shape"}, "official HTTPS"),
        ({"acquired_at": "2026-06-29"}, "acquisition date"),
        (
            {"reference_date_basis": {"url": "https://business.juso.go.kr/", "description": ""}},
            "reference-date basis",
        ),
    ],
)
def test_requires_explicit_verified_official_date_evidence(inputs, updates, match):
    metadata = json.loads(inputs[1].read_text())
    metadata.update(updates)
    inputs[1].write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        gemd.normalize(*inputs)
    assert not inputs[-1].exists()


def test_rejects_revised_bridge_and_different_reference_date(inputs):
    source, metadata, bridge, output = inputs
    bridge.write_text(bridge.read_text().replace("11010720", "11010721"))
    with pytest.raises(ValueError, match="recorded SHA-256"):
        gemd.normalize(*inputs)
    save_metadata(metadata, source, bridge)
    bridge.write_text(bridge.read_text().replace(REFERENCE, "2026-03-31"))
    save_metadata(metadata, source, bridge)
    with pytest.raises(ValueError, match="code or reference date"):
        gemd.normalize(*inputs)
    assert not output.exists()


def test_rejects_ambiguous_official_bridge(inputs):
    source, metadata, bridge, output = inputs
    with bridge.open("a") as handle:
        handle.write(f"{REFERENCE},11010999,{FIRST}\n")
    save_metadata(metadata, source, bridge)
    with pytest.raises(ValueError, match="one-to-one"):
        gemd.normalize(*inputs)
    assert not output.exists()


def test_rejects_duplicate_h10_even_when_polygons_are_identical(inputs):
    source, metadata, bridge, output = inputs
    write_shapes(source.parent, [(FIRST, rectangle()), (FIRST, rectangle())])
    save_metadata(metadata, source, bridge)
    with pytest.raises(ValueError, match="Duplicated H10"):
        gemd.normalize(*inputs)
    assert not output.exists()


def test_deleted_dbf_record_cannot_shift_polygon_to_next_h10(inputs):
    source, metadata, bridge, output = inputs
    path = source.with_suffix(".dbf")
    content = bytearray(path.read_bytes())
    first_record = int.from_bytes(content[8:10], "little")
    content[first_record] = ord("*")
    path.write_bytes(content)
    save_metadata(metadata, source, bridge)
    with pytest.raises(ValueError, match="Deleted DBF rows"):
        gemd.normalize(*inputs)
    assert not output.exists()


def test_excludes_invalid_geometry_without_repair(inputs):
    source, metadata, bridge, output = inputs
    bowtie = [
        [950000, 1950000],
        [951000, 1951000],
        [950000, 1951000],
        [951000, 1950000],
        [950000, 1950000],
    ]
    write_shapes(source.parent, [(FIRST, bowtie), (LAST, rectangle())])
    save_metadata(metadata, source, bridge)
    result = gemd.normalize(*inputs)
    assert result["row_count"] == 1
    assert result["exclusions"] == {"geometry_missing": 1, "invalid_geometry": 1}
    assert "no repair" in result["transformation"]


def test_rejects_input_changed_during_processing(inputs, monkeypatch):
    read_bridge = gemd._read_bridge

    def mutate_after_read(path, reference):
        value = read_bridge(path, reference)
        path.write_text(path.read_text() + "\n")
        return value

    monkeypatch.setattr(gemd, "_read_bridge", mutate_after_read)
    with pytest.raises(ValueError, match="input changed"):
        gemd.normalize(*inputs)
    assert not inputs[-1].exists()
