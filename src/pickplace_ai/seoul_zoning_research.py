"""Build descriptive Seoul administrative-dong zoning shares from official ZIPs.

These shares are research candidates. Source publication and geometry validity
dates remain unverified, so output is never declared training eligible.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

CLASSES = ("residential", "commercial", "industrial", "green")
CODE_TO_CLASS = {
    "UQA100": "residential",
    "UQA200": "commercial",
    "UQA300": "industrial",
    "UQA400": "green",
}
CLASS_TO_NAME_TOKEN = {
    "residential": "주거",
    "commercial": "상업",
    "industrial": "공업",
    "green": "녹지",
}
UNKNOWN_CLASS = "unknown"
AREA_CRS = "EPSG:6933"
OVERLAY_GRID_METRES = 0.01
CODE_SYSTEM = "seoul-adstrd-8"
_XLSX_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


class ZoningPreparationError(ValueError):
    """A source archive or prepared feature failed a quality gate."""


@dataclass(frozen=True)
class Codebook:
    allowed_hierarchy: frozenset[tuple[str, str, str, str]]
    broad_codes: dict[str, str]
    workbook_sha256: str


@dataclass(frozen=True)
class SpatialFeature:
    category: str
    geometry: object
    reason: str | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_members(archive: zipfile.ZipFile, required: set[str]) -> dict[str, bytes]:
    """Select the one shapefile set by suffix; ZIP path encodings vary by year."""
    members = archive.infolist()
    if not members or any(
        member.flag_bits & 1
        or member.filename.startswith("/")
        or "\\" in member.filename
        or ".." in Path(member.filename).parts
        or (member.external_attr >> 16) & 0o170000 == 0o120000
        for member in members
    ):
        raise ZoningPreparationError("Unsafe or encrypted source ZIP")
    if archive.testzip() is not None:
        raise ZoningPreparationError("Source ZIP CRC check failed")
    chosen = {}
    for extension in required:
        matches = [
            member
            for member in members
            if not member.is_dir() and Path(member.filename).suffix.lower() == extension
        ]
        if len(matches) != 1:
            raise ZoningPreparationError(f"Expected one {extension} member")
        chosen[extension] = archive.read(matches[0])
    return chosen


def _load_xlsx_cells(blob: bytes) -> list[dict[str, str]]:
    """Read the small official classification sheet without a spreadsheet runtime."""
    with zipfile.ZipFile(io.BytesIO(blob)) as workbook:
        if "xl/worksheets/sheet1.xml" not in workbook.namelist():
            raise ZoningPreparationError("Official codebook sheet is missing")
        strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            strings = [
                "".join(item.text or "" for item in entry.findall(".//m:t", _XLSX_NS))
                for entry in root.findall("m:si", _XLSX_NS)
            ]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(".//m:sheetData/m:row", _XLSX_NS):
        cells = {}
        for cell in row.findall("m:c", _XLSX_NS):
            value = cell.find("m:v", _XLSX_NS)
            content = value.text if value is not None else ""
            if cell.get("t") == "s" and content:
                content = strings[int(content)]
            elif cell.get("t") == "inlineStr":
                content = "".join(item.text or "" for item in cell.findall(".//m:t", _XLSX_NS))
            column = re.match(r"[A-Z]+", cell.get("r", ""))
            if column:
                cells[column.group()] = (content or "").strip()
        rows.append(cells)
    return rows


def load_codebook(archive_path: Path) -> Codebook:
    with zipfile.ZipFile(archive_path) as archive:
        blobs = _checked_members(archive, {".xlsx"})
    blob = blobs[".xlsx"]
    rows = _load_xlsx_cells(blob)
    allowed = set()
    broad = {}
    for cells in rows:
        code = cells.get("F", "")
        if not code.startswith("UQA"):
            continue
        name = cells.get("E", "")
        if code in broad and broad[code] != name:
            raise ZoningPreparationError("Conflicting broad codebook class")
        broad[code] = name
        allowed.add((code, cells.get("H", ""), cells.get("J", ""), cells.get("L", "")))
    expected = {
        "UQA100": "주거지역",
        "UQA200": "상업지역",
        "UQA300": "공업지역",
        "UQA400": "녹지지역",
        "UQA500": "미지정지역",
        "UQA999": "기타",
        "UQA000": "미분류",
    }
    if broad != expected or len(allowed) != 26:
        raise ZoningPreparationError("Codebook changed; review its classes before preparing shares")
    return Codebook(frozenset(allowed), broad, hashlib.sha256(blob).hexdigest())


def classify_record(record: dict, codebook: Codebook) -> tuple[str, str | None]:
    code = str(record.get("LCLAS_CL") or "").strip()
    category = CODE_TO_CLASS.get(code)
    if category is None:
        return UNKNOWN_CLASS, "unmapped_or_other_class"
    hierarchy = tuple(
        str(record.get(field) or "").strip()
        for field in ("LCLAS_CL", "MLSFC_CL", "SCLAS_CL", "ATRB_SE")
    )
    if hierarchy not in codebook.allowed_hierarchy:
        return UNKNOWN_CLASS, "code_hierarchy_mismatch"
    name = str(record.get("DGM_NM") or "").strip()
    token = CLASS_TO_NAME_TOKEN[category]
    if token not in name or any(
        other != category and other_token in name
        for other, other_token in CLASS_TO_NAME_TOKEN.items()
    ):
        return UNKNOWN_CLASS, "name_category_mismatch"
    return category, None


def _read_shapefile(archive_path: Path, expected_epsg: int, encoding: str):
    try:
        import shapefile
        from pyproj import CRS
    except ImportError as exc:
        raise ZoningPreparationError(
            "Install optional zoning dependencies from requirements-zoning.txt"
        ) from exc
    with zipfile.ZipFile(archive_path) as archive:
        parts = _checked_members(archive, {".shp", ".shx", ".dbf", ".prj"})
    crs = CRS.from_wkt(parts[".prj"].decode("utf-8"))
    if crs.to_epsg() != expected_epsg:
        raise ZoningPreparationError(f"Source CRS differs from expected EPSG:{expected_epsg}")
    reader = shapefile.Reader(
        shp=io.BytesIO(parts[".shp"]),
        shx=io.BytesIO(parts[".shx"]),
        dbf=io.BytesIO(parts[".dbf"]),
        encoding=encoding,
    )
    return reader


def _polygonal(geometry):
    import shapely

    if geometry.is_empty:
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if geometry.geom_type == "GeometryCollection":
        polygon_parts = [
            part for part in geometry.geoms if part.geom_type in {"Polygon", "MultiPolygon"}
        ]
        return shapely.union_all(polygon_parts) if polygon_parts else None
    return None


def _polygon_or_empty(geometry):
    from shapely.geometry import Polygon

    polygon = _polygonal(geometry)
    return polygon if polygon is not None else Polygon()


def _transform_polygon(raw_shape, transformer) -> tuple[object | None, bool, bool]:
    import shapely
    from shapely.geometry import shape

    geometry = shape(raw_shape.__geo_interface__)
    repaired = not geometry.is_valid
    repair_preserved_area = False
    if repaired:
        original_area = geometry.area
        reason = shapely.is_valid_reason(geometry)
        original = geometry
        geometry = _polygonal(shapely.make_valid(geometry))
        try:
            symmetric_difference_ratio = (
                shapely.symmetric_difference(original, geometry).area / original_area
                if geometry is not None and original_area > 0
                else float("inf")
            )
        except shapely.GEOSException:
            symmetric_difference_ratio = float("inf")
        repair_preserved_area = (
            geometry is not None
            and original_area > 0
            and reason.startswith("Ring Self-intersection")
            and abs(geometry.area - original_area) / original_area <= 0.000001
            and symmetric_difference_ratio <= 0.000001
        )
    if geometry is None or geometry.is_empty:
        return None, repaired, False
    geometry = shapely.transform(geometry, transformer.transform, interleaved=False)
    if not geometry.is_valid:
        repaired = True
        projected_original = geometry
        projected_area = geometry.area
        geometry = _polygonal(shapely.make_valid(geometry))
        try:
            projected_symdiff_ratio = (
                shapely.symmetric_difference(projected_original, geometry).area / projected_area
                if geometry is not None and projected_area > 0
                else float("inf")
            )
        except shapely.GEOSException:
            projected_symdiff_ratio = float("inf")
        repair_preserved_area = (
            repair_preserved_area
            and geometry is not None
            and projected_area > 0
            and abs(geometry.area - projected_area) / projected_area <= 0.000001
            and projected_symdiff_ratio <= 0.000001
        )
    if geometry is None or geometry.is_empty or geometry.area <= 0:
        return None, repaired, False
    geometry = _polygonal(shapely.set_precision(geometry, OVERLAY_GRID_METRES))
    if geometry is None or geometry.is_empty or geometry.area <= 0:
        return None, repaired, False
    return geometry, repaired, repair_preserved_area


def load_admin_geometries(
    archive_path: Path,
) -> tuple[dict[str, object], dict[str, str], Counter]:
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:5181", AREA_CRS, always_xy=True)
    result = {}
    names = {}
    quality = Counter()
    with _read_shapefile(archive_path, 5181, "utf-8") as reader:
        fields = {field[0] for field in reader.fields[1:]}
        if not {"ADSTRD_CD", "ADSTRD_NM"} <= fields:
            raise ZoningPreparationError("Administrative boundary code fields are missing")
        for shape_record in reader.iterShapeRecords():
            code = str(shape_record.record["ADSTRD_CD"]).strip()
            if not re.fullmatch(r"\d{8}", code) or code in result:
                raise ZoningPreparationError("Invalid or duplicate 8-digit administrative code")
            geometry, repaired, _ = _transform_polygon(shape_record.shape, transformer)
            if geometry is None or repaired:
                raise ZoningPreparationError("Administrative polygon is invalid or empty")
            result[code] = geometry
            names[code] = str(shape_record.record["ADSTRD_NM"]).strip()
            quality["admin_polygons"] += 1
    if not result:
        raise ZoningPreparationError("Administrative boundary ZIP has no polygons")
    return result, names, quality


def load_zone_geometries(
    archive_path: Path, codebook: Codebook
) -> tuple[list[SpatialFeature], Counter]:
    import shapefile
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:5174", AREA_CRS, always_xy=True)
    zones = []
    quality = Counter()
    with _read_shapefile(archive_path, 5174, "cp949") as reader:
        fields = {field[0] for field in reader.fields[1:]}
        if not {"LCLAS_CL", "MLSFC_CL", "SCLAS_CL", "ATRB_SE", "DGM_NM"} <= fields:
            raise ZoningPreparationError("Zoning class fields are missing")
        if reader.numRecords != reader.numShapes:
            raise ZoningPreparationError("Zoning SHP/DBF row counts differ")
        for row_index in range(reader.numRecords):
            quality["source_zone_rows"] += 1
            record = reader.record(row_index).as_dict()
            category, reason = classify_record(record, codebook)
            try:
                source_shape = reader.shape(row_index)
            except shapefile.ShapefileException as exc:
                if "requires non-empty points" not in str(exc):
                    raise
                quality["empty_or_nonpolygon_zone_rows"] += 1
                continue
            if source_shape.shapeType == shapefile.NULL:
                quality["empty_or_nonpolygon_zone_rows"] += 1
                continue
            geometry, repaired, repair_preserved_area = _transform_polygon(
                source_shape, transformer
            )
            if geometry is None:
                quality["empty_or_nonpolygon_zone_rows"] += 1
                continue
            if repaired:
                quality["repaired_invalid_zone_rows"] += 1
                quality["repaired_zone_area_m2"] += geometry.area
                if repair_preserved_area:
                    quality["repaired_area_preserved_zone_rows"] += 1
                else:
                    category, reason = UNKNOWN_CLASS, "unsafe_geometry_repair"
            if reason:
                quality[f"excluded_{reason}"] += 1
            else:
                quality[f"classified_{category}"] += 1
            zones.append(SpatialFeature(category, geometry, reason))
    if not zones:
        raise ZoningPreparationError("Zoning ZIP has no usable polygon geometry")
    return zones, quality


def area_shares(
    admin: dict[str, object],
    zones: list[SpatialFeature],
    admin_names: dict[str, str] | None = None,
) -> list[dict]:
    """Give ambiguous overlaps to unknown; each row's area shares sum to one."""
    import shapely
    from shapely.strtree import STRtree

    geometries = [zone.geometry for zone in zones]
    tree = STRtree(geometries)
    rows = []
    for region_code, boundary in sorted(admin.items()):
        if boundary.area <= 0:
            raise ZoningPreparationError("Administrative polygon has zero area")
        pieces = defaultdict(list)
        unknown_reasons = set()
        intersects = tree.query(boundary, predicate="intersects")
        for index in intersects:
            zone = zones[int(index)]
            clipped = shapely.intersection(boundary, zone.geometry, grid_size=OVERLAY_GRID_METRES)
            if not clipped.is_empty and clipped.area > 0:
                pieces[zone.category].append(clipped)
                if zone.category == UNKNOWN_CLASS and zone.reason:
                    unknown_reasons.add(zone.reason)
        unions = {
            category: _polygon_or_empty(
                shapely.union_all(pieces.get(category, []), grid_size=OVERLAY_GRID_METRES)
            )
            for category in (*CLASSES, UNKNOWN_CLASS)
        }
        pair_overlaps = [
            _polygon_or_empty(
                shapely.intersection(unions[left], unions[right], grid_size=OVERLAY_GRID_METRES)
            )
            for index, left in enumerate((*CLASSES, UNKNOWN_CLASS))
            for right in (*CLASSES, UNKNOWN_CLASS)[index + 1 :]
            if not unions[left].is_empty and not unions[right].is_empty
        ]
        ambiguity = _polygon_or_empty(
            shapely.union_all(pair_overlaps, grid_size=OVERLAY_GRID_METRES)
        )
        unknown = _polygon_or_empty(
            shapely.union_all([unions[UNKNOWN_CLASS], ambiguity], grid_size=OVERLAY_GRID_METRES)
        )
        known = {}
        already_assigned = unknown
        for category in CLASSES:
            known[category] = _polygon_or_empty(
                shapely.difference(
                    unions[category], already_assigned, grid_size=OVERLAY_GRID_METRES
                )
            )
            already_assigned = _polygon_or_empty(
                shapely.union_all(
                    [already_assigned, known[category]], grid_size=OVERLAY_GRID_METRES
                )
            )
        uncovered = _polygon_or_empty(
            shapely.difference(boundary, already_assigned, grid_size=OVERLAY_GRID_METRES)
        )
        denominator = boundary.area
        shares = {
            f"{category}_share": max(0.0, known[category].area / denominator)
            for category in CLASSES
        }
        shares["unknown_share"] = max(0.0, unknown.area / denominator)
        shares["overlap_ambiguity_share"] = max(0.0, ambiguity.area / denominator)
        assigned_keys = [*(f"{category}_share" for category in CLASSES), "unknown_share"]
        assigned_total = sum(shares[key] for key in assigned_keys)
        measured_uncovered = max(0.0, uncovered.area / denominator)
        raw_total = assigned_total + measured_uncovered
        if abs(raw_total - 1.0) > 0.0001 or assigned_total > 1.0001:
            raise ZoningPreparationError(
                f"Area partition exceeds numerical tolerance for {region_code}: "
                f"raw_total={raw_total}, assigned_total={assigned_total}"
            )
        if assigned_total > 1:
            for key in assigned_keys:
                shares[key] /= assigned_total
            assigned_total = 1.0
        shares["uncovered_share"] = max(0.0, 1.0 - assigned_total)
        quality_reasons = []
        if shares["unknown_share"] > 0.05:
            quality_reasons.append("unknown_area_over_5pct")
        if shares["overlap_ambiguity_share"] > 0.05:
            quality_reasons.append("overlap_ambiguity_over_5pct")
        if shares["uncovered_share"] > 0.01:
            quality_reasons.append("uncovered_area_over_1pct")
        rows.append(
            {
                "region_code_system": CODE_SYSTEM,
                "region_code": region_code,
                "region_name": (admin_names or {}).get(region_code, ""),
                "admin_area_m2": round(denominator, 3),
                **{key: round(value, 10) for key, value in shares.items()},
                "measured_uncovered_share": round(measured_uncovered, 10),
                "numerical_partition_residual_share": round(raw_total - 1.0, 10),
                "intersecting_zone_rows": len(intersects),
                "unknown_source_reasons": "|".join(sorted(unknown_reasons)),
                "quality_status": "review_required" if quality_reasons else "candidate",
                "quality_reasons": "|".join(quality_reasons),
                "training_eligible": False,
            }
        )
    return rows


def _read_source_metadata(path: Path) -> dict:
    metadata_path = path.with_name("source.metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise ZoningPreparationError("Source ZIP or its metadata is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("archive_sha256") != sha256(path):
        raise ZoningPreparationError("Source ZIP hash does not match its metadata")
    return metadata


def prepare_version(
    version_id: str,
    zoning_root: Path,
    boundary_archive: Path,
    output_root: Path,
    *,
    codebook_reference_version: str = "20241107",
) -> tuple[Path, dict]:
    if not re.fullmatch(r"20\d{6}", version_id):
        raise ZoningPreparationError("Version ID must be YYYYMMDD from the source listing")
    zoning_archive = zoning_root / version_id / "source.zip"
    zoning_meta = _read_source_metadata(zoning_archive)
    boundary_meta = _read_source_metadata(boundary_archive)
    reference_archive = zoning_root / codebook_reference_version / "source.zip"
    reference_meta = _read_source_metadata(reference_archive)
    reference_codebook = load_codebook(reference_archive)
    if version_id == "20231024":
        codebook = reference_codebook
        codebook_borrowed = True
    else:
        codebook = load_codebook(zoning_archive)
        if (
            codebook.allowed_hierarchy != reference_codebook.allowed_hierarchy
            or codebook.broad_codes != reference_codebook.broad_codes
        ):
            raise ZoningPreparationError("Version codebook differs from the reviewed reference")
        codebook_borrowed = False
    destination = output_root / f"zoning_shares_{version_id}.csv"
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing zoning research output")
    admins, admin_names, admin_quality = load_admin_geometries(boundary_archive)
    zones, zone_quality = load_zone_geometries(zoning_archive, codebook)
    rows = area_shares(admins, zones, admin_names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = temp_meta = None
    linked_metadata = False
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=destination.parent, delete=False
        ) as stream:
            temp_csv = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        report = {
            "artifact_kind": "seoul_zoning_area_shares_research_v1",
            "version_id": version_id,
            "source_listing_modified_date": zoning_meta.get("source_listing_modified_date"),
            "region_code_system": CODE_SYSTEM,
            "area_crs": AREA_CRS,
            "overlay_grid_metres": OVERLAY_GRID_METRES,
            "source_zoning_crs": "EPSG:5174",
            "source_boundary_crs": "EPSG:5181",
            "zoning_archive_sha256": zoning_meta["archive_sha256"],
            "boundary_archive_sha256": boundary_meta["archive_sha256"],
            "archive_sha256": zoning_meta["archive_sha256"],
            "codebook_reference_archive_sha256": reference_meta["archive_sha256"],
            "codebook_workbook_sha256": codebook.workbook_sha256,
            "codebook_borrowed_from_2024": codebook_borrowed,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "row_count": len(rows),
            "quality_counts": dict(admin_quality + zone_quality),
            "max_unknown_share": max(row["unknown_share"] for row in rows),
            "max_uncovered_share": max(row["uncovered_share"] for row in rows),
            "max_overlap_ambiguity_share": max(row["overlap_ambiguity_share"] for row in rows),
            "max_abs_numerical_partition_residual_share": max(
                abs(row["numerical_partition_residual_share"]) for row in rows
            ),
            "area_weighted_unknown_share": sum(
                row["unknown_share"] * row["admin_area_m2"] for row in rows
            )
            / sum(row["admin_area_m2"] for row in rows),
            "area_weighted_overlap_ambiguity_share": sum(
                row["overlap_ambiguity_share"] * row["admin_area_m2"] for row in rows
            )
            / sum(row["admin_area_m2"] for row in rows),
            "unknown_above_5pct_count": sum(row["unknown_share"] > 0.05 for row in rows),
            "unknown_above_20pct_count": sum(row["unknown_share"] > 0.20 for row in rows),
            "overlap_above_5pct_count": sum(row["overlap_ambiguity_share"] > 0.05 for row in rows),
            "overlap_above_20pct_count": sum(row["overlap_ambiguity_share"] > 0.20 for row in rows),
            "review_required_count": sum(
                row["quality_status"] == "review_required" for row in rows
            ),
            "geometry_repair_policy": (
                "Keep class only for ring self-intersection repaired with <=1e-6 "
                "relative area and symmetric-difference change and independently valid "
                "code/name; otherwise unknown"
            ),
            "first_available_at": zoning_meta.get("first_available_at"),
            "availability_evidence": None,
            "legal_effective_date": zoning_meta.get("legal_effective_date"),
            "geometry_valid_at": None,
            "geometry_valid_from": None,
            "geometry_valid_through": None,
            "geometry_validity_evidence": None,
            "boundary_geometry_asof": boundary_meta.get("geometry_asof"),
            "boundary_first_available_at": boundary_meta.get("first_available_at"),
            "boundary_availability_evidence": None,
            "boundary_geometry_valid_from": None,
            "boundary_geometry_valid_through": None,
            "boundary_geometry_validity_evidence": None,
            "historical_validity_verified": False,
            "training_eligible": False,
            "limitations": [
                "Source listing modification date is not a verified first availability date.",
                "Zoning geometry and boundary validity dates are unverified.",
                "Ambiguous, unsafe-repair, and unmapped areas are assigned to unknown.",
                "Area shares are research candidates, not sales-model features.",
            ],
            "csv_sha256": sha256(temp_csv),
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            temp_meta = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_meta, metadata_path)
        linked_metadata = True
        os.link(temp_csv, destination)
    except BaseException:
        if linked_metadata:
            metadata_path.unlink(missing_ok=True)
        raise
    finally:
        if temp_csv is not None:
            temp_csv.unlink(missing_ok=True)
        if temp_meta is not None:
            temp_meta.unlink(missing_ok=True)
    return destination, report
