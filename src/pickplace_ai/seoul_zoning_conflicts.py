"""Trace cross-category polygon overlaps to individual source ZIP records.

This is a diagnostic aid. Pair intersections are not additive when three or
more source polygons cover the same place. Outputs never grant training use.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pickplace_ai.seoul_zoning_research import (
    AREA_CRS,
    OVERLAY_GRID_METRES,
    SpatialFeature,
    ZoningPreparationError,
    _polygon_or_empty,
    _read_shapefile,
    _read_source_metadata,
    _transform_polygon,
    area_shares,
    classify_record,
    load_admin_geometries,
    load_codebook,
    repeated_midcode_review_candidate,
    sha256,
)

CODE_FIELDS = ("LCLAS_CL", "MLSFC_CL", "SCLAS_CL", "ATRB_SE")
IDENTIFIER_FIELDS = ("PRESENT_SN", "NTFC_SN", "WTNNC_SN")
SOURCE_FIELDS = (*IDENTIFIER_FIELDS, "DGM_NM", *CODE_FIELDS)
PAIR_FIELDS = (
    "version_id",
    "region_code",
    "region_name",
    "admin_area_m2",
    "pair_type",
    "pair_intersection_m2",
    "pair_intersection_share",
    *(
        f"left_{name}"
        for name in (
            "source_row_index",
            "feature_key",
            *SOURCE_FIELDS,
            "category",
            "reason",
            "review_candidate_class",
        )
    ),
    *(
        f"right_{name}"
        for name in (
            "source_row_index",
            "feature_key",
            *SOURCE_FIELDS,
            "category",
            "reason",
            "review_candidate_class",
        )
    ),
)


@dataclass(frozen=True)
class SourceZone:
    """A projected source polygon with a row-scoped identity and DBF lineage."""

    source_row_index: int
    feature_key: str
    record: dict[str, str]
    category: str
    reason: str | None
    geometry: object
    review_candidate_class: str | None = None


def _str_field(record: dict, field: str) -> str:
    return str(record.get(field) or "").strip()


def load_source_zones(archive_path: Path, codebook, version_id: str) -> list[SourceZone]:
    """Use the same classification and geometry repair policy as area shares."""
    import shapefile
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:5174", AREA_CRS, always_xy=True)
    zones = []
    with _read_shapefile(archive_path, 5174, "cp949") as reader:
        fields = {field[0] for field in reader.fields[1:]}
        if not set(SOURCE_FIELDS) <= fields:
            raise ZoningPreparationError("Zoning class or source identifier fields are missing")
        if reader.numRecords != reader.numShapes:
            raise ZoningPreparationError("Zoning SHP/DBF row counts differ")
        for row_index in range(reader.numRecords):
            record = reader.record(row_index).as_dict()
            category, reason = classify_record(record, codebook)
            try:
                source_shape = reader.shape(row_index)
            except shapefile.ShapefileException as exc:
                if "requires non-empty points" not in str(exc):
                    raise
                continue
            if source_shape.shapeType == shapefile.NULL:
                continue
            geometry, repaired, repair_preserved_area = _transform_polygon(
                source_shape, transformer
            )
            if geometry is None:
                continue
            if repaired and not repair_preserved_area:
                category, reason = "unknown", "unsafe_geometry_repair"
            geometry_hash = hashlib.sha256(geometry.wkb).hexdigest()
            zones.append(
                SourceZone(
                    source_row_index=row_index,
                    feature_key=f"{version_id}:{row_index}:{geometry_hash}",
                    record={field: _str_field(record, field) for field in SOURCE_FIELDS},
                    category=category,
                    reason=reason,
                    geometry=geometry,
                    review_candidate_class=(
                        repeated_midcode_review_candidate(record, codebook)
                        if reason == "code_hierarchy_mismatch"
                        else None
                    ),
                )
            )
    if not zones:
        raise ZoningPreparationError("Zoning ZIP has no usable polygon geometry")
    return zones


def _side_fields(prefix: str, zone: SourceZone) -> dict[str, object]:
    return {
        f"{prefix}_source_row_index": zone.source_row_index,
        f"{prefix}_feature_key": zone.feature_key,
        **{f"{prefix}_{field}": zone.record.get(field, "") for field in SOURCE_FIELDS},
        f"{prefix}_category": zone.category,
        f"{prefix}_reason": zone.reason or "",
        f"{prefix}_review_candidate_class": zone.review_candidate_class or "",
    }


def diagnose_region(
    version_id: str,
    region_code: str,
    region_name: str,
    boundary,
    zones: list[SourceZone],
    *,
    min_pair_area_m2: float = 0.0,
) -> tuple[list[dict], dict]:
    """Return source pairs plus the unioned ambiguity share for one admin dong."""
    import shapely
    from shapely.strtree import STRtree

    if boundary.is_empty or boundary.area <= 0:
        raise ZoningPreparationError("Administrative polygon has zero area")
    if not math.isfinite(min_pair_area_m2) or min_pair_area_m2 < 0:
        raise ValueError("min_pair_area_m2 must be finite and nonnegative")
    geometries = [zone.geometry for zone in zones]
    tree = STRtree(geometries)
    candidate_indexes = sorted(int(i) for i in tree.query(boundary, predicate="intersects"))
    clipped = {}
    for index in candidate_indexes:
        piece = _polygon_or_empty(
            shapely.intersection(boundary, zones[index].geometry, grid_size=OVERLAY_GRID_METRES)
        )
        if not piece.is_empty and piece.area > 0:
            clipped[index] = piece
    pairs = []
    overlaps = []
    indexes = sorted(clipped)
    for position, left_index in enumerate(indexes):
        left = zones[left_index]
        for right_index in indexes[position + 1 :]:
            right = zones[right_index]
            if left.category == right.category:
                continue
            intersection = _polygon_or_empty(
                shapely.intersection(
                    clipped[left_index], clipped[right_index], grid_size=OVERLAY_GRID_METRES
                )
            )
            if intersection.is_empty or intersection.area <= 0:
                continue
            overlaps.append(intersection)
            if intersection.area < min_pair_area_m2:
                continue
            pairs.append(
                {
                    "version_id": version_id,
                    "region_code": region_code,
                    "region_name": region_name,
                    "admin_area_m2": round(boundary.area, 3),
                    "pair_type": (
                        "unknown_involved"
                        if "unknown" in (left.category, right.category)
                        else "cross_known_class"
                    ),
                    "pair_intersection_m2": round(intersection.area, 3),
                    "pair_intersection_share": round(intersection.area / boundary.area, 10),
                    **_side_fields("left", left),
                    **_side_fields("right", right),
                }
            )
    ambiguous_area = (
        shapely.union_all(overlaps, grid_size=OVERLAY_GRID_METRES).area if overlaps else 0.0
    )
    summary = area_shares(
        {region_code: boundary},
        [
            SpatialFeature(zones[index].category, clipped[index], zones[index].reason)
            for index in indexes
        ],
        {region_code: region_name},
    )[0]
    measured_share = ambiguous_area / boundary.area
    if abs(measured_share - summary["overlap_ambiguity_share"]) > 0.00001:
        raise ZoningPreparationError("Diagnostic pairs disagree with area-share ambiguity")
    return pairs, {
        "region_code": region_code,
        "region_name": region_name,
        "admin_area_m2": round(boundary.area, 3),
        "pair_count": len(pairs),
        "overlap_ambiguity_share": summary["overlap_ambiguity_share"],
        "unknown_share": summary["unknown_share"],
        "quality_status": summary["quality_status"],
        "quality_reasons": summary["quality_reasons"],
        "training_eligible": False,
    }


def diagnose_version(
    version_id: str,
    zoning_root: Path,
    boundary_archive: Path,
    output_path: Path,
    region_codes: list[str],
    *,
    codebook_reference_version: str = "20241107",
    min_pair_area_m2: float = 0.0,
) -> dict:
    """Write pair CSV and linked metadata without changing source archives."""
    if not re.fullmatch(r"20\d{6}", version_id):
        raise ZoningPreparationError("Version ID must be YYYYMMDD")
    if not region_codes or any(not re.fullmatch(r"\d{8}", code) for code in region_codes):
        raise ZoningPreparationError("Pass one or more 8-digit administrative region codes")
    if len(set(region_codes)) != len(region_codes):
        raise ZoningPreparationError("Duplicate administrative region code")
    zoning_archive = zoning_root / version_id / "source.zip"
    source_meta = _read_source_metadata(zoning_archive)
    boundary_meta = _read_source_metadata(boundary_archive)
    reference_archive = zoning_root / codebook_reference_version / "source.zip"
    _read_source_metadata(reference_archive)
    reference_codebook = load_codebook(reference_archive)
    codebook = reference_codebook if version_id == "20231024" else load_codebook(zoning_archive)
    if (
        codebook.allowed_hierarchy != reference_codebook.allowed_hierarchy
        or codebook.broad_codes != reference_codebook.broad_codes
    ):
        raise ZoningPreparationError("Version codebook differs from the reviewed reference")
    admins, names, _ = load_admin_geometries(boundary_archive)
    missing = sorted(set(region_codes) - admins.keys())
    if missing:
        raise ZoningPreparationError(f"Unknown administrative region codes: {missing}")
    zones = load_source_zones(zoning_archive, codebook, version_id)
    rows = []
    summaries = []
    for code in sorted(region_codes):
        pairs, summary = diagnose_region(
            version_id,
            code,
            names[code],
            admins[code],
            zones,
            min_pair_area_m2=min_pair_area_m2,
        )
        rows.extend(pairs)
        summaries.append(summary)
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing zoning diagnostic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = temp_meta = None
    csv_linked = False
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=output_path.parent, delete=False
        ) as stream:
            temp_csv = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = {
            "artifact_kind": "seoul_zoning_source_pair_diagnostic_v1",
            "version_id": version_id,
            "zoning_archive_sha256": source_meta["archive_sha256"],
            "boundary_archive_sha256": boundary_meta["archive_sha256"],
            "area_crs": AREA_CRS,
            "overlay_grid_metres": OVERLAY_GRID_METRES,
            "region_summaries": summaries,
            "pair_row_count": len(rows),
            "min_pair_area_m2": min_pair_area_m2,
            "pair_intersection_shares_additive": False,
            "pair_area_warning": (
                "Pair intersections can cover the same ground more than once; "
                "do not sum pair_intersection_share. Use unioned overlap_ambiguity_share."
            ),
            "training_eligible": False,
            "csv_sha256": sha256(temp_csv),
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output_path.parent, delete=False
        ) as stream:
            temp_meta = Path(stream.name)
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_csv, output_path)
        csv_linked = True
        os.link(temp_meta, metadata_path)
    except Exception:
        if csv_linked:
            output_path.unlink(missing_ok=True)
        raise
    finally:
        if temp_csv is not None:
            temp_csv.unlink(missing_ok=True)
        if temp_meta is not None:
            temp_meta.unlink(missing_ok=True)
    return metadata
