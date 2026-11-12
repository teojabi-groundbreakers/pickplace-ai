"""Audit a narrowly pinned 2026 natural-green supersession hypothesis.

This is a counterfactual comparison. It never changes the official archive,
the strict zoning output, or eligibility for model training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from pickplace_ai.seoul_zoning_conflicts import SourceZone, load_source_zones
from pickplace_ai.seoul_zoning_research import (
    OVERLAY_GRID_METRES,
    SpatialFeature,
    ZoningPreparationError,
    _polygon_or_empty,
    _read_source_metadata,
    area_shares,
    load_admin_geometries,
    load_codebook,
    sha256,
)

PRIOR_VERSION = "20250805"
PRIOR_ARCHIVE_SHA256 = "b26a4c201c348071a00d5f06a152fffd3912cb45fd6ecbe750167aa5be97674d"
NEW_GREEN_PRESENT_SN = "11000UQ111PS202509030002"
NEW_GREEN_NOTICE_SN = "11000NTC202507300001"
NEW_GREEN_DECISION_SN = "11000ARZ202507300005"
NEW_GREEN_CODES = ("UQA400", "UQA430", "", "UQA430")
NEW_RESIDENTIAL_PRESENT_SN = "11000UQ111PS202509030001"
NEW_RESIDENTIAL_DECISION_SN = "11000ARZ202507300001"
NEW_RESIDENTIAL_CODES = ("UQA100", "UQA100", "UQA121", "UQA121")
LEGACY_CLASSES = frozenset(("residential", "commercial"))


@dataclass(frozen=True)
class PinnedSource:
    archive_sha256: str
    source_row_index: int
    geometry_sha256: str


PINNED_NEW_GREEN = {
    "20260304": PinnedSource(
        archive_sha256="cc7054dbb3c396526660e37233da57d898f5538f7203bc2dec935abb42208bf7",
        source_row_index=7088,
        geometry_sha256="a4298301dac7a4008121ef0639353ef4208c33f149569a9807def537398e2213",
    ),
    "20260911": PinnedSource(
        archive_sha256="e92cc4e8e3c154a57cdeb580c38bb103aee0f33961cd6e77a2a029fc02724cd3",
        source_row_index=7149,
        geometry_sha256="cb8e4f29fdbeb3ddf5420355159947272a08bbd0fcf082546eda4c473ec09158",
    ),
}
PINNED_NEW_RESIDENTIAL = {
    "20260304": PinnedSource(
        archive_sha256=PINNED_NEW_GREEN["20260304"].archive_sha256,
        source_row_index=7087,
        geometry_sha256="90ce880e4c7bfd9f239c0ca1e00ebb8b5f95e6585be5df8cd24ac353d09a855b",
    ),
    "20260911": PinnedSource(
        archive_sha256=PINNED_NEW_GREEN["20260911"].archive_sha256,
        source_row_index=7148,
        geometry_sha256="efb1028d17ef32a91e81120a81842839276ecec0a95c4baf777308338c1028db",
    ),
}

SUMMARY_FIELDS = (
    "region_code",
    "region_name",
    "admin_area_m2",
    "strict_residential_share",
    "scenario_residential_share",
    "strict_commercial_share",
    "scenario_commercial_share",
    "strict_green_share",
    "scenario_green_share",
    "strict_unknown_share",
    "scenario_unknown_share",
    "strict_overlap_ambiguity_share",
    "scenario_overlap_ambiguity_share",
    "strict_overlap_known_only_share",
    "scenario_overlap_known_only_share",
    "strict_overlap_unknown_involved_share",
    "scenario_overlap_unknown_involved_share",
    "strict_quality_status",
    "scenario_quality_status",
    "training_eligible",
)


def _geometry_sha256(zone: SourceZone) -> str:
    return hashlib.sha256(zone.geometry.wkb).hexdigest()


def _legacy_identity(zone: SourceZone) -> tuple[str, ...]:
    """Match the same prior record and geometry, never PRESENT_SN alone."""
    record = zone.record
    return (
        zone.category,
        record["PRESENT_SN"],
        record["NTFC_SN"],
        record["WTNNC_SN"],
        record["DGM_NM"],
        record["LCLAS_CL"],
        record["MLSFC_CL"],
        record["SCLAS_CL"],
        record["ATRB_SE"],
        _geometry_sha256(zone),
    )


def _require_pinned_green(zones: list[SourceZone], pin: PinnedSource) -> SourceZone:
    candidates = [zone for zone in zones if zone.source_row_index == pin.source_row_index]
    if len(candidates) != 1:
        raise ZoningPreparationError("Pinned natural-green source row is missing or duplicated")
    zone = candidates[0]
    record = zone.record
    if (
        zone.category != "green"
        or zone.reason is not None
        or _geometry_sha256(zone) != pin.geometry_sha256
        or record["PRESENT_SN"] != NEW_GREEN_PRESENT_SN
        or record["NTFC_SN"] != NEW_GREEN_NOTICE_SN
        or record["WTNNC_SN"] != NEW_GREEN_DECISION_SN
        or record["DGM_NM"] != "자연녹지지역"
        or tuple(record[field] for field in ("LCLAS_CL", "MLSFC_CL", "SCLAS_CL", "ATRB_SE"))
        != NEW_GREEN_CODES
    ):
        raise ZoningPreparationError("Pinned natural-green source record changed")
    return zone


def _require_pinned_new_residential(zones: list[SourceZone], pin: PinnedSource) -> SourceZone:
    candidates = [zone for zone in zones if zone.source_row_index == pin.source_row_index]
    if len(candidates) != 1:
        raise ZoningPreparationError("Pinned new-residential source row is missing or duplicated")
    zone = candidates[0]
    record = zone.record
    if (
        zone.category != "unknown"
        or zone.reason != "code_hierarchy_mismatch"
        or _geometry_sha256(zone) != pin.geometry_sha256
        or record["PRESENT_SN"] != NEW_RESIDENTIAL_PRESENT_SN
        or record["NTFC_SN"] != NEW_GREEN_NOTICE_SN
        or record["WTNNC_SN"] != NEW_RESIDENTIAL_DECISION_SN
        or record["DGM_NM"] != "제1종일반주거지역"
        or tuple(record[field] for field in ("LCLAS_CL", "MLSFC_CL", "SCLAS_CL", "ATRB_SE"))
        != NEW_RESIDENTIAL_CODES
    ):
        raise ZoningPreparationError("Pinned new-residential source record changed")
    return zone


def simulate_pinned_supersession(
    prior_zones: list[SourceZone], current_zones: list[SourceZone], pin: PinnedSource
) -> tuple[list[SourceZone], dict]:
    """Clip only exact prior residential/commercial rows under the pinned green.

    The new unknown residential row and all unverified current rows stay in
    place. This leaves the new-new conflict visible for separate review.
    """
    import shapely

    new_green = _require_pinned_green(current_zones, pin)
    prior_identities = {
        _legacy_identity(zone) for zone in prior_zones if zone.category in LEGACY_CLASSES
    }
    simulated = []
    removed_parts = []
    all_known_overlap_parts = []
    clipped_keys = []
    for zone in current_zones:
        if zone.category in LEGACY_CLASSES and zone.geometry.intersects(new_green.geometry):
            known_overlap = _polygon_or_empty(
                shapely.intersection(
                    zone.geometry, new_green.geometry, grid_size=OVERLAY_GRID_METRES
                )
            )
            if not known_overlap.is_empty and known_overlap.area > 0:
                all_known_overlap_parts.append(known_overlap)
        if zone.category not in LEGACY_CLASSES or _legacy_identity(zone) not in prior_identities:
            simulated.append(zone)
            continue
        if not zone.geometry.intersects(new_green.geometry):
            simulated.append(zone)
            continue
        removed = _polygon_or_empty(
            shapely.intersection(zone.geometry, new_green.geometry, grid_size=OVERLAY_GRID_METRES)
        )
        if removed.is_empty or removed.area <= 0:
            simulated.append(zone)
            continue
        remaining = _polygon_or_empty(
            shapely.difference(zone.geometry, new_green.geometry, grid_size=OVERLAY_GRID_METRES)
        )
        removed_parts.append(removed)
        clipped_keys.append(zone.feature_key)
        if not remaining.is_empty and remaining.area > 0:
            simulated.append(replace(zone, geometry=remaining))
    union_removed = _polygon_or_empty(
        shapely.union_all(removed_parts, grid_size=OVERLAY_GRID_METRES)
    )
    all_known_overlap = _polygon_or_empty(
        shapely.union_all(all_known_overlap_parts, grid_size=OVERLAY_GRID_METRES)
    )
    unresolved_known_overlap = _polygon_or_empty(
        shapely.difference(all_known_overlap, union_removed, grid_size=OVERLAY_GRID_METRES)
    )
    return simulated, {
        "new_green_feature_key": new_green.feature_key,
        "new_green_area_m2": round(new_green.geometry.area, 3),
        "exact_prior_rows_clipped": len(clipped_keys),
        "clipped_feature_keys": clipped_keys,
        "clipped_area_union_m2": round(union_removed.area, 3),
        "new_green_known_overlap_before_m2": round(all_known_overlap.area, 3),
        "new_green_known_overlap_unresolved_m2": round(unresolved_known_overlap.area, 3),
        "pair_areas_are_not_additive": True,
    }


def _as_spatial(zones: list[SourceZone]) -> list[SpatialFeature]:
    return [SpatialFeature(zone.category, zone.geometry, zone.reason) for zone in zones]


def _weighted(rows: list[dict], key: str) -> float:
    total_area = sum(row["admin_area_m2"] for row in rows)
    return sum(row[key] * row["admin_area_m2"] for row in rows) / total_area


def _comparison_row(strict: dict, scenario: dict) -> dict:
    fields = (
        "residential_share",
        "commercial_share",
        "green_share",
        "unknown_share",
        "overlap_ambiguity_share",
        "overlap_known_only_share",
        "overlap_unknown_involved_share",
        "quality_status",
    )
    return {
        "region_code": strict["region_code"],
        "region_name": strict["region_name"],
        "admin_area_m2": strict["admin_area_m2"],
        **{f"strict_{field}": strict[field] for field in fields},
        **{f"scenario_{field}": scenario[field] for field in fields},
        "training_eligible": False,
    }


def compare_pinned_supersession(
    version_id: str,
    zoning_root: Path,
    boundary_archive: Path,
    destination: Path,
) -> dict:
    """Produce a side-by-side CSV/manifest; reject changed or missing inputs."""
    if version_id not in PINNED_NEW_GREEN:
        raise ZoningPreparationError("Only the pinned 2026 source versions are supported")
    pin = PINNED_NEW_GREEN[version_id]
    current_archive = zoning_root / version_id / "source.zip"
    prior_archive = zoning_root / PRIOR_VERSION / "source.zip"
    current_meta = _read_source_metadata(current_archive)
    prior_meta = _read_source_metadata(prior_archive)
    boundary_meta = _read_source_metadata(boundary_archive)
    if current_meta["archive_sha256"] != pin.archive_sha256:
        raise ZoningPreparationError("Current zoning archive differs from pinned source")
    if prior_meta["archive_sha256"] != PRIOR_ARCHIVE_SHA256:
        raise ZoningPreparationError("Prior zoning archive differs from pinned source")
    prior_book = load_codebook(prior_archive)
    current_book = load_codebook(current_archive)
    if (
        prior_book.allowed_hierarchy != current_book.allowed_hierarchy
        or prior_book.broad_codes != current_book.broad_codes
    ):
        raise ZoningPreparationError("Official zoning codebooks differ")
    prior_zones = load_source_zones(prior_archive, prior_book, PRIOR_VERSION)
    current_zones = load_source_zones(current_archive, current_book, version_id)
    new_residential = _require_pinned_new_residential(
        current_zones, PINNED_NEW_RESIDENTIAL[version_id]
    )
    scenario_zones, audit = simulate_pinned_supersession(prior_zones, current_zones, pin)
    import shapely

    new_green = _require_pinned_green(current_zones, pin)
    audit["new_residential_green_overlap_before_m2"] = round(
        shapely.intersection(
            new_residential.geometry, new_green.geometry, grid_size=OVERLAY_GRID_METRES
        ).area,
        3,
    )
    scenario_new_residential = _require_pinned_new_residential(
        scenario_zones, PINNED_NEW_RESIDENTIAL[version_id]
    )
    scenario_new_green = _require_pinned_green(scenario_zones, pin)
    audit["new_residential_green_overlap_after_m2"] = round(
        shapely.intersection(
            scenario_new_residential.geometry,
            scenario_new_green.geometry,
            grid_size=OVERLAY_GRID_METRES,
        ).area,
        3,
    )
    if (
        audit["new_residential_green_overlap_before_m2"]
        != audit["new_residential_green_overlap_after_m2"]
    ):
        raise ZoningPreparationError("The unresolved new-new overlap was unexpectedly changed")
    admins, names, _ = load_admin_geometries(boundary_archive)
    strict_rows = area_shares(admins, _as_spatial(current_zones), names)
    scenario_rows = area_shares(admins, _as_spatial(scenario_zones), names)
    if len(strict_rows) != len(scenario_rows):
        raise ZoningPreparationError("Strict and scenario region counts differ")
    compared = []
    for strict, scenario in zip(strict_rows, scenario_rows, strict=True):
        if strict["region_code"] != scenario["region_code"]:
            raise ZoningPreparationError("Strict and scenario region codes differ")
        compared.append(_comparison_row(strict, scenario))
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite a zoning supersession comparison")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_csv = temp_meta = None
    csv_linked = False
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=destination.parent, delete=False
        ) as stream:
            temp_csv = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(compared)
            stream.flush()
            os.fsync(stream.fileno())
        report = {
            "artifact_kind": "seoul_zoning_pinned_supersession_counterfactual_v1",
            "version_id": version_id,
            "prior_version_id": PRIOR_VERSION,
            "current_archive_sha256": pin.archive_sha256,
            "prior_archive_sha256": PRIOR_ARCHIVE_SHA256,
            "boundary_archive_sha256": boundary_meta["archive_sha256"],
            "pinned_source_row_index": pin.source_row_index,
            "pinned_geometry_sha256": pin.geometry_sha256,
            "source_notice_code": NEW_GREEN_NOTICE_SN,
            "context_sources": [
                "https://data.seoul.go.kr/dataList/OA-21136/S/1/datasetView.do",
                "https://eum.go.kr/web/gs/gv/gvGosiDet.jsp?seq=614617",
                "https://mas.smc.seoul.kr/attach/record/SEOUL/appendix/a11/A0069176.pdf?time=20260419160027",
            ],
            "policy": (
                "Only current residential/commercial rows that exactly match a 2025-08 "
                "record and projected geometry are clipped beneath the pinned new green row."
            ),
            "audit": audit,
            "region_count": len(compared),
            "strict_review_required_count": sum(
                row["quality_status"] == "review_required" for row in strict_rows
            ),
            "scenario_review_required_count": sum(
                row["quality_status"] == "review_required" for row in scenario_rows
            ),
            "strict_area_weighted_overlap_share": _weighted(strict_rows, "overlap_ambiguity_share"),
            "scenario_area_weighted_overlap_share": _weighted(
                scenario_rows, "overlap_ambiguity_share"
            ),
            "strict_area_weighted_known_only_overlap_share": _weighted(
                strict_rows, "overlap_known_only_share"
            ),
            "scenario_area_weighted_known_only_overlap_share": _weighted(
                scenario_rows, "overlap_known_only_share"
            ),
            "training_eligible": False,
            "scenario_approved_for_training": False,
            "first_available_at": None,
            "historical_validity_verified": False,
            "limitations": [
                "The notice number to source-record mapping remains unverified.",
                "Only exact prior geometry matches are clipped; other overlaps remain unresolved.",
                "The new residential and new-green mutual overlap is retained.",
                "This diagnostic cannot prove legal supersession or historical availability.",
            ],
            "csv_sha256": sha256(temp_csv),
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            temp_meta = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_csv, destination)
        csv_linked = True
        os.link(temp_meta, metadata_path)
    except Exception:
        if csv_linked:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if temp_csv is not None:
            temp_csv.unlink(missing_ok=True)
        if temp_meta is not None:
            temp_meta.unlink(missing_ok=True)
    return report
