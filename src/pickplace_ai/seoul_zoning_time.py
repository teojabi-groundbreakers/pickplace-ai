"""Fail-closed temporal selection for Seoul zoning research inputs.

This only checks whether independently verified source versions were known by
the end of a target's base quarter. It does not certify geometry quality,
administrative-code mapping, legal effect, or model training readiness.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import PurePosixPath
from urllib.parse import parse_qsl, urlsplit

_PERIOD = re.compile(r"(\d{4})Q([1-4])\Z")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def base_quarter_cutoff(target_period: str) -> tuple[str, date]:
    """Return the preceding quarter and its final calendar day."""
    match = _PERIOD.fullmatch(target_period) if isinstance(target_period, str) else None
    if match is None:
        raise ValueError("target_period must be YYYYQ1 through YYYYQ4")
    year, quarter = map(int, match.groups())
    if year < 2:
        raise ValueError("target_period has no representable base quarter")
    if quarter == 1:
        year -= 1
        quarter = 4
    else:
        quarter -= 1
    month = quarter * 3
    return f"{year:04d}Q{quarter}", date(year, month, calendar.monthrange(year, month)[1])


def _verified_date(value: object) -> date | None:
    if not isinstance(value, str) or _DATE.fullmatch(value) is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _evidence_reference(value: object) -> bool:
    """Accept a traceable HTTPS source or a repository-local evidence document."""
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if value.startswith("https://"):
        parsed = urlsplit(value)
        sensitive = {"key", "apikey", "servicekey", "token", "access_token", "secret"}
        return (
            bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and not any(name.lower() in sensitive for name, _ in parse_qsl(parsed.query))
        )
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and len(path.parts) >= 2
        and path.parts[0] == "docs"
        and ".." not in path.parts
        and path.suffix in {".md", ".json"}
    )


def _check_version(kind: str, version: Mapping[str, object], cutoff: date) -> str | None:
    version_id = version.get("version_id")
    digest = version.get("archive_sha256")
    if (
        not isinstance(version_id, str)
        or not version_id
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
    ):
        return "metadata_invalid"
    if (
        version.get("historical_validity_verified") is not True
        or version.get("training_eligible") is not True
    ):
        return "provenance_unverified"
    available = _verified_date(version.get("first_available_at"))
    geometry = _verified_date(version.get("geometry_valid_at"))
    valid_from = _verified_date(version.get("geometry_valid_from"))
    valid_through = _verified_date(version.get("geometry_valid_through"))
    # A known future date blocks the source even if some other date is absent.
    if available is not None and available > cutoff:
        return "not_available_by_cutoff"
    if (geometry is not None and geometry > cutoff) or (
        valid_from is not None and valid_from > cutoff
    ):
        return "geometry_after_cutoff"
    if available is None:
        return "availability_date_unverified"
    if kind == "boundary" or valid_from is not None or valid_through is not None:
        if valid_from is None or valid_through is None or valid_from > valid_through:
            return "validity_interval_unverified"
        if valid_through < cutoff:
            return "geometry_not_valid_at_cutoff"
    elif geometry is None:
        return "geometry_date_unverified"
    if not _evidence_reference(version.get("availability_evidence")):
        return "availability_evidence_missing"
    if not _evidence_reference(version.get("geometry_validity_evidence")):
        return "geometry_evidence_missing"
    return None


def _geometry_date(kind: str, version: Mapping[str, object]) -> date:
    field = "geometry_valid_from" if kind == "boundary" else "geometry_valid_at"
    if kind == "zoning" and version.get("geometry_valid_from") is not None:
        field = "geometry_valid_from"
    value = _verified_date(version[field])
    if value is None:
        raise ValueError("Verified geometry date disappeared")
    return value


def _choose(
    kind: str, versions: Sequence[Mapping[str, object]], cutoff: date
) -> tuple[dict[str, object] | None, str | None, list[dict[str, str]]]:
    if not isinstance(versions, (list, tuple)):
        raise TypeError(f"{kind} versions must be a list or tuple")
    if not versions:
        return None, f"{kind}_source_missing", []
    ids: set[str] = set()
    eligible: list[Mapping[str, object]] = []
    rejected: list[dict[str, str]] = []
    for version in versions:
        if not isinstance(version, Mapping):
            rejected.append({"version_id": "", "reason": "metadata_invalid"})
            continue
        version_id = version.get("version_id")
        if isinstance(version_id, str) and version_id:
            if version_id in ids:
                return None, f"{kind}_duplicate_version_id", rejected
            ids.add(version_id)
        reason = _check_version(kind, version, cutoff)
        if reason is None:
            eligible.append(version)
        else:
            rejected.append(
                {
                    "version_id": version_id if isinstance(version_id, str) else "",
                    "reason": reason,
                }
            )
    if not eligible:
        reasons = {item["reason"] for item in rejected}
        if len(reasons) == 1:
            return None, f"{kind}_{next(iter(reasons))}", rejected
        return None, f"{kind}_no_verified_snapshot_by_cutoff", rejected
    newest = max(
        eligible,
        key=lambda version: (
            _geometry_date(kind, version),
            _verified_date(version["first_available_at"]),
        ),
    )
    tied = [
        version
        for version in eligible
        if (
            _geometry_date(kind, version),
            version["first_available_at"],
        )
        == (_geometry_date(kind, newest), newest["first_available_at"])
    ]
    if len(tied) != 1:
        return None, f"{kind}_ambiguous_latest_version", rejected
    selected = {
        "version_id": newest["version_id"],
        "archive_sha256": newest["archive_sha256"],
        "first_available_at": newest["first_available_at"],
        "geometry_valid_at": newest.get("geometry_valid_at"),
        "geometry_valid_from": newest.get("geometry_valid_from"),
        "geometry_valid_through": newest.get("geometry_valid_through"),
        "availability_evidence": newest["availability_evidence"],
        "geometry_validity_evidence": newest["geometry_validity_evidence"],
    }
    return selected, None, rejected


def select_temporal_sources(
    target_period: str,
    zoning_versions: Sequence[Mapping[str, object]],
    boundary_versions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Select the last verified zoning and boundary snapshots known at base cutoff.

    Zoning needs verified `first_available_at` and an as-of geometry date or
    validity interval. A boundary needs a verified validity interval covering
    the entire base-quarter cutoff date. A listed file time, ZIP time, or DBF
    `CREATE_DAT` must never be substituted for either value. Source manifests
    must explicitly set both verification flags and provide evidence references
    before this gate passes. References are traceability, not automatic proof.
    """
    base_period, cutoff = base_quarter_cutoff(target_period)
    zoning, zoning_reason, zoning_rejected = _choose("zoning", zoning_versions, cutoff)
    boundary, boundary_reason, boundary_rejected = _choose("boundary", boundary_versions, cutoff)
    passed = zoning is not None and boundary is not None
    return {
        "temporal_status": "passed" if passed else "blocked",
        "reason": None if passed else zoning_reason or boundary_reason,
        "target_period": target_period,
        "base_period": base_period,
        "base_cutoff": cutoff.isoformat(),
        "zoning": zoning if passed else None,
        "boundary": boundary if passed else None,
        "rejected_versions": {"zoning": zoning_rejected, "boundary": boundary_rejected},
    }
