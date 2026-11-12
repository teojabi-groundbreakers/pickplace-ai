"""Compare strict 2026 zoning with a pinned, unapproved green-supersession scenario.

The scenario preserves the new residential unknown row and all unresolved
overlaps. It never produces a training-eligible feature table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pickplace_ai.seoul_zoning_supersession import PINNED_NEW_GREEN, compare_pinned_supersession

REPO = Path(__file__).resolve().parents[1]
ZONING_ROOT = REPO / "data/raw/seoul/zoning/urban_area/source_versions"
BOUNDARY_ARCHIVE = REPO / "data/raw/seoul/boundaries/official_polygon/imported_20261112/source.zip"
OUTPUT_ROOT = REPO / "data/processed/seoul/zoning_supersession_scenario"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=tuple(PINNED_NEW_GREEN), required=True)
    parser.add_argument("--zoning-root", type=Path, default=ZONING_ROOT)
    parser.add_argument("--boundary-archive", type=Path, default=BOUNDARY_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    destination = args.output_root / f"zoning_supersession_{args.version}.csv"
    report = compare_pinned_supersession(
        args.version, args.zoning_root, args.boundary_archive, destination
    )
    print(
        json.dumps(
            {
                "csv": str(destination),
                "version_id": report["version_id"],
                "exact_prior_rows_clipped": report["audit"]["exact_prior_rows_clipped"],
                "clipped_area_union_m2": report["audit"]["clipped_area_union_m2"],
                "strict_review_required_count": report["strict_review_required_count"],
                "scenario_review_required_count": report["scenario_review_required_count"],
                "strict_area_weighted_overlap_share": report["strict_area_weighted_overlap_share"],
                "scenario_area_weighted_overlap_share": report[
                    "scenario_area_weighted_overlap_share"
                ],
                "training_eligible": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
