"""Trace ambiguous Seoul zoning areas to original source polygon pairs.

Example:
  PYTHONPATH=src:. python scripts/diagnose_seoul_zoning_conflicts.py \
    --version 20260911 --region-code 11620695 --region-code 11410620 \
    --zoning-root /path/to/source_versions \
    --boundary-archive /path/to/boundary/source.zip
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pickplace_ai.seoul_zoning_conflicts import diagnose_version

REPO = Path(__file__).resolve().parents[1]
ZONING_ROOT = REPO / "data/raw/seoul/zoning/urban_area/source_versions"
BOUNDARY_ARCHIVE = REPO / "data/raw/seoul/boundaries/official_polygon/imported_20261112/source.zip"
OUTPUT_ROOT = REPO / "data/processed/seoul/zoning_conflict_diagnostics"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Source ZIP listing version YYYYMMDD")
    parser.add_argument(
        "--region-code",
        action="append",
        required=True,
        dest="region_codes",
        help="8-digit Seoul administrative-dong code; repeat to inspect multiple regions",
    )
    parser.add_argument("--zoning-root", type=Path, default=ZONING_ROOT)
    parser.add_argument("--boundary-archive", type=Path, default=BOUNDARY_ARCHIVE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--min-pair-area-m2", type=float, default=0.0)
    args = parser.parse_args()
    stem = "_".join((args.version, *sorted(args.region_codes)))
    destination = args.output_root / f"zoning_conflicts_{stem}.csv"
    metadata = diagnose_version(
        args.version,
        args.zoning_root,
        args.boundary_archive,
        destination,
        args.region_codes,
        min_pair_area_m2=args.min_pair_area_m2,
    )
    print(
        json.dumps(
            {
                "csv": str(destination),
                "pair_row_count": metadata["pair_row_count"],
                "region_summaries": metadata["region_summaries"],
                "training_eligible": metadata["training_eligible"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
