"""Create descriptive zoning area shares from the imported official Seoul ZIPs.

Outputs are deliberately ineligible for model training until independent
publication and historical geometry validity evidence is reviewed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pickplace_ai.seoul_zoning_research import prepare_version

REPO = Path(__file__).resolve().parents[1]
ZONING_ROOT = REPO / "data/raw/seoul/zoning/urban_area/source_versions"
BOUNDARY_ARCHIVE = REPO / "data/raw/seoul/boundaries/official_polygon/imported_20261112/source.zip"
OUTPUT_ROOT = REPO / "data/processed/seoul/zoning_area_shares"
VERSIONS = ("20231024", "20241107", "20250320", "20250805", "20260304", "20260911")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        choices=(*VERSIONS, "all"),
        default="all",
        help="Official source ZIP version to prepare; default: all six",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    versions = VERSIONS if args.version == "all" else (args.version,)
    for version in versions:
        output, manifest = prepare_version(
            version,
            ZONING_ROOT,
            BOUNDARY_ARCHIVE,
            args.output_root,
        )
        print(
            json.dumps(
                {
                    "csv": str(output),
                    "version_id": version,
                    "row_count": manifest["row_count"],
                    "training_eligible": manifest["training_eligible"],
                    "max_unknown_share": manifest["max_unknown_share"],
                    "max_overlap_ambiguity_share": manifest["max_overlap_ambiguity_share"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
