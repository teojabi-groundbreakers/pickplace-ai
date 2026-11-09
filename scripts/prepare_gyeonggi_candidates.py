"""Separate review-only Gyeonggi monthly sales candidates and excluded source rows."""

import argparse
import json
from pathlib import Path

from pickplace_ai.gyeonggi_candidates import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--industries", required=True, type=Path)
    parser.add_argument("--current-regions", required=True, type=Path)
    parser.add_argument("--region-reference-month", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = prepare(
        source=args.source,
        industries=args.industries,
        current_regions=args.current_regions,
        region_reference_month=args.region_reference_month,
        output_dir=args.output_dir,
    )
    print(json.dumps({"rows": summary["rows"], "counts": summary["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
