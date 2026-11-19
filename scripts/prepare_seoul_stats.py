"""Prepare Seoul quarterly raw-source descriptive stats as SQLite and manifest."""

import argparse
import json
import sys
from pathlib import Path

from pickplace_ai.ingest import ImportDataError
from pickplace_ai.seoul_stats_ingest import prepare_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", type=Path, nargs="+", required=True)
    parser.add_argument("--stores", type=Path, nargs="+", required=True)
    parser.add_argument("--resident", type=Path, required=True)
    parser.add_argument("--floating", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        metadata = prepare_snapshot(
            args.sales, args.stores, args.resident, args.floating, args.output_dir
        )
    except (ImportDataError, OSError) as exc:
        print(f"Seoul snapshot failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({name: item["row_count"] for name, item in metadata["datasets"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
