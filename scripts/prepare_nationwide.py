"""Prepare unjoined nationwide store and population snapshot aggregates."""

from __future__ import annotations

import argparse
from pathlib import Path

from pickplace_ai.nationwide_ingest import prepare_nationwide


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stores", nargs="+", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--store-reference-label", required=True)
    parser.add_argument("--population-reference-month", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = prepare_nationwide(
        stores=args.stores,
        population=args.population,
        output_dir=args.output_dir,
        store_reference_label=args.store_reference_label,
        population_reference_month=args.population_reference_month,
    )
    print(
        f"상가 원본 {summary['stores']['row_count']}행 → {summary['stores']['group_count']}개 집계"
    )
    print(f"인구 읍면동 {summary['population']['row_count']}행; 지역 코드 연계 미검증")


if __name__ == "__main__":
    main()
