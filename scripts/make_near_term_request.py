"""Prepare a local five-quarter request for the 2026Q2 research model.

This writes selected public observations into an ignored local file. It does
not send the request or certify that an upstream dataset is authentic.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from pickplace_ai.near_term_api import NearTermRequest
from scripts.make_research_request import make_request


def make_near_term_request(input_path, region_code, industry_code, target_period, as_of):
    if target_period not in {"2026Q3", "2026Q4"}:
        raise ValueError("Target period must be 2026Q3 or 2026Q4")
    payload = make_request(input_path, region_code, industry_code, "2026Q2", as_of, "selected")
    payload.pop("model_variant")
    payload["target_period"] = target_period
    request = NearTermRequest.model_validate(payload)
    comparison_period = "2025Q3" if target_period == "2026Q3" else "2025Q4"
    comparison = next((row for row in request.history if row.period == comparison_period), None)
    if comparison is None or comparison.sales_amount is None or comparison.sales_amount <= 0:
        raise ValueError("Missing or non-positive comparison sales")
    return request.model_dump(mode="json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--region-code", required=True)
    parser.add_argument("--industry-code", required=True)
    parser.add_argument("--target-period", choices=["2026Q3", "2026Q4"], required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("Refusing to overwrite output")
        payload = make_near_term_request(
            args.input, args.region_code, args.industry_code, args.target_period, args.as_of
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Saved a five-quarter near-term research request to {args.output}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
