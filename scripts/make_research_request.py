"""Build one historical replay request from a local canonical Seoul snapshot.

The source data stays on the local machine; only a deliberately selected
five-quarter request is written to the caller-selected output path.
"""

import argparse
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

from pickplace_ai.data import FLOAT_FIELDS, INTEGER_FIELDS
from pickplace_ai.features import FeatureUnavailable, feature_record
from pickplace_ai.periods import quarter_end, shift_quarter
from pickplace_ai.schemas import Observation, PredictionRequest

REQUIRED_COLUMNS = {
    "region_code",
    "region_code_system",
    "industry_code",
    "period",
    "available_at",
    "source_kind",
    "source_dataset",
}


def make_request(input_path, region_code, industry_code, base_period, as_of, model_variant):
    """Return a validated request for one Seoul region, industry and replay origin."""
    if not re.fullmatch(r"11[0-9]{6}", region_code):
        raise ValueError("Expected an eight-digit Seoul administrative region code")
    if not industry_code:
        raise ValueError("Industry code is required")
    if model_variant not in {"selected", "learned"}:
        raise ValueError("Unknown model variant")
    periods = [shift_quarter(base_period, delta) for delta in range(-4, 1)]
    if quarter_end(shift_quarter(base_period, 1)) > as_of:
        raise ValueError("Historical replay requires a completed target quarter")

    selected = {}
    with Path(input_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        if not headers or any(not name for name in headers) or len(set(headers)) != len(headers):
            raise ValueError("Missing, duplicate or empty canonical CSV header")
        missing_columns = REQUIRED_COLUMNS - set(headers)
        if missing_columns:
            raise ValueError(f"Missing canonical columns: {sorted(missing_columns)}")
        for line_number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV line {line_number} has inconsistent number of columns")
            if (
                row["region_code"] != region_code
                or row["industry_code"] != industry_code
                or row["period"] not in periods
            ):
                continue
            period = row["period"]
            if period in selected:
                raise ValueError(f"Duplicate selected quarter: {period}")
            if row["region_code_system"] != "seoul-adstrd" or row["source_kind"] != "official":
                raise ValueError("Selected rows must use official seoul-adstrd data")
            for field in INTEGER_FIELDS | FLOAT_FIELDS:
                if field in row:
                    value = row[field]
                    row[field] = (
                        None
                        if value == ""
                        else int(value)
                        if field in INTEGER_FIELDS
                        else float(value)
                    )
            try:
                selected[period] = Observation.model_validate(row)
            except ValueError as exc:
                raise ValueError(f"Invalid selected row at CSV line {line_number}: {exc}") from exc

    missing_periods = [period for period in periods if period not in selected]
    if missing_periods:
        raise ValueError(f"Missing required history quarters: {', '.join(missing_periods)}")
    request = PredictionRequest(
        region_code=region_code,
        region_code_system="seoul-adstrd",
        industry_code=industry_code,
        base_period=base_period,
        as_of=as_of,
        source_kind="official",
        history=[selected[period] for period in periods],
    )
    try:
        feature_record(request.history, base_period, as_of)
    except FeatureUnavailable as exc:
        raise ValueError(f"Insufficient research features: {exc.reason}") from exc
    return {**request.model_dump(mode="json"), "model_variant": model_variant}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--region-code", required=True)
    parser.add_argument("--industry-code", required=True)
    parser.add_argument("--base-period", required=True)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-variant", choices=["selected", "learned"], default="selected")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
        payload = make_request(
            args.input,
            args.region_code,
            args.industry_code,
            args.base_period,
            args.as_of,
            args.model_variant,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Saved a five-quarter retrospective research request to {args.output}")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
