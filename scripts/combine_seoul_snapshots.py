"""Combine a verified historical Seoul snapshot with acquired API quarters.

This preserves each cohort's actual acquisition date, never a fabricated
historical release date. The resulting dataset is for research and must not
be treated as a point-in-time historical backtest.
"""

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pickplace_ai.artifacts import file_sha256
from pickplace_ai.data import read_observations, write_observations
from pickplace_ai.ingest import import_seoul
from pickplace_ai.periods import quarter_end

_PERIOD = re.compile(r"20\d\dQ[1-4]\Z")
_KINDS = {"sales", "stores"}


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_original_sources(manifest, raw_root):
    if manifest.get("industry_code") != "all" or manifest.get("sales_basis") != "as_published":
        raise ValueError("Expected an all-industry, as-published source manifest")
    files = manifest.get("source_files")
    if not isinstance(files, list) or not files:
        raise ValueError("Historical manifest has no source files")
    seen = set()
    for item in files:
        kind, name = item["category"], item["name"]
        if kind not in _KINDS or not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid historical source path")
        if (kind, name) in seen:
            raise ValueError("Duplicate historical source")
        seen.add((kind, name))
        if file_sha256(raw_root / kind / name) != item["sha256"]:
            raise ValueError("Historical source checksum mismatch")


def _verify_api_source(raw_root, kind, period):
    csv_path = raw_root / kind / f"seoul-api-{kind}-{period}.csv"
    meta_path = csv_path.with_suffix(".metadata.json")
    metadata_hash = file_sha256(meta_path)
    meta = _load(meta_path)
    if meta.get("kind") != kind or meta.get("period") != period:
        raise ValueError("API metadata does not match source period and kind")
    expected_sha = meta.get("sha256")
    if file_sha256(csv_path) != expected_sha:
        raise ValueError("API source checksum mismatch")
    acquired = datetime.fromisoformat(meta["acquired_at_utc"])
    started = datetime.fromisoformat(meta["collection_started_at_utc"])
    if acquired.utcoffset() is None or started.utcoffset() is None or acquired < started:
        raise ValueError("Invalid API acquisition time")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Invalid API CSV header")
        if "STDR_YYQU_CD" not in reader.fieldnames:
            raise ValueError("Missing API quarter field")
        quarters = Counter()
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed API CSV row")
            quarters[row["STDR_YYQU_CD"]] += 1
    if quarters != {period.replace("Q", ""): meta.get("rows")}:
        raise ValueError("API CSV count or quarter does not match metadata")
    if file_sha256(csv_path) != expected_sha or file_sha256(meta_path) != metadata_hash:
        raise ValueError("API source changed while reading")
    return csv_path, meta, metadata_hash, acquired


def combine(historical_input, historical_summary, historical_manifest, raw_root, periods, output):
    historical_input, historical_summary, historical_manifest, raw_root, output = map(
        Path, (historical_input, historical_summary, historical_manifest, raw_root, output)
    )
    if output.suffix.lower() != ".csv":
        raise ValueError("Output must be a CSV file")
    metadata_out = output.with_suffix(".manifest.json")
    if output.exists() or metadata_out.exists():
        raise FileExistsError("Refusing to overwrite output or manifest")
    periods = list(periods)
    if (
        not periods
        or len(set(periods)) != len(periods)
        or any(not _PERIOD.fullmatch(p) for p in periods)
    ):
        raise ValueError("Provide distinct quarters in YYYYQ1..YYYYQ4 form")
    summary_hash = file_sha256(historical_summary)
    manifest_hash = file_sha256(historical_manifest)
    summary, manifest = _load(historical_summary), _load(historical_manifest)
    if summary.get("raw_source_manifest_sha256") != manifest_hash:
        raise ValueError("Historical manifest checksum mismatch")
    old_input_hash = file_sha256(historical_input)
    if summary.get("input_sha256") != old_input_hash:
        raise ValueError("Historical canonical input checksum mismatch")
    historic_at = date.fromisoformat(summary["snapshot_available_at"])
    if manifest.get("snapshot_available_at") != historic_at.isoformat():
        raise ValueError("Historical acquisition dates do not match")
    _verify_original_sources(manifest, raw_root)
    existing = read_observations(historical_input)
    if len(existing) != summary["input_observations"]:
        raise ValueError("Historical row count mismatch")
    if any(
        r.available_at != historic_at
        or r.source_kind != "official"
        or r.region_code_system != "seoul-adstrd"
        or re.fullmatch(r"11[0-9]{6}", r.region_code) is None
        for r in existing
    ):
        raise ValueError("Historical data has an unexpected acquisition date, source or region")

    combined, sources, rows_by_period, available_at_by_period = existing, [], {}, {}
    for period in periods:
        paths, acquisitions = {}, []
        for kind in ("sales", "stores"):
            path, meta, meta_hash, acquired = _verify_api_source(raw_root, kind, period)
            paths[kind] = path
            acquisitions.append(acquired)
            sources.append(
                {
                    "kind": kind,
                    "period": period,
                    "file": path.name,
                    "sha256": meta["sha256"],
                    "metadata_sha256": meta_hash,
                    "rows": meta["rows"],
                    "acquired_at_utc": meta["acquired_at_utc"],
                    "catalog": meta["catalog"],
                }
            )
        local_date = max(acquisitions).astimezone(ZoneInfo("Asia/Seoul")).date()
        available_at_by_period[period] = local_date.isoformat()
        if quarter_end(period) > local_date:
            raise ValueError("API source acquired before its quarter ended")
        rows = import_seoul(
            [paths["sales"]], [paths["stores"]], local_date, sales_basis="as_published"
        )
        expected = next(
            x["rows"] for x in sources if x["kind"] == "sales" and x["period"] == period
        )
        if len(rows) != expected or any(r.period != period for r in rows):
            raise ValueError("Imported sales rows do not match API metadata")
        combined.extend(rows)
        rows_by_period[period] = len(rows)

    if (
        file_sha256(historical_input) != old_input_hash
        or file_sha256(historical_summary) != summary_hash
    ):
        raise ValueError("Historical input changed during processing")
    if file_sha256(historical_manifest) != manifest_hash:
        raise ValueError("Historical manifest changed during processing")
    for source in sources:
        path = raw_root / source["kind"] / source["file"]
        if (
            file_sha256(path) != source["sha256"]
            or file_sha256(path.with_suffix(".metadata.json")) != source["metadata_sha256"]
        ):
            raise ValueError("API source changed during processing")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seoul-combine-", dir=output.parent) as directory:
        stage = Path(directory) / "combined.csv"
        stage_meta = Path(directory) / "manifest.json"
        write_observations(combined, stage)  # validates duplicate scope and source kind
        result = {
            "output_sha256": file_sha256(stage),
            "rows": len(combined),
            "historical_input_sha256": old_input_hash,
            "historical_summary_sha256": summary_hash,
            "historical_manifest_sha256": manifest_hash,
            "historical_available_at": historic_at.isoformat(),
            "api_sources": sources,
            "api_rows_by_period": rows_by_period,
            "api_available_at_by_period": available_at_by_period,
            "sales_basis": "as_published",
            "region_code_system": "seoul-adstrd",
            "note": (
                "Acquisition dates are not original publication dates; current revised snapshots "
                "do not establish point-in-time backtest validity; API pagination is not atomic."
            ),
        }
        stage_meta.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.link(stage_meta, metadata_out)
        try:
            os.link(stage, output)
        except OSError:
            if os.path.samestat(os.stat(stage_meta), os.stat(metadata_out)):
                metadata_out.unlink()
            raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-input", type=Path, required=True)
    parser.add_argument("--historical-summary", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--periods", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = combine(
            args.historical_input,
            args.historical_summary,
            args.historical_manifest,
            args.raw_root,
            args.periods,
            args.output,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "rows": result["rows"],
                    "api_rows_by_period": result["api_rows_by_period"],
                },
                ensure_ascii=False,
            )
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
