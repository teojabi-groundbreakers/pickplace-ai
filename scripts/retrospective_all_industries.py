"""Run isolated industry experiments on a single locally acquired snapshot.

Results are retrospective research only. They cannot establish point-in-time
performance or be loaded by the production API.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from contextlib import ExitStack
from datetime import date
from pathlib import Path

from pickplace_ai.artifacts import file_sha256
from scripts import retrospective_experiment as experiment

INDUSTRY_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
REQUIRED = {
    "region_code_system",
    "region_code",
    "industry_code",
    "period",
    "available_at",
    "source_kind",
}


def _partition(input_path, directory, snapshot_at, source_kind):
    """Stream canonical records into industry files without inventing dates."""
    files, counts, seen, folded = {}, Counter(), set(), {}
    with input_path.open(encoding="utf-8-sig", newline="") as handle, ExitStack() as stack:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames
        if not headers or any(not name for name in headers) or len(set(headers)) != len(headers):
            raise ValueError("Missing, duplicate or empty canonical CSV header")
        if REQUIRED - set(headers):
            raise ValueError(f"Missing canonical fields: {sorted(REQUIRED - set(headers))}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Line {line}: inconsistent number of CSV fields")
            code = row["industry_code"]
            if not INDUSTRY_CODE.fullmatch(code):
                raise ValueError(f"Line {line}: invalid industry code {code!r}")
            if code.casefold() in folded and folded[code.casefold()] != code:
                raise ValueError(f"Line {line}: case-insensitive industry code collision")
            folded[code.casefold()] = code
            if row["available_at"] != snapshot_at.isoformat():
                raise ValueError(f"Line {line}: snapshot availability does not match")
            if row["source_kind"] != source_kind:
                raise ValueError(f"Line {line}: source kind does not match")
            key = (row["region_code_system"], row["region_code"], code, row["period"])
            if not all(key):
                raise ValueError(f"Line {line}: missing scope or period")
            if key in seen:
                raise ValueError(f"Line {line}: duplicate scope/quarter {key}")
            seen.add(key)
            if code not in files:
                path = directory / f"{code}.csv"
                output = stack.enter_context(path.open("x", encoding="utf-8", newline=""))
                writer = csv.DictWriter(output, fieldnames=headers)
                writer.writeheader()
                files[code] = (path, writer)
            files[code][1].writerow(row)
            counts[code] += 1
    if not files:
        raise ValueError("Dataset contains no observations")
    return {code: item[0] for code, item in files.items()}, counts


def _source_manifest(manifest_bytes, raw_root, snapshot_at):
    data = json.loads(manifest_bytes)
    if data.get("snapshot_available_at") != snapshot_at.isoformat():
        raise ValueError("Manifest snapshot date does not match")
    if data.get("industry_code") != "all":
        raise ValueError("An all-industry manifest must identify industry_code as 'all'")
    sources = data.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Source manifest has no files")
    seen = set()
    for item in sources:
        category, name = item["category"], item["name"]
        if category not in {"sales", "stores"} or Path(name).name != name:
            raise ValueError("Invalid source manifest path")
        key = (category, name)
        if key in seen:
            raise ValueError("Duplicate source manifest entry")
        seen.add(key)
        if file_sha256(raw_root / category / name) != item["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
    return data


def run_all(
    input_path,
    output_path,
    snapshot_at,
    manifest_path=None,
    raw_root=None,
    *,
    source_kind="official",
    industry_names_path=None,
):
    """Train each industry independently and record successes and failures."""
    input_path, output_path = Path(input_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    if source_kind not in {"official", "synthetic"}:
        raise ValueError("Unknown source kind")
    names = {}
    names_bytes = None
    if industry_names_path is not None:
        names_bytes = Path(industry_names_path).read_bytes()
        names_data = json.loads(names_bytes)
        names = names_data.get("names")
        if not isinstance(names, dict) or names_data.get("conflicts"):
            raise ValueError("Industry names must be a conflict-free code-to-name mapping")
        if any(
            not INDUSTRY_CODE.fullmatch(code) or not isinstance(name, str) or not name
            for code, name in names.items()
        ):
            raise ValueError("Invalid industry name mapping")
    manifest = None
    manifest_bytes = None
    if source_kind == "official":
        if manifest_path is None or raw_root is None:
            raise ValueError("Official data require an all-industry manifest and raw root")
        manifest_path, raw_root = Path(manifest_path), Path(raw_root)
        manifest_bytes = manifest_path.read_bytes()
        manifest = _source_manifest(manifest_bytes, raw_root, snapshot_at)
    whole_input_sha = file_sha256(input_path)
    with tempfile.TemporaryDirectory(prefix="pickplace-industry-split-") as tmp:
        tmp_path = Path(tmp)
        files, counts = _partition(input_path, tmp_path, snapshot_at, source_kind)
        if file_sha256(input_path) != whole_input_sha:
            raise ValueError("Input changed while being partitioned")
        output_path.mkdir(parents=True, exist_ok=False)
        models = output_path / "industries"
        models.mkdir()
        rows = []
        for index, code in enumerate(sorted(files), 1):
            print(f"Processing industry {index}/{len(files)}: {code}", file=sys.stderr, flush=True)
            record = {
                "industry_code": code,
                "industry_name": names.get(code),
                "status": "failed",
                "input_observations": counts[code],
                "input_sha256": file_sha256(files[code]),
            }
            try:
                per_industry_manifest = None
                if manifest is not None:
                    per_industry_manifest = tmp_path / f"{code}.manifest.json"
                    per_industry_manifest.write_text(
                        json.dumps(
                            {**manifest, "industry_code": code}, ensure_ascii=False, indent=2
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                metadata, evaluation = experiment.run(
                    files[code],
                    models / code,
                    snapshot_at,
                    source_kind=source_kind,
                    manifest_path=per_industry_manifest,
                    raw_root=raw_root,
                )
                selection = metadata["selection"]
                learned = metadata["learned_model_selection"]
                metadata.update(
                    {
                        "industry_code": code,
                        "industry_name": names.get(code),
                        "all_industries_input_sha256": whole_input_sha,
                        "all_industries_runner_sha256": file_sha256(Path(__file__)),
                    }
                )
                evaluation.update({"industry_code": code, "industry_name": names.get(code)})
                for filename, value in (
                    ("metadata.json", metadata),
                    ("evaluation.json", evaluation),
                ):
                    (models / code / filename).write_text(
                        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8",
                    )
                record.update(
                    {
                        "status": "completed",
                        "eligible_rows": metadata["eligible_rows"],
                        "rows_by_partition": metadata["rows_by_partition"],
                        "quarter_splits": metadata["quarter_splits"],
                        "embargo_quarters": metadata["embargo_quarters"],
                        "selection": selection,
                        "learned_model_selection": learned,
                        "validation_mae_pp": evaluation["validation"][selection]["mae_pp"],
                        "test_mae_pp": evaluation["test"][selection]["mae_pp"],
                        "learned_validation_mae_pp": evaluation["validation"][learned]["mae_pp"],
                        "learned_test_mae_pp": evaluation["test"][learned]["mae_pp"],
                        "warnings": [],
                    }
                )
                if metadata["rows_by_partition"]["test"] < 30:
                    record["warnings"].append("fewer_than_30_test_samples")
                if metadata["rows_by_partition"]["validation"] < 30:
                    record["warnings"].append("fewer_than_30_validation_samples")
                if metadata["rows_by_partition"]["train"] < 100:
                    record["warnings"].append("fewer_than_100_training_samples")
                if industry_names_path is not None and code not in names:
                    record["warnings"].append("missing_industry_name")
                card = models / code / "MODEL_CARD.md"
                with card.open("a", encoding="utf-8") as stream:
                    stream.write(
                        f"\n- 대상 업종: {code} ({names.get(code) or '업종명 미확인'})\n"
                        f"- 시험 표본: {record['rows_by_partition']['test']}건\n"
                        f"- 주의: {', '.join(record['warnings']) or '추가 표본 경고 없음'}\n"
                    )
                record["artifact_status"] = "complete"
                record["artifact_path"] = f"industries/{code}"
            except Exception as exc:
                record["status"] = "failed"
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)
                record["artifact_status"] = "none"
                partial = models / code
                if partial.exists():
                    try:
                        quarantine = output_path / "failed"
                        quarantine.mkdir(exist_ok=True)
                        partial.rename(quarantine / code)
                        record["artifact_status"] = "quarantined_partial"
                        record["artifact_path"] = f"failed/{code}"
                    except OSError as quarantine_error:
                        record["artifact_status"] = "partial_not_quarantined"
                        record["artifact_path"] = f"industries/{code}"
                        record["quarantine_error"] = str(quarantine_error)
                print(f"Industry {code} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append(record)
        complete = [row for row in rows if row["status"] == "completed"]
        test_periods = Counter(tuple(row["quarter_splits"]["test"]) for row in complete)
        reference = list(test_periods.most_common(1)[0][0]) if test_periods else []
        for row in complete:
            if row["quarter_splits"]["test"] != reference:
                row["warnings"].append("different_test_period_from_most_industries")
        summary = {
            "artifact_kind": "all_industries_retrospective_research_v1",
            "release_status": "retrospective_research_only",
            "interpretation": (
                "Current revised snapshot, not point-in-time backtesting or production."
            ),
            "snapshot_available_at": snapshot_at.isoformat(),
            "source_kind": source_kind,
            "input_file": input_path.name,
            "input_sha256": whole_input_sha,
            "input_observations": sum(counts.values()),
            "runner_sha256": file_sha256(Path(__file__)),
            "raw_source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()
            if manifest_bytes is not None
            else None,
            "industry_names_sha256": hashlib.sha256(names_bytes).hexdigest()
            if names_bytes is not None
            else None,
            "reference_test_quarters": reference,
            "warning_thresholds": {
                "test_rows_below": 30,
                "validation_rows_below": 30,
                "training_rows_below": 100,
            },
            "industry_count": len(rows),
            "completed_count": len(complete),
            "failed_count": len(rows) - len(complete),
            "industries": rows,
        }
        (output_path / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        fields = [
            "industry_code",
            "industry_name",
            "status",
            "input_observations",
            "eligible_rows",
            "selection",
            "learned_model_selection",
            "validation_mae_pp",
            "test_mae_pp",
            "learned_validation_mae_pp",
            "learned_test_mae_pp",
            "train_rows",
            "validation_rows",
            "test_rows",
            "train_quarters",
            "validation_quarters",
            "test_quarters",
            "warnings",
            "artifact_status",
            "artifact_path",
            "error_type",
            "error",
            "quarantine_error",
        ]
        with (output_path / "summary.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                result = {name: row.get(name, "") for name in fields}
                for partition in ("train", "validation", "test"):
                    result[f"{partition}_rows"] = row.get("rows_by_partition", {}).get(
                        partition, ""
                    )
                    result[f"{partition}_quarters"] = ",".join(
                        row.get("quarter_splits", {}).get(partition, [])
                    )
                result["warnings"] = ",".join(row.get("warnings", []))
                writer.writerow(result)
        if manifest_bytes is not None:
            (output_path / "source-manifest.json").write_bytes(manifest_bytes)
        if names_bytes is not None:
            (output_path / "industry-names.json").write_bytes(names_bytes)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-at", type=date.fromisoformat, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--industry-names", type=Path)
    args = parser.parse_args()
    result = run_all(
        args.input,
        args.output,
        args.snapshot_at,
        args.manifest,
        args.raw_root,
        industry_names_path=args.industry_names,
    )
    print(
        json.dumps(
            {key: result[key] for key in ("industry_count", "completed_count", "failed_count")},
            ensure_ascii=False,
        )
    )
    return 1 if result["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
