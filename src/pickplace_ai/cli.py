"""Local preparation commands. Importing the API never trains or generates data."""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from pickplace_ai.data import quality_report, read_observations, write_observations
from pickplace_ai.demo import example_request, synthetic_observations
from pickplace_ai.features import build_training_frame


def _json_file(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="pickplace-ai")
    commands = result.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("make-demo", help="Generate explicitly synthetic local fixtures")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--request-output", type=Path)
    demo.add_argument("--regions", type=int, default=32)
    demo.add_argument("--quarters", type=int, default=24)
    demo.add_argument("--seed", type=int, default=42)
    inspect = commands.add_parser("inspect", help="Check canonical input quality and coverage")
    inspect.add_argument("--input", required=True, type=Path)
    inspect.add_argument("--output", type=Path)
    prepare = commands.add_parser("prepare", help="Build publication-aware feature/label rows")
    prepare.add_argument("--input", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    fit = commands.add_parser("train", help="Compare baselines and train an experimental model")
    fit.add_argument("--input", required=True, type=Path)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--model", choices=["hist", "lightgbm"], default="hist")
    fit.add_argument("--validation-quarters", type=int, default=2)
    fit.add_argument("--test-quarters", type=int, default=2)
    fit.add_argument("--seed", type=int, default=42)
    seoul = commands.add_parser("import-seoul", help="Import officially downloaded Seoul CSVs")
    seoul.add_argument("--sales", nargs="+", required=True, type=Path)
    seoul.add_argument("--stores", nargs="*", default=[], type=Path)
    seoul.add_argument("--available-at", required=True, type=date.fromisoformat)
    seoul.add_argument("--industry-code")
    seoul.add_argument("--sales-basis", required=True, choices=["as_published"])
    seoul.add_argument("--output", required=True, type=Path)
    combine = commands.add_parser("combine", help="Combine canonical, nonoverlapping data batches")
    combine.add_argument("--input", nargs="+", required=True, type=Path)
    combine.add_argument("--output", required=True, type=Path)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "make-demo":
            rows = synthetic_observations(args.regions, args.quarters, args.seed)
            if args.request_output is not None and args.request_output.exists():
                raise FileExistsError(f"Request file already exists: {args.request_output}")
            write_observations(rows, args.output)
            if args.request_output is not None:
                _json_file(args.request_output, example_request(rows).model_dump(mode="json"))
            print(
                json.dumps(
                    {
                        "rows": len(rows),
                        "source_kind": "synthetic",
                        "warning": "Not real commercial data or real model accuracy",
                    }
                )
            )
        elif args.command == "inspect":
            report = quality_report(read_observations(args.input))
            if args.output:
                _json_file(args.output, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.command == "prepare":
            if args.output.exists():
                raise FileExistsError(f"Output already exists: {args.output}")
            frame = build_training_frame(read_observations(args.input))
            if frame.empty:
                raise ValueError(
                    "No eligible feature/label rows. Check five contiguous quarters, positive "
                    "comparison sales, actual historical publication dates and future labels."
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            print(json.dumps({"eligible_rows": len(frame), "output": str(args.output)}))
        elif args.command == "train":
            from pickplace_ai.training import train

            metadata = train(
                read_observations(args.input),
                args.output,
                args.model,
                args.validation_quarters,
                args.test_quarters,
                args.seed,
            )
            print(
                json.dumps(
                    {
                        "model_version": metadata["model_version"],
                        "source_kind": metadata["source_kind"],
                        "release_status": metadata["release_status"],
                        "selected_model": metadata["selected_model"],
                        "test_metrics": metadata["test_metrics"],
                        "output": str(args.output),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.command == "import-seoul":
            from pickplace_ai.ingest import import_seoul

            rows = import_seoul(
                args.sales,
                args.stores,
                args.available_at,
                args.industry_code,
                sales_basis=args.sales_basis,
            )
            write_observations(rows, args.output)
            print(
                json.dumps(
                    {
                        "rows": len(rows),
                        "source_kind": "official",
                        "publication_date_verified_by": "caller",
                        "output": str(args.output),
                    },
                    ensure_ascii=False,
                )
            )
        elif args.command == "combine":
            rows = [row for path in args.input for row in read_observations(path)]
            write_observations(rows, args.output)
            print(json.dumps({"rows": len(rows), "output": str(args.output)}))
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
