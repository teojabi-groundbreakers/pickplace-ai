"""Download complete Seoul administrative-dong API quarters as source CSVs.

This uses the provider's published API field names, which import-seoul accepts.
The official endpoint currently documents HTTP, with the API key in the URL;
avoid untrusted networks and never log request URLs or exception details.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, build_opener

SERVICES = {"sales": "VwsmAdstrdSelngW", "stores": "VwsmAdstrdStorW"}
ROOT = "http://openapi.seoul.go.kr:8088"
PAGE_SIZE = 1000
CATALOG = {
    "sales": "https://data.seoul.go.kr/dataList/OA-22175/S/1/datasetView.do",
    "stores": "https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do",
}
REQUIRED = {
    "sales": {"STDR_YYQU_CD", "ADSTRD_CD", "SVC_INDUTY_CD", "THSMON_SELNG_AMT"},
    "stores": {"STDR_YYQU_CD", "ADSTRD_CD", "SVC_INDUTY_CD", "SIMILR_INDUTY_STOR_CO"},
}


class FetchError(ValueError):
    """An API failure, incomplete result, or changed source dataset."""


def normalize_period(period):
    match = re.fullmatch(r"(20\d{2})Q?([1-4])", period.upper())
    if not match:
        raise ValueError("Expected a period such as 2026Q1 or 20261")
    return "".join(match.groups())


def validated_key(value):
    if value and value.lower() == "sample":
        raise FetchError("The sample key is limited to five records; use your own API key")
    if not value or not re.fullmatch(r"[A-Za-z0-9]{8,128}", value):
        raise FetchError("Set a valid SEOUL_OPEN_DATA_API_KEY environment variable")
    return value


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_page(key, service, start, end, period):
    # Do not expose the key in exceptions, diagnostic text, or saved metadata.
    url = f"{ROOT}/{key}/json/{service}/{start}/{end}/{period}"
    try:
        with build_opener(NoRedirect).open(url, timeout=30) as response:
            return json.load(response, parse_float=Decimal)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        raise FetchError("API request failed or returned invalid JSON; no CSV was saved") from None


def _page(data, service, expected_total):
    if not isinstance(data, dict):
        raise FetchError("Unexpected API response")
    envelope = data.get(service)
    if envelope is not None and not isinstance(envelope, dict):
        raise FetchError("Invalid API service envelope")
    result = data.get("RESULT") if envelope is None else envelope.get("RESULT")
    if not isinstance(result, dict) or result.get("CODE") != "INFO-000":
        code = result.get("CODE") if isinstance(result, dict) else "unknown"
        # Whitelist error codes so arbitrary server-provided text is not logged.
        code = (
            code
            if isinstance(code, str) and re.fullmatch(r"(?:INFO|ERROR)-\d{3}", code)
            else "unknown"
        )
        raise FetchError(f"API returned {code}; no CSV was saved")
    if not isinstance(envelope, dict):
        raise FetchError("Missing API service envelope")
    total = envelope.get("list_total_count")
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise FetchError("Invalid total count; no CSV was saved")
    if expected_total is not None and total != expected_total:
        raise FetchError("API total changed during download; retry the entire quarter")
    rows = envelope.get("row")
    if not isinstance(rows, list) or not rows:
        raise FetchError("Empty or invalid page; no CSV was saved")
    return total, rows


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise FetchError("Unexpected API field value")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise FetchError("Non-finite API value")
        return format(value, "f")
    return str(value)


def fetch_quarter(
    kind, period, output_root, key, *, allow_insecure_http=False, fetch_page=request_page
):
    """Create exactly one source CSV or leave the destination untouched."""
    key = validated_key(key)
    if not allow_insecure_http:
        raise FetchError(
            "Official API uses unencrypted HTTP and exposes the key on the network; "
            "request requires --allow-insecure-http"
        )
    if kind not in SERVICES:
        raise ValueError("Unknown dataset")
    period = normalize_period(period)
    service = SERVICES[kind]
    folder = Path(output_root) / kind
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"seoul-api-{kind}-{period[:4]}Q{period[-1]}.csv"
    metadata_path = destination.with_suffix(".metadata.json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite files for {destination}")

    temp_name, metadata_temp = None, None
    started_at = datetime.now(UTC).isoformat()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=folder,
            prefix=".seoul-api-",
            delete=False,
        ) as stream:
            temp_name = stream.name
            writer, header, total, seen = None, None, None, set()
            start = 1
            while total is None or start <= total:
                end = start + PAGE_SIZE - 1 if total is None else min(start + PAGE_SIZE - 1, total)
                page = fetch_page(key, service, start, end, period)
                total, rows = _page(page, service, total)
                expected = min(PAGE_SIZE, total - start + 1)
                if len(rows) != expected:
                    raise FetchError("Incomplete or oversized API page; no CSV was saved")
                for row in rows:
                    if not isinstance(row, dict) or any(not isinstance(k, str) for k in row):
                        raise FetchError("Invalid API row")
                    if header is None:
                        header = list(row)
                        if not REQUIRED[kind].issubset(header):
                            raise FetchError("Required API fields are missing")
                        writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
                        writer.writeheader()
                    if set(row) != set(header):
                        raise FetchError("API fields changed across rows")
                    values = {name: _cell(value) for name, value in row.items()}
                    region = values["ADSTRD_CD"]
                    industry = values["SVC_INDUTY_CD"]
                    if values["STDR_YYQU_CD"] != period:
                        raise FetchError("API returned an unexpected quarter")
                    if not re.fullmatch(r"11[0-9]{6}", region):
                        raise FetchError("Invalid Seoul administrative-dong code")
                    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", industry):
                        raise FetchError("Invalid industry code")
                    identity = (period, region, industry)
                    if identity in seen:
                        raise FetchError("Duplicate API row; retry the entire quarter")
                    seen.add(identity)
                    writer.writerow(values)
                start += len(rows)
            if len(seen) != total:
                raise FetchError("Collected row count does not match API total")
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(Path(temp_name).read_bytes()).hexdigest()
        metadata = {
            "kind": kind,
            "period": f"{period[:4]}Q{period[-1]}",
            "rows": total,
            "sha256": digest,
            "collection_started_at_utc": started_at,
            "acquired_at_utc": datetime.now(UTC).isoformat(),
            "catalog": CATALOG[kind],
            "service": service,
            "source_fields": header,
            "encoding": "utf-8-sig",
            "note": (
                "Acquisition time is not the source's historical publication time; "
                "stable count and unique keys do not guarantee a transactional snapshot."
            ),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=folder, prefix=".seoul-meta-", delete=False
        ) as stream:
            metadata_temp = stream.name
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # link() is an atomic no-overwrite commit on the same filesystem. The
        # metadata is committed first and rolled back if the CSV link fails.
        os.link(metadata_temp, metadata_path)
        try:
            os.link(temp_name, destination)
        except OSError:
            if os.path.samestat(os.stat(metadata_temp), os.stat(metadata_path)):
                metadata_path.unlink()
            raise
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
        if metadata_temp is not None:
            Path(metadata_temp).unlink(missing_ok=True)
    return {**metadata, "path": str(destination), "metadata_path": str(metadata_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", nargs="+", required=True, help="e.g. 2026Q1 2026Q2")
    parser.add_argument("--types", nargs="+", choices=sorted(SERVICES), default=sorted(SERVICES))
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/seoul"))
    parser.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="explicitly permit the official HTTP endpoint to transmit the key unencrypted",
    )
    args = parser.parse_args(argv)
    try:
        periods = [normalize_period(value) for value in args.periods]
        if len(set(periods)) != len(periods) or len(set(args.types)) != len(args.types):
            raise ValueError("Duplicate period or dataset")
        key = validated_key(os.environ.get("SEOUL_OPEN_DATA_API_KEY"))
        for period in periods:
            for kind in args.types:
                result = fetch_quarter(
                    kind,
                    period,
                    args.output_root,
                    key,
                    allow_insecure_http=args.allow_insecure_http,
                )
                print(json.dumps(result, ensure_ascii=False), flush=True)
    except (FetchError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
