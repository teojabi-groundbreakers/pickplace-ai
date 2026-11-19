"""Save one month of an explicitly registered KOSIS population table as source CSV."""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, build_opener

API_URL = "https://kosis.kr/openapi/statisticsData.do"
TABLE_ID = "DT_1B04005N"
MAX_CELLS = 40000


class FetchError(ValueError):
    """The source response could not be safely saved."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_month(key, stats_id, month):
    params = {
        "method": "getList",
        "apiKey": key,
        "userStatsId": stats_id,
        "prdSe": "M",
        "startPrdDe": month,
        "endPrdDe": month,
        "format": "json",
        "jsonVD": "Y",
        "smblChk": "Y",
    }
    try:
        with build_opener(NoRedirect).open(f"{API_URL}?{urlencode(params)}", timeout=60) as r:
            return json.load(r)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        raise FetchError("KOSIS request failed or returned invalid JSON") from None


def _check(rows, month):
    if isinstance(rows, dict):
        code = str(rows.get("err", "unknown"))
        code = code if re.fullmatch(r"\d{1,3}", code) else "unknown"
        raise FetchError(f"KOSIS returned error {code}; no CSV was saved")
    if not isinstance(rows, list) or not rows or len(rows) >= MAX_CELLS:
        raise FetchError("Empty, invalid, or potentially oversized KOSIS response")
    if not isinstance(rows[0], dict):
        raise FetchError("Invalid source row")
    header = list(rows[0])
    required = {"ORG_ID", "TBL_ID", "PRD_SE", "PRD_DE", "C1", "ITM_ID", "DT"}
    if not required.issubset(header) or any(not isinstance(k, str) for k in header):
        raise FetchError("Required source fields are missing")
    dimensions = [f"C{i}" for i in range(1, 9) if f"C{i}" in header]
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(header):
            raise FetchError("Inconsistent source fields")
        if any(v is not None and not isinstance(v, str) for v in row.values()):
            raise FetchError("Unexpected source value type")
        if (row["ORG_ID"], row["TBL_ID"], row["PRD_SE"], row["PRD_DE"]) != (
            "101",
            TABLE_ID,
            "M",
            month,
        ):
            raise FetchError("Wrong organization, table, period or frequency")
        if not row["ITM_ID"] or any(not row[k] for k in dimensions):
            raise FetchError("A dimension or item code is missing")
        identity = tuple(row[k] for k in [*dimensions, "ITM_ID"])
        if identity in identities:
            raise FetchError("Duplicate statistical observation")
        identities.add(identity)
    return header


def fetch_month(month, stats_id, key, destination, *, request=request_month):
    if not re.fullmatch(r"20\d{2}(?:0[1-9]|1[0-2])", month):
        raise FetchError("Month must be YYYYMM")
    if not key or any(c.isspace() for c in key):
        raise FetchError("Set KOSIS_API_KEY in the environment")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/101/DT_1B04005N/[A-Za-z0-9_./-]+", stats_id):
        raise FetchError("Register table 101/DT_1B04005N and provide its userStatsId")
    destination = Path(destination)
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing CSV")
    started_at = datetime.now(UTC).isoformat()
    rows = request(key, stats_id, month)
    header = _check(rows, month)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = metadata_temp = None
    metadata_linked = False
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8-sig", newline="", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=header, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        metadata = {
            "provider": "KOSIS / 행정안전부",
            "source": "https://kosis.kr/statisticsList/mass/mass_list.jsp?org_id=101&tbl_id=DT_1B04005N",
            "api": API_URL,
            "table_id": TABLE_ID,
            "period": month,
            "frequency": "M",
            "request_started_at_utc": started_at,
            "fetched_at_utc": datetime.now(UTC).isoformat(),
            "registered_selection_sha256": hashlib.sha256(stats_id.encode()).hexdigest(),
            "row_count": len(rows),
            "csv_sha256": digest,
            "coverage_verified": False,
            "publication_date": None,
        }
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            metadata_temp = Path(stream.name)
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link is atomic and never replaces an existing file.
        os.link(metadata_temp, metadata_path)
        metadata_linked = True
        os.link(temporary, destination)
    except BaseException:
        if metadata_linked:
            metadata_path.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if metadata_temp is not None:
            metadata_temp.unlink(missing_ok=True)
    return len(rows), digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, help="Month, YYYYMM")
    parser.add_argument("--user-stats-id", required=True, help="KOSIS registered table ID")
    parser.add_argument("--output", type=Path, required=True, help="New source CSV path")
    args = parser.parse_args()
    try:
        count, digest = fetch_month(
            args.month, args.user_stats_id, os.environ.get("KOSIS_API_KEY"), args.output
        )
    except (FetchError, OSError) as exc:
        print(f"Collection failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {count} rows to {args.output}; SHA-256: {digest}")
    print("Coverage and historical code mapping must be verified before model use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
