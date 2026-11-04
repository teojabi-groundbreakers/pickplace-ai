import csv
import hashlib
import json
from datetime import date
from decimal import Decimal
from urllib.error import URLError

import pytest

from pickplace_ai.ingest import import_seoul
from scripts import fetch_seoul_api as api

KEY = "testkey123456"


def row(index, period="20261", *, stores=False):
    return {
        "STDR_YYQU_CD": period,
        "ADSTRD_CD": f"11{index:06d}",
        "ADSTRD_CD_NM": "테스트동",
        "SVC_INDUTY_CD": "CS100010",
        **(
            {"SIMILR_INDUTY_STOR_CO": Decimal("10.0"), "OPBIZ_STOR_CO": 1}
            if stores
            else {"THSMON_SELNG_AMT": Decimal("4.5019103E9")}
        ),
    }


def envelope(service, rows, total=None):
    return {
        service: {
            "list_total_count": len(rows) if total is None else total,
            "RESULT": {"CODE": "INFO-000"},
            "row": rows,
        }
    }


def test_pagination_compatible_csv_and_secret_free_metadata(tmp_path):
    calls = []

    def get_page(key, service, start, end, period):
        calls.append((start, end))
        rows = [row(n) for n in range(start, end + 1)]
        return envelope(service, rows, total=1001)

    output = api.fetch_quarter(
        "sales", "2026Q1", tmp_path, KEY, allow_insecure_http=True, fetch_page=get_page
    )
    assert calls == [(1, 1000), (1001, 1001)]
    path = tmp_path / "sales/seoul-api-sales-2026Q1.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1001
    assert rows[0]["THSMON_SELNG_AMT"] == "4501910300"
    assert rows[0]["ADSTRD_CD_NM"] == "테스트동"
    meta_text = path.with_suffix(".metadata.json").read_text()
    meta = json.loads(meta_text)
    assert meta["rows"] == 1001
    assert meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert KEY not in meta_text
    assert output["sha256"] == meta["sha256"]
    observations = import_seoul([path], [], date(2026, 11, 4))
    assert len(observations) == 1001
    assert observations[0].sales_amount == 4_501_910_300


def test_both_api_schemas_work_with_existing_importer(tmp_path):
    def get_page(key, service, start, end, period):
        return envelope(service, [row(1, stores=service.endswith("StorW"))])

    for kind in ("sales", "stores"):
        api.fetch_quarter(
            kind, "20261", tmp_path, KEY, allow_insecure_http=True, fetch_page=get_page
        )
    records = import_seoul(
        [tmp_path / "sales/seoul-api-sales-2026Q1.csv"],
        [tmp_path / "stores/seoul-api-stores-2026Q1.csv"],
        date(2026, 11, 4),
    )
    assert len(records) == 1
    assert records[0].sales_amount == 4_501_910_300
    assert records[0].store_count == 10
    assert records[0].opening_count == 1


@pytest.mark.parametrize("bad", ["duplicate", "wrong_period", "incomplete", "total_change"])
def test_incomplete_and_inconsistent_pages_leave_no_output(tmp_path, bad):
    def get_page(key, service, start, end, period):
        if start == 1:
            first = [row(n) for n in range(1, 1001)]
            if bad == "wrong_period":
                first[0] = row(1, period="20262")
            if bad == "incomplete":
                first.pop()
            return envelope(service, first, total=1001)
        return envelope(
            service,
            [row(1 if bad == "duplicate" else 1001)],
            total=1002 if bad == "total_change" else 1001,
        )

    with pytest.raises(api.FetchError):
        api.fetch_quarter(
            "sales", "20261", tmp_path, KEY, allow_insecure_http=True, fetch_page=get_page
        )
    assert list((tmp_path / "sales").iterdir()) == []


def test_error_no_data_and_existing_file_are_not_overwritten(tmp_path):
    def no_data(*args):
        return {"RESULT": {"CODE": "INFO-200", "MESSAGE": "No data"}}

    with pytest.raises(api.FetchError, match="INFO-200"):
        api.fetch_quarter(
            "sales", "20263", tmp_path, KEY, allow_insecure_http=True, fetch_page=no_data
        )
    path = tmp_path / "sales/seoul-api-sales-2026Q1.csv"
    path.write_text("existing")
    with pytest.raises(FileExistsError):
        api.fetch_quarter("sales", "20261", tmp_path, KEY, allow_insecure_http=True)
    assert path.read_text() == "existing"


def test_no_network_without_explicit_http_opt_in(tmp_path):
    def unexpected(*args):
        raise AssertionError("network call must be blocked")

    with pytest.raises(api.FetchError, match="unencrypted HTTP"):
        api.fetch_quarter("sales", "20261", tmp_path, KEY, fetch_page=unexpected)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(api.FetchError, match="sample key"):
        api.validated_key("sample")


def test_network_exception_and_redirect_do_not_reveal_key(monkeypatch):
    class Opener:
        def open(self, url, timeout):
            raise URLError(f"failed URL {url}")

    monkeypatch.setattr(api, "build_opener", lambda *_: Opener())
    with pytest.raises(api.FetchError) as failure:
        api.request_page(KEY, api.SERVICES["sales"], 1, 1, "20261")
    assert KEY not in str(failure.value)
    assert failure.value.__cause__ is None
    assert api.NoRedirect().redirect_request(None, None, 302, None, None, "http://attacker") is None
