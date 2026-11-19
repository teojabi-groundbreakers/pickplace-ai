#!/usr/bin/env python3
"""Snapshot current Bizinfo announcements. This is not a policy-spending dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

API_URL = "https://www.bizinfo.go.kr/uss/rss/bizinfoApi.do"
DOCUMENTATION_URL = "https://www.bizinfo.go.kr/apiDetail.do?id=bizinfoApi"


class CollectionError(Exception):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CollectionError("API가 다른 주소로 리디렉션했습니다. 인증키 보호를 위해 중단합니다.")


def fetch_page(key: str, page: int, size: int, hashtags: str | None) -> bytes:
    params = {
        "crtfcKey": key,
        "dataType": "json",
        "searchCnt": "0",
        "pageUnit": str(size),
        "pageIndex": str(page),
    }
    if hashtags:
        params["hashtags"] = hashtags
    request = Request(f"{API_URL}?{urlencode(params)}", headers={"Accept": "application/json"})
    try:
        with build_opener(NoRedirect()).open(request, timeout=30) as response:
            return response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        # urllib exceptions may contain the entire URL, including the credential.
        raise CollectionError(
            f"API 요청 실패 ({type(exc).__name__}). 인증 상태와 네트워크를 확인하세요."
        ) from None


def parse_page(raw: bytes) -> tuple[list[dict], int]:
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
        rows = payload["jsonArray"]["item"]
        if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
            raise ValueError("empty or malformed item")
        counts = {int(row["totCnt"]) for row in rows}
        if len(counts) != 1:
            raise ValueError("inconsistent totals")
        total = counts.pop()
        if total < len(rows):
            raise ValueError("invalid total")
        for row in rows:
            if not isinstance(row.get("seq"), str) or not row["seq"]:
                raise ValueError("missing seq")
            if any(
                not isinstance(value, (str, int, float, bool, type(None))) for value in row.values()
            ):
                raise ValueError("nested values")
        return rows, total
    except (UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise CollectionError(
            "API 응답 형식이나 전체 건수를 확인할 수 없습니다. "
            "빈 결과를 성공으로 저장하지 않았습니다."
        ) from exc


def collect(key: str, size: int, max_pages: int, hashtags: str | None, fetch=fetch_page):
    if not key.strip():
        raise CollectionError("BIZINFO_API_KEY 환경변수가 필요합니다.")
    if size <= 0 or max_pages <= 0:
        raise CollectionError("페이지 크기와 최대 페이지 수는 양수여야 합니다.")
    rows, pages, identifiers = [], [], set()
    expected = None
    for page in range(1, max_pages + 1):
        raw = fetch(key, page, size, hashtags)
        if key.encode() in raw:
            raise CollectionError("응답에 인증키가 포함되어 저장을 중단합니다.")
        items, total = parse_page(raw)
        if expected is not None and expected != total:
            raise CollectionError("페이지 사이 전체 건수가 변경돼 다시 수집해야 합니다.")
        expected = total
        for row in items:
            if row["seq"] in identifiers:
                raise CollectionError("중복 공고 ID를 발견했습니다. 페이지가 불안정할 수 있습니다.")
            identifiers.add(row["seq"])
        rows.extend(items)
        pages.append(raw)
        if len(rows) == expected:
            return rows, pages
        if len(rows) > expected or len(items) < size:
            raise CollectionError("전체 건수보다 적거나 많은 결과가 반환됐습니다.")
    raise CollectionError("최대 페이지 수에 도달해 전체 결과를 저장하지 않았습니다.")


def save_snapshot(
    destination: Path, rows: list[dict], pages: list[bytes], size: int, hashtags: str | None
):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CollectionError("출력 폴더가 이미 존재합니다. 다른 경로를 사용하세요.")
    tmp = Path(tempfile.mkdtemp(prefix=".bizinfo-", dir=destination.parent))
    try:
        fields = sorted({field for row in rows for field in row})
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        csv_bytes = output.getvalue().encode("utf-8-sig")
        (tmp / "announcements.csv").write_bytes(csv_bytes)
        hashes = {"announcements.csv": hashlib.sha256(csv_bytes).hexdigest()}
        for number, raw in enumerate(pages, 1):
            name = f"response_{number:04d}.json"
            (tmp / name).write_bytes(raw)
            hashes[name] = hashlib.sha256(raw).hexdigest()
        metadata = {
            "source": "기업마당 지원사업정보 API",
            "documentation_url": DOCUMENTATION_URL,
            "endpoint": API_URL,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "parameters": {
                "dataType": "json",
                "searchCnt": "0",
                "pageUnit": size,
                "hashtags": hashtags,
            },
            "row_count": len(rows),
            "page_count": len(pages),
            "sha256": hashes,
            "coverage": "최신 공고 스냅샷; 과거 전체 이력과 지역별 집행을 보장하지 않음",
            "usage_terms": "기업마당 직접 API 및 개별 공고의 이용·재배포 조건을 확인할 것",
        }
        (tmp / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        # Readers must require metadata.json as the completion marker.
        destination.mkdir()
        try:
            for child in sorted(tmp.iterdir(), key=lambda item: item.name == "metadata.json"):
                os.link(child, destination / child.name)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
        shutil.rmtree(tmp)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="새로 생성할 스냅샷 폴더")
    parser.add_argument(
        "--confirm-source-terms",
        action="store_true",
        help="기업마당 직접 API 및 개별 공고의 CSV 변환과 이용조건을 확인한 경우에만 지정",
    )
    parser.add_argument("--hashtags", help="기업마당 해시태그 필터. 미지정 시 전체")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        if not args.confirm_source_terms:
            raise CollectionError(
                "기업마당 직접 API와 개별 공고의 CSV 변환·이용조건을 확인한 뒤 "
                "--confirm-source-terms를 지정하세요."
            )
        rows, pages = collect(
            os.environ.get("BIZINFO_API_KEY", ""), args.page_size, args.max_pages, args.hashtags
        )
        save_snapshot(args.output_dir, rows, pages, args.page_size, args.hashtags)
    except (CollectionError, OSError) as exc:
        print(f"수집 실패: {exc}", file=sys.stderr)
        return 1
    print(f"수집 완료: {len(rows)}건, {len(pages)}페이지 → {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
