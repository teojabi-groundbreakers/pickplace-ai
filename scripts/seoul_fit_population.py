"""행안부 월간 인구·세대 원본에서 서울 동별 인구를 H10 그대로 읽는다."""

import calendar
import csv
import io
import re
from datetime import date
from pathlib import Path

OFFICIAL_SOURCE = "https://jumin.mois.go.kr/statMonth.do"
_LABEL = re.compile(r"^(?P<name>[^()]+?)\s*\((?P<code>[0-9]{10})\)$")
_MONTH_COLUMN = re.compile(r"^(?P<year>[0-9]{4})년(?P<month>[0-9]{2})월_.+$")
_INTEGER = re.compile(r"(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)$")
_DISTRICT = re.compile(r"11[0-9]{3}00000$")
_DONG = re.compile(r"11[0-9]{6}00$")


def _month_end(reference_date: date | str) -> date:
    if isinstance(reference_date, str):
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", reference_date):
            raise ValueError("Expected an ISO month-end reference date")
        reference_date = date.fromisoformat(reference_date)
    if type(reference_date) is not date:
        raise ValueError("Expected a month-end reference date")
    if reference_date.day != calendar.monthrange(reference_date.year, reference_date.month)[1]:
        raise ValueError("MOIS monthly population must use the final day of its month")
    return reference_date


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError("Population CSV is not valid UTF-8 or CP949")


def _integer(value: str) -> int:
    value = value.strip()
    if not _INTEGER.fullmatch(value):
        raise ValueError("Population must be a nonnegative integer with valid comma grouping")
    return int(value.replace(",", ""))


def read_mois_population(path: Path, reference_date: date | str) -> dict[str, int]:
    """다운로드 원문은 보존하고 서울 동만 반환한다. H10을 다른 코드로 변환하지 않는다.

    대상 형식은 공식 statMonth의 `전체읍면동현황` CSV다. `행정구역`은
    `지역명 (10자리 행정기관코드)`, 총인구 열은 `YYYY년MM월_총인구수`다.
    서울 시 합계는 `1100000000`, 구 합계는 `11xxx00000`, 동은
    `11xxxYYY00`(YYY != 000)로 구분하되 이름 깊이와 상위 구도 함께 대조한다.
    이 접두·접미 검사는 원문 내 계층 검증이며 다른 코드 체계로의 연계가 아니다.
    """
    reference = _month_end(reference_date)
    rows = csv.reader(io.StringIO(_decode(Path(path).read_bytes()), newline=""), strict=True)
    try:
        header = next(rows)
    except StopIteration as exc:
        raise ValueError("Population CSV is empty") from exc
    if not header or header[0] != "행정구역" or len(set(header)) != len(header):
        raise ValueError("Unexpected or duplicate population CSV columns")
    expected_month = (reference.year, reference.month)
    for column in header[1:]:
        match = _MONTH_COLUMN.fullmatch(column)
        if match is None or (int(match["year"]), int(match["month"])) != expected_month:
            raise ValueError("Population CSV contains a different or ambiguous reference month")
    total_column = f"{reference.year:04d}년{reference.month:02d}월_총인구수"
    if total_column not in header:
        raise ValueError("Population CSV has no total population for the reference month")
    total_index = header.index(total_column)
    seen_codes: set[str] = set()
    seen_seoul_names: set[str] = set()
    city: int | None = None
    districts: dict[str, tuple[str, int]] = {}
    dongs: dict[str, tuple[str, int]] = {}
    try:
        for row in rows:
            if len(row) != len(header):
                raise ValueError("Population CSV contains a blank or malformed row")
            label = _LABEL.fullmatch(row[0].strip())
            if label is None:
                raise ValueError("Expected an explicit 10-digit administrative code in each label")
            code = label["code"]
            if code in seen_codes:
                raise ValueError("Duplicate administrative code in the monthly population CSV")
            seen_codes.add(code)
            population = _integer(row[total_index])
            parts = label["name"].split()
            if code.startswith("11") != (parts[0] == "서울특별시"):
                raise ValueError("Seoul code and region name disagree")
            if not code.startswith("11"):
                continue
            name = " ".join(parts)
            if name in seen_seoul_names:
                raise ValueError("Conflicting administrative codes for the same Seoul region name")
            seen_seoul_names.add(name)
            if code == "1100000000":
                if parts != ["서울특별시"]:
                    raise ValueError("Seoul city aggregate has an inconsistent name")
                city = population
            elif _DISTRICT.fullmatch(code):
                if len(parts) != 2 or not parts[1].endswith("구"):
                    raise ValueError("District aggregate code and name depth disagree")
                districts[code[:5]] = (parts[1], population)
            elif _DONG.fullmatch(code) and code[5:8] != "000":
                if len(parts) != 3 or not parts[1].endswith("구") or not parts[2].endswith("동"):
                    raise ValueError("Dong administrative code and name depth disagree")
                dongs[code] = (parts[1], population)
            else:
                raise ValueError("Unexpected Seoul administrative code suffix")
    except csv.Error as exc:
        raise ValueError("Malformed population CSV") from exc
    if city is None or not districts or not dongs:
        raise ValueError("Seoul city, district aggregates and dong records are required")
    district_sums = dict.fromkeys(districts, 0)
    for code, (district_name, population) in dongs.items():
        if code[:5] not in districts or districts[code[:5]][0] != district_name:
            raise ValueError("Dong parent code and district name disagree")
        district_sums[code[:5]] += population
    if any(district_sums[code] != record[1] for code, record in districts.items()):
        raise ValueError("District totals do not equal the included dong population")
    if sum(record[1] for record in districts.values()) != city:
        raise ValueError("Seoul city total does not equal its district totals")
    return {code: population for code, (_, population) in sorted(dongs.items())}
