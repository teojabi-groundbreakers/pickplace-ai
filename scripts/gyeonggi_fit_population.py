"""행안부 월간 원본에서 경기 읍면동급 주민등록인구를 H10 그대로 읽는다."""

import csv
import io
import re
from datetime import date
from pathlib import Path

from scripts.seoul_fit_population import _LABEL, _MONTH_COLUMN, _decode, _integer, _month_end

_PROVINCE = "4100000000"
_AGGREGATE = re.compile(r"41[0-9]{3}00000$")
_LOCAL = re.compile(r"41[0-9]{6}00$")


def _records(path: Path, reference_date: date | str) -> list[tuple[str, list[str], int]]:
    """서울 파서와 같은 월말·인코딩·정수 규칙으로 원문을 검증한다."""
    reference = _month_end(reference_date)
    reader = csv.reader(io.StringIO(_decode(Path(path).read_bytes()), newline=""), strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise ValueError("Population CSV is empty or malformed") from exc
    if not header or header[0] != "행정구역" or len(set(header)) != len(header):
        raise ValueError("Unexpected or duplicate population CSV columns")
    for column in header[1:]:
        match = _MONTH_COLUMN.fullmatch(column)
        if match is None or (int(match["year"]), int(match["month"])) != (
            reference.year,
            reference.month,
        ):
            raise ValueError("Population CSV contains a different or ambiguous reference month")
    total_column = f"{reference.year:04d}년{reference.month:02d}월_총인구수"
    if total_column not in header:
        raise ValueError("Population CSV has no total population for the reference month")
    total_index = header.index(total_column)
    seen_codes: set[str] = set()
    seen_names: set[str] = set()
    result: list[tuple[str, list[str], int]] = []
    try:
        for row in reader:
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
            if code.startswith("41") != (parts[0] == "경기도"):
                raise ValueError("Gyeonggi code and region name disagree")
            if not code.startswith("41"):
                continue
            name = " ".join(parts)
            if name in seen_names:
                raise ValueError("Conflicting administrative codes for the same Gyeonggi name")
            seen_names.add(name)
            result.append((code, parts, population))
    except csv.Error as exc:
        raise ValueError("Malformed population CSV") from exc
    return result


def read_mois_gyeonggi_population(path: Path, reference_date: date | str) -> dict[str, int]:
    """statMonth `전체읍면동현황`의 경기 읍면동급 H10과 총인구를 반환한다.

    시군·일반구·도 합계는 반환에서 제외하되 하위 합계와 대조한다. 상위
    출장소 합계와 화도읍 동부·남부 출장소 같은 읍면동급 레코드는 코드의
    집계 접미부와 이름 깊이로 구분한다. 읍면동급 출장소의 인구도 포함한다.
    숫자 0은 유효한 관측이며 빈 값은 허용하지 않는다. 이 검사는 원문 내부
    계층만 확인하며 외부 코드표와의 연계·과거 경계 동일성을 승인하지 않는다.
    """
    aggregates: dict[str, tuple[list[str], int, str]] = {}
    locals_: dict[str, tuple[list[str], int]] = {}
    for code, parts, population in _records(path, reference_date):
        if code == _PROVINCE:
            if parts != ["경기도"]:
                raise ValueError("Gyeonggi province aggregate has an inconsistent name")
            aggregates[code] = (parts, population, "province")
        elif _AGGREGATE.fullmatch(code):
            if len(parts) == 2 and parts[1].endswith(("시", "군")) and code[4] == "0":
                kind = "municipality"
            elif len(parts) == 3 and parts[2].endswith("구") and code[4] != "0":
                kind = "district"
            elif len(parts) == 2 and parts[1].endswith("출장소") and code[4] != "0":
                kind = "branch"
            else:
                raise ValueError("Gyeonggi aggregate code and name depth disagree")
            aggregates[code] = (parts, population, kind)
        elif _LOCAL.fullmatch(code) and code[5:8] != "000":
            if len(parts) not in (3, 4) or not parts[-1].endswith(("읍", "면", "동", "출장소")):
                raise ValueError("Gyeonggi local code and name depth disagree")
            locals_[code] = (parts, population)
        else:
            raise ValueError("Unexpected Gyeonggi administrative code suffix")
    municipalities = {code for code, (_, _, kind) in aggregates.items() if kind == "municipality"}
    if _PROVINCE not in aggregates or not municipalities or not locals_:
        raise ValueError("Gyeonggi province, municipality and local records are required")

    # 코드 접두 관계는 원문 안의 부모 검증에만 사용하며 외부 코드로 바꾸지 않는다.
    child_sums = dict.fromkeys(aggregates, 0)
    child_counts = dict.fromkeys(aggregates, 0)
    for code, (parts, population) in locals_.items():
        parent = code[:5] + "00000"
        if parent not in aggregates or aggregates[parent][0] != parts[:-1]:
            raise ValueError("Local parent code and aggregate name disagree")
        if aggregates[parent][2] == "province":
            raise ValueError("Local code cannot be a direct child of the province")
        child_sums[parent] += population
        child_counts[parent] += 1
    for code, (parts, population, kind) in aggregates.items():
        if kind in ("province", "municipality"):
            continue
        parent = code[:4] + "000000"
        if parent not in municipalities:
            raise ValueError("District or branch has no matching municipality parent")
        if kind == "district" and aggregates[parent][0] != parts[:-1]:
            raise ValueError("District parent code and municipality name disagree")
        child_sums[parent] += population
        child_counts[parent] += 1
    for code, (_, population, kind) in aggregates.items():
        if kind == "province":
            continue
        if not child_counts[code] and kind != "branch":
            raise ValueError("Gyeonggi aggregate has no included child records")
        if child_sums[code] != population:
            raise ValueError("Gyeonggi aggregate totals do not equal included child population")
    if sum(aggregates[code][1] for code in municipalities) != aggregates[_PROVINCE][1]:
        raise ValueError("Gyeonggi province total does not equal municipality totals")
    return {code: population for code, (_, population) in sorted(locals_.items())}
