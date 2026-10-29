"""Quarter arithmetic without assuming contiguous rows are contiguous periods."""

import re
from datetime import date

PATTERN = re.compile(r"^(\d{4})Q([1-4])$")


def quarter_index(period: str) -> int:
    match = PATTERN.fullmatch(period)
    if match is None:
        raise ValueError(f"Invalid quarter: {period}; expected YYYYQ1..YYYYQ4")
    year, quarter = map(int, match.groups())
    if not 1900 <= year <= 2200:
        raise ValueError("Quarter year must be in 1900..2200")
    return year * 4 + quarter - 1


def shift_quarter(period: str, delta: int) -> str:
    index = quarter_index(period) + delta
    return f"{index // 4:04d}Q{index % 4 + 1}"


def quarter_end(period: str) -> date:
    index = quarter_index(period)
    year, quarter = index // 4, index % 4 + 1
    month = quarter * 3
    day = 31 if month in (3, 12) else 30
    return date(year, month, day)
