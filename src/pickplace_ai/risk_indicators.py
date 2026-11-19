"""관측 매출 변화와 폐업 부담을 독립 계산한다. 예측 확률을 만들지 않는다."""

from __future__ import annotations

from math import isfinite

from pickplace_ai.periods import shift_quarter
from pickplace_ai.schemas import Observation


def _ratio(numerator: float | int, denominator: float | int) -> float | None:
    try:
        value = numerator / denominator
    except (OverflowError, ZeroDivisionError):
        return None
    return value if isfinite(value) else None


def _unavailable(result: dict, reason: str) -> dict:
    return {**result, "status": "unavailable", "value": None, "reason": reason}


def observe_risk_indicators(history: list[Observation], base_period: str) -> dict:
    """API 검증을 통과한 단일 범위의 이력으로 현재 관측 지표를 반환한다.

    매출은 기준분기와 전년 동기만 필요하다. 폐업 부담은 기준분기의 점포와
    폐업 수만 사용하므로 다른 지표의 결측 때문에 함께 제공을 중단하지 않는다.
    """
    indexed = {row.period: row for row in history}
    current = indexed.get(base_period)
    comparison_period = shift_quarter(base_period, -4)
    comparison = indexed.get(comparison_period)
    current_sales = None if current is None else current.sales_amount
    comparison_sales = None if comparison is None else comparison.sales_amount
    stores = None if current is None else current.store_count
    closings = None if current is None else current.closing_count
    sales = {
        "status": "available",
        "value": None,
        "unit": "ratio",
        "reason": None,
        "base_period": base_period,
        "comparison_period": comparison_period,
        "current_sales_amount": current_sales,
        "comparison_sales_amount": comparison_sales,
    }
    closure = {
        "status": "available",
        "value": None,
        "unit": "ratio",
        "reason": None,
        "base_period": base_period,
        "closing_count": closings,
        "store_count": stores,
    }
    if current is None:
        return {
            "sales_yoy_change": _unavailable(sales, "missing_base_period"),
            "closure_burden": _unavailable(closure, "missing_base_period"),
        }

    if current_sales is None:
        sales = _unavailable(sales, "missing_current_sales")
    elif comparison_sales is None:
        sales = _unavailable(sales, "missing_comparison_sales")
    elif comparison_sales <= 0:
        sales = _unavailable(sales, "nonpositive_comparison_sales")
    else:
        ratio = _ratio(current_sales, comparison_sales)
        if ratio is None or not isfinite(ratio - 1):
            sales = _unavailable(sales, "nonfinite_derived_indicator")
        else:
            sales["value"] = ratio - 1

    if stores is None:
        closure = _unavailable(closure, "missing_store_count")
    elif stores <= 0:
        closure = _unavailable(closure, "nonpositive_store_count")
    elif closings is None:
        closure = _unavailable(closure, "missing_closing_count")
    elif closings > stores:
        # 이전 분기 점포 수로 바꾸거나 비율을 1로 잘라 집계 정의 차이를 숨기지 않는다.
        closure = _unavailable(closure, "closing_count_exceeds_store_count")
    else:
        ratio = _ratio(closings, stores)
        if ratio is None:
            closure = _unavailable(closure, "nonfinite_derived_indicator")
        else:
            closure["value"] = ratio

    return {"sales_yoy_change": sales, "closure_burden": closure}
