"""Versioned draft diagnostics; numbers are rule indices, never success probabilities."""

from math import isfinite

from pickplace_ai.periods import shift_quarter
from pickplace_ai.schemas import Observation

POLICY_VERSION = "draft-v0.1"


def _bounded(value: float) -> float:
    return min(100.0, max(0.0, value))


def _finite_ratio(numerator: float | int, denominator: float | int) -> float | None:
    try:
        value = numerator / denominator
    except (OverflowError, ZeroDivisionError):
        return None
    return value if isfinite(value) else None


def _unavailable(reason: str) -> dict:
    return {
        "status": "unavailable",
        "score": None,
        "reason": reason,
        "policy_status": "draft",
        "components": [],
    }


def diagnose(history: list[Observation], base_period: str) -> dict:
    """Fixed, disclosed component sets: no hidden reweighting when data is missing."""
    indexed = {r.period: r for r in history}
    current = indexed.get(base_period)
    if current is None:
        return {
            "risk": _unavailable("missing_base_period"),
            "fit": _unavailable("missing_base_period"),
            "policy_version": POLICY_VERSION,
            "warnings": ["draft_scoring_policy"],
        }
    risk = _unavailable("missing_risk_indicators")
    fit = _unavailable("missing_fit_indicators")
    previous_year = indexed.get(shift_quarter(base_period, -4))
    stores = current.store_count
    if (
        stores is not None
        and stores > 0
        and current.closing_count is not None
        and current.sales_amount is not None
        and previous_year is not None
        and previous_year.sales_amount is not None
        and previous_year.sales_amount > 0
    ):
        sales_ratio = _finite_ratio(current.sales_amount, previous_year.sales_amount)
        closure_ratio = _finite_ratio(current.closing_count, stores)
        if sales_ratio is None or closure_ratio is None:
            risk = _unavailable("nonfinite_derived_indicator")
        else:
            yoy = sales_ratio - 1
            sales_risk = _bounded(-yoy / 0.25 * 100)
            closure_risk = _bounded(closure_ratio / 0.15 * 100)
            risk = {
                "status": "available",
                "score": round((sales_risk + closure_risk) / 2, 2),
                "reason": None,
                "policy_status": "draft",
                "components": [
                    {
                        "name": "sales_decline",
                        "value": yoy,
                        "unit": "ratio",
                        "contribution_points": round(sales_risk / 2, 4),
                    },
                    {
                        "name": "closings_per_current_store",
                        "value": closure_ratio,
                        "unit": "ratio",
                        "contribution_points": round(closure_risk / 2, 4),
                    },
                ],
            }
    if (
        stores is not None
        and stores > 0
        and current.resident_population is not None
        and current.area_km2 is not None
    ):
        residents_per_store = _finite_ratio(current.resident_population, stores)
        density = _finite_ratio(stores, current.area_km2)
        if residents_per_store is None or density is None:
            fit = _unavailable("nonfinite_derived_indicator")
        else:
            demand_score = _bounded(residents_per_store / 500 * 100)
            competition_score = _bounded((1 - density / 50) * 100)
            fit = {
                "status": "available",
                "score": round((demand_score + competition_score) / 2, 2),
                "reason": None,
                "policy_status": "draft",
                "components": [
                    {
                        "name": "residents_per_store",
                        "value": residents_per_store,
                        "unit": "persons",
                        "contribution_points": round(demand_score / 2, 4),
                    },
                    {
                        "name": "store_density",
                        "value": density,
                        "unit": "stores_per_km2",
                        "contribution_points": round(competition_score / 2, 4),
                    },
                ],
            }
    return {
        "risk": risk,
        "fit": fit,
        "policy_version": POLICY_VERSION,
        "warnings": [
            "draft_scoring_policy",
            "indices_are_not_probabilities",
            "risk_closure_ratio_uses_current_stock_not_official_closure_rate",
            "fit_is_population_and_competition_only_not_validated_business_suitability",
        ],
    }
