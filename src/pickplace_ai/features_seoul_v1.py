"""Shared feature creation for both training and inference, respecting event time."""

from collections import defaultdict
from datetime import date
from math import nan

import numpy as np
import pandas as pd

from pickplace_ai.data import validate_observations
from pickplace_ai.periods import quarter_end, quarter_index, shift_quarter
from pickplace_ai.schemas import Observation

FEATURE_SCHEMA_VERSION = "1.0"
FEATURE_COLUMNS = [
    "sales_amount",
    "sales_lag1",
    "sales_lag4",
    "sales_yoy",
    "sales_qoq",
    "sales_mean4",
    "sales_cv4",
    "store_count",
    "opening_stock_ratio",
    "closing_stock_ratio",
    "resident_population",
    "working_population",
    "floating_population",
    "store_density",
    "resident_per_store",
    "quarter",
]


class FeatureUnavailable(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None or denominator <= 0:
        return nan
    return float(numerator / denominator)


def feature_record(
    history: list[Observation],
    base_period: str,
    as_of: date,
) -> dict[str, float]:
    """Use five *calendar* quarters; missing periods never become adjacent rows."""
    if not history:
        raise FeatureUnavailable("insufficient_history")
    scope = {(r.region_code_system, r.region_code, r.industry_code) for r in history}
    if len(scope) != 1:
        raise ValueError("Features require exactly one region-code system, region, and industry")
    periods = [r.period for r in history]
    if len(periods) != len(set(periods)):
        raise ValueError("Duplicate history quarter")
    eligible = {
        r.period: r
        for r in history
        if r.available_at <= as_of and quarter_index(r.period) <= quarter_index(base_period)
    }
    required = [shift_quarter(base_period, offset) for offset in range(-4, 1)]
    if any(period not in eligible for period in required):
        raise FeatureUnavailable("insufficient_history")
    records = [eligible[p] for p in required]
    if any(row.sales_amount is None for row in records):
        raise FeatureUnavailable("missing_sales")
    sales = np.array([row.sales_amount for row in records], dtype=float)
    if sales[0] <= 0 or sales[-2] <= 0 or sales[1] <= 0:
        raise FeatureUnavailable("invalid_sales_denominator")
    current = records[-1]
    mean4 = float(np.mean(sales[-4:]))
    values = {
        "sales_amount": float(sales[-1]),
        "sales_lag1": float(sales[-2]),
        "sales_lag4": float(sales[0]),
        "sales_yoy": float(sales[-1] / sales[0] - 1),
        "sales_qoq": float(sales[-1] / sales[-2] - 1),
        "sales_mean4": mean4,
        "sales_cv4": float(np.std(sales[-4:]) / mean4) if mean4 > 0 else nan,
        "store_count": float(current.store_count) if current.store_count is not None else nan,
        "opening_stock_ratio": _ratio(current.opening_count, current.store_count),
        "closing_stock_ratio": _ratio(current.closing_count, current.store_count),
        "resident_population": (
            float(current.resident_population) if current.resident_population is not None else nan
        ),
        "working_population": (
            float(current.working_population) if current.working_population is not None else nan
        ),
        "floating_population": (
            float(current.floating_population) if current.floating_population is not None else nan
        ),
        "store_density": _ratio(current.store_count, current.area_km2),
        "resident_per_store": _ratio(current.resident_population, current.store_count),
        "quarter": float(quarter_index(base_period) % 4 + 1),
    }
    return {name: values[name] for name in FEATURE_COLUMNS}


def build_training_frame(observations: list[Observation]) -> pd.DataFrame:
    validate_observations(observations)
    groups = defaultdict(list)
    for row in observations:
        groups[(row.region_code_system, row.region_code, row.industry_code)].append(row)
    prepared = []
    for history in groups.values():
        indexed = {r.period: r for r in history}
        for current in sorted(history, key=lambda r: quarter_index(r.period)):
            target_period = shift_quarter(current.period, 1)
            # A newly downloaded historical snapshot is not evidence that it was
            # available before the old target ended. Never backdate availability.
            if current.available_at >= quarter_end(target_period):
                continue
            label = indexed.get(target_period)
            comparison = indexed.get(shift_quarter(current.period, -3))
            if label is None or comparison is None:
                continue
            if label.sales_amount is None or comparison.sales_amount is None:
                continue
            if comparison.sales_amount <= 0:
                continue
            try:
                values = feature_record(history, current.period, current.available_at)
            except FeatureUnavailable:
                continue
            prepared.append(
                {
                    **values,
                    "target": float(label.sales_amount / comparison.sales_amount - 1),
                    "region_code": current.region_code,
                    "region_code_system": current.region_code_system,
                    "industry_code": current.industry_code,
                    "source_kind": current.source_kind,
                    "base_period": current.period,
                    "target_period": target_period,
                    "as_of": current.available_at,
                    "label_available_at": label.available_at,
                }
            )
    columns = FEATURE_COLUMNS + [
        "target",
        "region_code",
        "region_code_system",
        "industry_code",
        "source_kind",
        "base_period",
        "target_period",
        "as_of",
        "label_available_at",
    ]
    if not prepared:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(prepared, columns=columns)
        .sort_values(["target_period", "region_code_system", "region_code", "industry_code"])
        .reset_index(drop=True)
    )
