"""Fixed Seoul research withholding policy for the Nov 2026 pooled model.

Thresholds are exploratory. They indicate input risk, not a proven data error.
The caller must verify compatibility with the specific trusted research artifact.
"""

from __future__ import annotations

import math

import numpy as np

from pickplace_ai.periods import quarter_index

POLICY_VERSION = "seoul-sales-history-20261109-v1"
LOW_REFERENCE_SALES_KRW = 10541122.800000003
LOW_LATEST_SALES_KRW = 8101054.100000001
MAX_QOQ_INCREASE = 2.066570717884198
MIN_QOQ_DECREASE = -0.7278775244727316
SOURCE_SHA256 = "20e87aa7a90468a4ef98a76dbd9acb90b59de2805e7478fd2462e5d60d9986d5"
EVALUATION_MODEL_SHA256 = "6ebf924808cf1c5c2b2c575e81fd7600ffec42711d8d623b76340bd19ab42581"
FINAL_MODEL_SHA256 = "6ca3340c68029f9659e280d079905d4bf816af765785a91ed8dd27949ca812cb"
SIGNALS = {
    "low_reference_sales": "전년 동기 매출이 연구용 저매출 기준보다 낮습니다.",
    "low_latest_sales": "최근 분기 매출이 연구용 저매출 기준보다 낮습니다.",
    "rapid_sales_increase": "최근 분기 매출 증가율이 연구용 급증 기준을 초과했습니다.",
    "rapid_sales_decrease": "최근 분기 매출 감소율이 연구용 급감 기준을 초과했습니다.",
}


def policy_compatible(metadata: dict, model_hash: str) -> bool:
    """Refuse to reuse Seoul thresholds with a different model or source snapshot."""
    return (
        model_hash == FINAL_MODEL_SHA256
        and metadata.get("source_hashes", {}).get("seoul") == SOURCE_SHA256
        and metadata.get("model_sha256", {}).get("evaluation-pooled.joblib")
        == EVALUATION_MODEL_SHA256
    )


def assess_sales_history(values: list[float]) -> list[str]:
    """Return all triggered signals using only four pre-target input quarters."""
    x = np.asarray(values, dtype=float)
    if x.shape != (4,) or not np.isfinite(x).all() or (x < 0).any() or (x[:3] <= 0).any():
        raise ValueError("Expected four finite nonnegative sales with positive denominators")
    signals = []
    if x[0] < LOW_REFERENCE_SALES_KRW:
        signals.append("low_reference_sales")
    if x[-1] < LOW_LATEST_SALES_KRW:
        signals.append("low_latest_sales")
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        growth = x[1:] / x[:-1] - 1
    if not all(math.isfinite(float(value)) for value in growth):
        raise ValueError("Nonfinite growth")
    if max(growth) > MAX_QOQ_INCREASE:
        signals.append("rapid_sales_increase")
    if min(growth) < MIN_QOQ_DECREASE:
        signals.append("rapid_sales_decrease")
    return signals


def describe_sales_history(values: list[float], periods: list[str]) -> list[dict]:
    """Explain triggered input signals without using or predicting a target value.

    For a change signal, identify the most extreme consecutive transition;
    ties use the earliest transition. The input unit remains caller asserted.
    """
    if len(periods) != 4 or any(not isinstance(period, str) for period in periods):
        raise ValueError("Expected four periods")
    indexes = [quarter_index(period) for period in periods]
    if indexes != list(range(indexes[0], indexes[0] + 4)):
        raise ValueError("Expected four consecutive chronological quarters")
    signals = assess_sales_history(values)
    x = np.asarray(values, dtype=float)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        growth = x[1:] / x[:-1] - 1
    details = []
    for signal in signals:
        if signal == "low_reference_sales":
            details.append(
                {
                    "signal": signal,
                    "from_period": None,
                    "to_period": periods[0],
                    "observed_value": float(x[0]),
                    "threshold_value": LOW_REFERENCE_SALES_KRW,
                    "unit": "KRW",
                    "comparison": "below",
                }
            )
        elif signal == "low_latest_sales":
            details.append(
                {
                    "signal": signal,
                    "from_period": None,
                    "to_period": periods[-1],
                    "observed_value": float(x[-1]),
                    "threshold_value": LOW_LATEST_SALES_KRW,
                    "unit": "KRW",
                    "comparison": "below",
                }
            )
        elif signal in {"rapid_sales_increase", "rapid_sales_decrease"}:
            increase = signal == "rapid_sales_increase"
            index = int(np.argmax(growth) if increase else np.argmin(growth))
            details.append(
                {
                    "signal": signal,
                    "from_period": periods[index],
                    "to_period": periods[index + 1],
                    "observed_value": float(growth[index]),
                    "threshold_value": MAX_QOQ_INCREASE if increase else MIN_QOQ_DECREASE,
                    "unit": "ratio",
                    "comparison": "above" if increase else "below",
                }
            )
        else:
            raise ValueError("Unsupported sales history signal")
    return details
