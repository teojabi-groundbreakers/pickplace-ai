"""초안 진단지수의 분포와 관측 대용 결과를 집계한다. 모델 학습은 하지 않는다."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _require(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing scoring audit columns: {sorted(missing)}")


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _groups(frame: pd.DataFrame):
    yield "ALL", frame
    yield from frame.groupby("industry_code", sort=True, dropna=False)


def _period_fields(frame: pd.DataFrame) -> dict:
    periods = sorted(frame.period.dropna().astype(str).unique())
    return {
        "period": periods[0] if len(periods) == 1 else None,
        "period_count": len(periods),
        "period_min": periods[0] if periods else None,
        "period_max": periods[-1] if periods else None,
    }


def _fraction(condition: pd.Series, eligible: pd.Series) -> float | None:
    count = int(eligible.sum())
    return float(condition.loc[eligible].sum() / count) if count else None


def _score_summary(score: pd.Series, prefix: str) -> dict:
    observed = score.dropna()
    if ((observed < 0) | (observed > 100)).any():
        raise ValueError("Draft scores must be between 0 and 100")
    result = {
        f"{prefix}_zero_fraction": float((observed == 0).mean()) if len(observed) else None,
        f"{prefix}_saturated_fraction": (
            float((observed == 100).mean()) if len(observed) else None
        ),
        f"{prefix}_p10": float(observed.quantile(0.1)) if len(observed) else None,
        f"{prefix}_median": float(observed.median()) if len(observed) else None,
        f"{prefix}_p90": float(observed.quantile(0.9)) if len(observed) else None,
        f"{prefix}_unique_scores": int(observed.nunique()),
        f"{prefix}_10bin_entropy_bits": None,
    }
    if len(observed):
        # 고정 폭의 분포만 측정한다. 이 값을 예측력이나 보정된 확률로 해석하지 않는다.
        counts, _ = np.histogram(observed.to_numpy(float), bins=np.linspace(0, 100, 11))
        proportions = counts[counts > 0] / len(observed)
        result[f"{prefix}_10bin_entropy_bits"] = float(-np.sum(proportions * np.log2(proportions)))
    return result


def _median(values: pd.Series) -> float | None:
    observed = values.dropna()
    return float(observed.median()) if len(observed) else None


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    valid = left.notna() & right.notna()
    a, b = left.loc[valid], right.loc[valid]
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return None
    value = a.rank(method="average").corr(b.rank(method="average"))
    return float(value) if np.isfinite(value) else None


def distributions(frame: pd.DataFrame) -> pd.DataFrame:
    """업종별·전체 분포. 점수는 0~100점, 비율은 0~1로 반환한다.

    전달된 관측만 사용하며 이전 분기의 존재를 강제하지 않는다. 성분별
    결측이 다르면 분모를 별도 rows 열로 남겨 숨은 재가중을 피한다.
    """
    _require(frame, {"industry_code", "period", "risk_score", "fit_score", "has_sales"})
    rows = []
    for industry, part in _groups(frame):
        risk, fit = _numeric(part, "risk_score"), _numeric(part, "fit_score")
        yoy, closure = _numeric(part, "sales_yoy"), _numeric(part, "closure_ratio")
        demand = _numeric(part, "residents_per_store")
        density = _numeric(part, "stores_per_km2")
        sales = _numeric(part, "sales_per_store")
        has_sales = part.has_sales.fillna(False).eq(True)
        risk_sales = risk.notna() & yoy.notna()
        risk_closures = risk.notna() & closure.notna()
        fit_demand = fit.notna() & demand.notna()
        fit_competition = fit.notna() & density.notna()
        fit_both = fit_demand & fit_competition
        linked = fit.notna() & sales.notna() & has_sales
        rows.append(
            {
                "industry_code": industry,
                **_period_fields(part),
                "rows": len(part),
                "sales_rows": int(has_sales.sum()),
                "risk_rows": int(risk.notna().sum()),
                "fit_rows": int(fit.notna().sum()),
                **_score_summary(risk, "risk"),
                **_score_summary(fit, "fit"),
                "risk_sales_rows": int(risk_sales.sum()),
                "risk_closures_rows": int(risk_closures.sum()),
                "fit_demand_rows": int(fit_demand.sum()),
                "fit_competition_rows": int(fit_competition.sum()),
                "fit_both_component_rows": int(fit_both.sum()),
                "risk_sales_zero_fraction": _fraction(yoy >= 0, risk_sales),
                "risk_sales_saturated_fraction": _fraction(yoy <= -0.25, risk_sales),
                "risk_closures_zero_fraction": _fraction(closure == 0, risk_closures),
                "risk_closures_saturated_fraction": _fraction(closure >= 0.15, risk_closures),
                "fit_demand_saturated_fraction": _fraction(demand >= 500, fit_demand),
                "fit_competition_zero_fraction": _fraction(density >= 50, fit_competition),
                "fit_both_clipped_fraction": _fraction((demand >= 500) & (density >= 50), fit_both),
                "closure_ratio_median": _median(closure.where(risk.notna())),
                "residents_per_store_median": _median(demand.where(fit.notna())),
                "stores_per_km2_median": _median(density.where(fit.notna())),
                "fit_sales_per_store_rows": int(linked.sum()),
                "fit_sales_per_store_spearman": _spearman(fit.where(linked), sales.where(linked)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _auc(labels: pd.Series, score: pd.Series) -> float | None:
    if labels.nunique() < 2:
        return None
    return float(roc_auc_score(labels.astype(int), score))


def risk_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """같은 현재 입력 모집단에서 두 다음 분기 관측 사건을 비교한다.

    미래 관측은 사건 라벨로만 사용한다. 매출·폐업 단독 성분과 종합점수의
    AUC는 같은 행으로 계산하며, 한 종류의 라벨만 있으면 null로 남긴다.
    """
    _require(
        frame,
        {
            "industry_code",
            "period",
            "risk_score",
            "sales_risk_points",
            "closure_risk_points",
            "sales_yoy",
            "closure_ratio",
            "next_sales_yoy",
            "next_closure_ratio",
        },
    )
    rows = []
    for industry, part in _groups(frame):
        scores = {
            "risk_auc": _numeric(part, "risk_score"),
            "sales_component_auc": _numeric(part, "sales_risk_points"),
            "closure_component_auc": _numeric(part, "closure_risk_points"),
            "raw_sales_decline_auc": -_numeric(part, "sales_yoy"),
            "raw_closure_ratio_auc": _numeric(part, "closure_ratio"),
        }
        current = pd.concat(scores.values(), axis=1).notna().all(axis=1)
        risk = scores["risk_auc"]
        risk_rows = int(risk.notna().sum())
        events = (
            ("next_sales_decline_25pct", "next_sales_yoy", "le", -0.25),
            ("next_closure_burden_15pct", "next_closure_ratio", "ge", 0.15),
        )
        for event, column, comparison, threshold in events:
            measure = _numeric(part, column)
            eligible = current & measure.notna()
            values = measure.loc[eligible]
            labels = values.le(threshold) if comparison == "le" else values.ge(threshold)
            point = risk.loc[eligible]
            events_n = int(labels.sum())
            non_events_n = len(labels) - events_n
            rows.append(
                {
                    "industry_code": industry,
                    **_period_fields(part),
                    "outcome": event,
                    "outcome_column": column,
                    "event_comparison": comparison,
                    "event_threshold": threshold,
                    "rows": len(part),
                    "risk_rows": risk_rows,
                    "comparable_current_rows": int(current.sum()),
                    "labeled_rows": len(labels),
                    "unlabeled_risk_rows": risk_rows - len(labels),
                    "event_rows": events_n,
                    "non_event_rows": non_events_n,
                    "event_fraction": events_n / len(labels) if len(labels) else None,
                    **{name: _auc(labels, score.loc[eligible]) for name, score in scores.items()},
                    "event_risk_mean": float(point.loc[labels].mean()) if events_n else None,
                    "non_event_risk_mean": (
                        float(point.loc[~labels].mean()) if non_events_n else None
                    ),
                }
            )
    return pd.DataFrame.from_records(rows)
