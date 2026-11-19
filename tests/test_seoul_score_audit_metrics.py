"""합성 자료로 진단지수 비교의 모집단과 단일 사건 처리만 검증한다."""

import numpy as np
import pandas as pd
import pytest

from scripts.seoul_score_audit_metrics import risk_outcomes


def _constructed_rows():
    return pd.DataFrame(
        {
            "industry_code": ["CS100008"] * 6,
            "period": ["2025Q4"] * 6,
            "risk_score": [50.0, 70.0, 30.0, 40.0, 99.0, 99.0],
            "sales_risk_points": [40.0, 40.0, 10.0, 0.0, np.nan, 50.0],
            "closure_risk_points": [10.0, 30.0, 20.0, 40.0, 49.0, 49.0],
            "sales_yoy": [-0.20, -0.20, -0.05, 0.10, -0.25, -0.25],
            "closure_ratio": [0.03, 0.09, 0.06, 0.12, 0.147, 0.147],
            "next_sales_yoy": [-0.25, -0.40, 0.00, 0.25, 0.00, np.nan],
            "next_closure_ratio": [0.20, 0.10, 0.15, 0.00, 0.30, np.nan],
        }
    )


def test_outcomes_compare_combined_and_components_on_identical_labeled_rows():
    result = risk_outcomes(_constructed_rows())
    for industry in ("ALL", "CS100008"):
        part = result.loc[result.industry_code == industry].set_index("outcome")
        assert set(part.risk_rows) == {6}
        assert set(part.comparable_current_rows) == {5}
        assert set(part.labeled_rows) == {4}
        assert set(part.event_rows) == {2}
        assert set(part.non_event_rows) == {2}
        assert set(part.unlabeled_risk_rows) == {2}
        sales = part.loc["next_sales_decline_25pct"]
        assert sales.risk_auc == pytest.approx(1.0)
        assert sales.sales_component_auc == pytest.approx(1.0)
        assert sales.closure_component_auc == pytest.approx(0.25)
        assert sales.raw_sales_decline_auc == pytest.approx(1.0)
        assert sales.raw_closure_ratio_auc == pytest.approx(0.25)
        closures = part.loc["next_closure_burden_15pct"]
        assert closures.risk_auc == pytest.approx(0.25)
        assert closures.sales_component_auc == pytest.approx(0.625)
        assert closures.closure_component_auc == pytest.approx(0.0)


def test_single_label_outcomes_keep_counts_and_return_unavailable_auc():
    frame = _constructed_rows().iloc[:2].copy()
    frame["next_sales_yoy"] = [-0.25, -0.40]
    frame["next_closure_ratio"] = [0.01, 0.10]
    result = risk_outcomes(frame)
    auc_columns = [name for name in result if name.endswith("_auc")]
    assert result[auc_columns].isna().all().all()
    assert set(result.labeled_rows) == {2}
    sales = result.loc[result.outcome == "next_sales_decline_25pct"]
    assert set(sales.event_rows) == {2}
    assert set(sales.non_event_rows) == {0}
    assert sales.non_event_risk_mean.isna().all()
    assert set(sales.event_risk_mean) == {60.0}
    closures = result.loc[result.outcome == "next_closure_burden_15pct"]
    assert set(closures.event_rows) == {0}
    assert set(closures.non_event_rows) == {2}
    assert closures.event_risk_mean.isna().all()
