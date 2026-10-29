import json

from pickplace_ai.demo import synthetic_observations
from pickplace_ai.scoring import diagnose


def test_missing_components_return_null_instead_of_zero():
    rows = synthetic_observations(regions=1, quarters=10)
    last = rows[-1].model_copy(update={"area_km2": None, "closing_count": None})
    result = diagnose(rows[:-1] + [last], last.period)
    assert result["risk"]["status"] == "unavailable"
    assert result["risk"]["score"] is None
    assert result["fit"]["status"] == "unavailable"
    assert result["fit"]["score"] is None


def test_draft_scoring_is_bounded_and_components_sum_to_score():
    rows = synthetic_observations(regions=1, quarters=10)
    result = diagnose(rows, rows[-1].period)
    for name in ("risk", "fit"):
        assert 0 <= result[name]["score"] <= 100
        assert result[name]["policy_status"] == "draft"
        assert (
            abs(
                sum(c["contribution_points"] for c in result[name]["components"])
                - result[name]["score"]
            )
            < 0.02
        )


def test_higher_closings_increase_risk():
    rows = synthetic_observations(regions=1, quarters=10)
    before = diagnose(rows, rows[-1].period)["risk"]["score"]
    rows[-1] = rows[-1].model_copy(update={"closing_count": rows[-1].store_count})
    assert diagnose(rows, rows[-1].period)["risk"]["score"] >= before


def test_finite_input_overflow_yields_unavailable_serializable_result():
    rows = synthetic_observations(regions=1, quarters=10)
    rows[-1] = rows[-1].model_copy(update={"area_km2": 1e-308})
    result = diagnose(rows, rows[-1].period)
    assert result["fit"]["score"] is None
    assert result["fit"]["reason"] == "nonfinite_derived_indicator"
    json.dumps(result, allow_nan=False)
