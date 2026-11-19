"""Guarded joint training must use inputs and retain all evaluation rows."""

import numpy as np
import pandas as pd
import pytest

from scripts.train_seoul_guard_filtered import evaluate, select_eligible_seoul


def test_selection_ignores_targets_and_matches_strict_policy():
    history = np.array(
        [[10, 10, 10, 10], [4, 4, 4, 4], [5, 5, 5, 5], [10, 10, 40, 40]],
        dtype=float,
    )
    frame = pd.DataFrame({"target": [0, 1, 10000, -1]})
    before = frame.copy(deep=True)
    mask = select_eligible_seoul(frame, history, (5, 2, -0.7, 5))
    np.testing.assert_array_equal(mask, [True, False, True, False])
    frame.loc[:, "target"] = [-1000, 100000, -1, 7]
    np.testing.assert_array_equal(select_eligible_seoul(frame, history, (5, 2, -0.7, 5)), mask)
    assert before.shape == frame.shape


def test_rejects_history_length_mismatch():
    with pytest.raises(ValueError, match="alignment"):
        select_eligible_seoul(pd.DataFrame({"target": [0, 1]}), np.ones((1, 4)), (1, 2, -0.7, 1))


def test_evaluate_all_groups_with_one_common_mask_and_gyeonggi_all():
    seoul = pd.DataFrame({"target": [0.1, -0.2, 0.3]})
    gyeonggi = pd.DataFrame({"target": [0.5, 1.0]})
    history = np.array([[10, 10, 10, 10], [1, 1, 1, 1], [10, 10, 10, 10]])
    models = {"zero": (None, lambda _, f: np.zeros(len(f)))}
    result = evaluate(models, seoul, gyeonggi, history, (5, 2, -0.7, 5))["zero"]
    assert result["seoul_guard_pass"]["rows"] == 2
    assert result["seoul_guard_withheld"]["rows"] == 1
    assert result["seoul_all"]["rows"] == 3
    assert result["gyeonggi_all"]["rows"] == 2
    assert result["seoul_guard_withheld"]["mae_pp"] == pytest.approx(20)


def test_empty_withheld_group_has_null_metrics():
    seoul = pd.DataFrame({"target": [0.1]})
    gyeonggi = pd.DataFrame({"target": [0.2]})
    history = np.array([[10, 10, 10, 10]])
    models = {"zero": (None, lambda _, f: np.zeros(len(f)))}
    result = evaluate(models, seoul, gyeonggi, history, (5, 2, -0.7, 5))["zero"]
    assert result["seoul_guard_withheld"]["rows"] == 0
    assert result["seoul_guard_withheld"]["mae_pp"] is None
