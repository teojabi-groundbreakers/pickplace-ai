"""Small synthetic guards for pre-prediction signal boundaries and inputs."""

import numpy as np
import pytest

from scripts.audit_seoul_guard import guard_flags, minimum_growth, training_thresholds


def test_four_signals_are_strict_at_the_boundary():
    stable = np.array([[100.0, 100.0, 100.0, 100.0]])
    flags = guard_flags(stable, low=100.0, rise=0.0, fall=0.0, latest_low=100.0)
    assert not any(bool(value[0]) for value in flags.values())
    assert guard_flags(stable, 101.0, 1.0, -1.0, 0.0)["low_reference"][0]
    assert guard_flags(stable, 0.0, 1.0, -1.0, 101.0)["low_latest"][0]

    increase = np.array([[100.0, 100.0, 100.0, 200.0]])
    assert not guard_flags(increase, 0.0, 1.0, -1.0, 0.0)["rapid_increase"][0]
    assert guard_flags(increase, 0.0, np.nextafter(1.0, -np.inf), -1.0, 0.0)["rapid_increase"][0]

    decrease = np.array([[100.0, 100.0, 100.0, 50.0]])
    assert not guard_flags(decrease, 0.0, 1.0, -0.5, 0.0)["rapid_decrease"][0]
    assert guard_flags(decrease, 0.0, 1.0, np.nextafter(-0.5, np.inf), 0.0)["rapid_decrease"][0]


def test_target_changes_do_not_affect_training_thresholds_or_signals():
    histories = np.array([[100.0, 200.0, 100.0, 50.0], [200.0, 220.0, 240.0, 260.0]])
    targets_a = np.array([100.0, 500.0])
    targets_b = np.array([99999.0, 1.0])
    inputs_a = np.column_stack([histories, targets_a])[:, :4]
    inputs_b = np.column_stack([histories, targets_b])[:, :4]
    bounds_a, bounds_b = training_thresholds(inputs_a), training_thresholds(inputs_b)
    np.testing.assert_allclose(bounds_a, bounds_b)
    for key, actual in guard_flags(inputs_a, *bounds_a).items():
        np.testing.assert_array_equal(actual, guard_flags(inputs_b, *bounds_b)[key])


def test_zero_last_quarter_is_valid_and_positive_denominators_are_required():
    assert minimum_growth(np.array([[10.0, 10.0, 10.0, 0.0]])).tolist() == [-1.0]
    with pytest.raises(ValueError):
        guard_flags(np.array([[10.0, 0.0, 10.0, 10.0]]), 1.0, 1.0, -0.5, 1.0)
