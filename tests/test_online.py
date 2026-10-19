"""
Unit tests for online adaptive alpha allocation.
"""

import pytest
import numpy as np
from optialpha.combination.online import ExponentiatedGradientCombiner


def test_exponentiated_gradient_initialization():
    eg = ExponentiatedGradientCombiner(n_factors=4)
    w0 = eg.get_weights()

    assert len(w0) == 4
    np.testing.assert_allclose(w0, [0.25, 0.25, 0.25, 0.25])
    assert len(eg.get_history()) == 1


def test_exponentiated_gradient_updates():
    eg = ExponentiatedGradientCombiner(n_factors=3, learning_rate=0.5, exploration=0.0)

    # Reward factor 0 heavily
    rewards = np.array([2.0, -1.0, 0.0])
    w1 = eg.update(rewards)

    assert np.isclose(np.sum(w1), 1.0)
    assert np.all(w1 >= 0.0)
    # Factor 0 should have the highest weight now
    assert w1[0] > w1[2] > w1[1]

    # Multiple updates
    for _ in range(5):
        w = eg.update(np.array([1.0, 0.0, 0.0]))

    history = eg.get_history()
    assert history.shape == (7, 3)
    assert w[0] > 0.8  # Strong convergence towards dominant factor


def test_exponentiated_gradient_errors():
    with pytest.raises(ValueError, match="strictly positive"):
        ExponentiatedGradientCombiner(n_factors=0)
    with pytest.raises(ValueError, match="learning_rate"):
        ExponentiatedGradientCombiner(n_factors=3, learning_rate=-0.1)
    with pytest.raises(ValueError, match="exploration"):
        ExponentiatedGradientCombiner(n_factors=3, exploration=1.5)

    eg = ExponentiatedGradientCombiner(n_factors=3)
    with pytest.raises(ValueError, match="Expected reward vector"):
        eg.update(np.array([0.1, 0.2]))
