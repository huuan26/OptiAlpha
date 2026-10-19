"""
Unit tests for linear alpha combination models.
"""

import pytest
import numpy as np
import pandas as pd
from optialpha.combination.linear import (
    equal_weight_combination,
    ic_weighted_combination,
    icir_weighted_combination,
    inverse_variance_combination,
    grinold_kahn_combination,
)


@pytest.fixture
def sample_factor_data():
    np.random.seed(42)
    N, K = 50, 4
    factors = pd.DataFrame(
        np.random.randn(N, K),
        columns=["f1", "f2", "f3", "f4"],
        index=[f"A_{i}" for i in range(N)],
    )
    ic = np.array([0.05, 0.03, -0.01, 0.02])
    cov = np.diag([1.0, 1.2, 0.8, 1.5])
    return factors, ic, cov


def test_equal_weight_combination(sample_factor_data):
    factors, _, _ = sample_factor_data
    comp, w = equal_weight_combination(factors)

    assert isinstance(comp, pd.Series)
    assert len(comp) == 50
    assert len(w) == 4
    np.testing.assert_allclose(w, [0.25, 0.25, 0.25, 0.25])


def test_ic_weighted_combination(sample_factor_data):
    factors, ic, _ = sample_factor_data
    # Long-only: f3 with negative IC receives 0 weight
    comp, w = ic_weighted_combination(factors, ic, long_only=True)
    assert np.isclose(np.sum(w), 1.0)
    assert w[2] == 0.0
    assert w[0] > w[1] > w[3]

    # Unconstrained
    comp_uncon, w_uncon = ic_weighted_combination(factors, ic, long_only=False)
    assert np.isclose(np.sum(np.abs(w_uncon)), 1.0)
    assert w_uncon[2] < 0.0


def test_icir_weighted_combination(sample_factor_data):
    factors, _, _ = sample_factor_data
    np.random.seed(42)
    T, K = 30, 4
    ic_history = np.random.randn(T, K) * 0.05 + np.array([0.04, 0.02, 0.01, 0.03])

    comp, w = icir_weighted_combination(factors, ic_history, long_only=True)
    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w >= 0.0)


def test_inverse_variance_combination(sample_factor_data):
    factors, _, cov = sample_factor_data
    comp, w = inverse_variance_combination(factors, cov)

    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w > 0.0)
    # Factor with lowest variance (f3: var=0.8) should receive highest weight
    assert np.argmax(w) == 2


def test_grinold_kahn_combination(sample_factor_data):
    factors, ic, cov = sample_factor_data
    comp, w = grinold_kahn_combination(factors, ic, cov, long_only=True)

    assert np.isclose(np.sum(w), 1.0)
    assert np.all(w >= 0.0)


def test_linear_dimension_mismatch():
    factors = np.random.randn(10, 3)
    with pytest.raises(ValueError, match="does not match"):
        ic_weighted_combination(factors, np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="does not match"):
        grinold_kahn_combination(factors, np.array([0.1, 0.2, 0.3]), np.eye(2))
