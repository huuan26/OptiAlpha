"""
Unit tests for regularized convex alpha combination engine.
"""

import pytest
import numpy as np
import pandas as pd
from optialpha.combination.convex import (
    ConvexAlphaCombiner,
    convex_alpha_combination,
    OptimizationResult,
)


@pytest.fixture
def sample_market():
    np.random.seed(42)
    N, K = 80, 5
    factors = pd.DataFrame(
        np.random.randn(N, K),
        columns=[f"alpha_{k}" for k in range(K)],
        index=[f"asset_{i}" for i in range(N)],
    )
    ic = np.array([0.08, 0.05, 0.02, -0.01, 0.04])
    A = np.random.randn(K, K)
    cov = A.T @ A + 0.2 * np.eye(K)
    return factors, ic, cov


def test_convex_combiner_basic(sample_market):
    factors, ic, cov = sample_market
    combiner = ConvexAlphaCombiner(
        risk_aversion=1.0,
        turnover_penalty=0.0,
        lower_bound=0.0,
        upper_bound=1.0,
        solver="fista",
    )
    res = combiner.optimize(factors, ic, cov)

    assert isinstance(res, OptimizationResult)
    assert res.converged
    assert np.isclose(np.sum(res.weights), 1.0, atol=1e-5)
    assert np.all(res.weights >= 0.0)
    assert isinstance(res.composite_alpha, pd.Series)
    assert len(res.composite_alpha) == 80


def test_turnover_penalty_reduces_turnover(sample_market):
    factors, ic, cov = sample_market
    K = len(ic)
    # Set a fixed previous allocation
    w_prev = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # Unregularized turnover
    res_low_penalty = convex_alpha_combination(
        factors=factors,
        ic_vector=ic,
        covariance=cov,
        turnover_penalty=0.0,
        w_prev=w_prev,
        solver="fista",
    )

    # High turnover penalty
    res_high_penalty = convex_alpha_combination(
        factors=factors,
        ic_vector=ic,
        covariance=cov,
        turnover_penalty=0.10,
        w_prev=w_prev,
        solver="fista",
    )

    # Higher penalty must result in lower turnover from w_prev
    assert res_high_penalty.turnover < res_low_penalty.turnover


def test_risk_aversion_reduces_variance(sample_market):
    factors, ic, cov = sample_market
    res_low_gamma = convex_alpha_combination(
        factors=factors,
        ic_vector=ic,
        covariance=cov,
        risk_aversion=0.1,
        turnover_penalty=0.0,
        solver="fista",
    )
    res_high_gamma = convex_alpha_combination(
        factors=factors,
        ic_vector=ic,
        covariance=cov,
        risk_aversion=10.0,
        turnover_penalty=0.0,
        solver="fista",
    )

    # Higher risk aversion must yield lower or equal portfolio variance
    assert res_high_gamma.predicted_variance <= res_low_gamma.predicted_variance + 1e-6


def test_box_bounds_enforcement(sample_market):
    factors, ic, cov = sample_market
    max_weight = 0.30
    res = convex_alpha_combination(
        factors=factors,
        ic_vector=ic,
        covariance=cov,
        lower_bound=0.05,
        upper_bound=max_weight,
        solver="fista",
    )

    assert np.all(res.weights >= 0.05 - 1e-6)
    assert np.all(res.weights <= max_weight + 1e-6)
    assert np.isclose(np.sum(res.weights), 1.0, atol=1e-5)


def test_solver_slsqp_equivalence(sample_market):
    factors, ic, cov = sample_market
    res_slsqp = convex_alpha_combination(
        factors=factors,
        ic_vector=ic,
        covariance=cov,
        risk_aversion=1.0,
        turnover_penalty=0.01,
        solver="slsqp",
    )

    assert res_slsqp.converged
    assert np.isclose(np.sum(res_slsqp.weights), 1.0, atol=1e-5)


def test_convex_combiner_invalid_params():
    with pytest.raises(ValueError, match="Risk aversion gamma"):
        ConvexAlphaCombiner(risk_aversion=-1.0)
    with pytest.raises(ValueError, match="Turnover penalty"):
        ConvexAlphaCombiner(turnover_penalty=-0.1)
    with pytest.raises(ValueError, match="L2 regularization"):
        ConvexAlphaCombiner(l2_reg=-0.1)
    with pytest.raises(ValueError, match="cannot exceed"):
        ConvexAlphaCombiner(lower_bound=0.8, upper_bound=0.2)
    with pytest.raises(ValueError, match="Unknown solver"):
        ConvexAlphaCombiner(solver="invalid_solver")
