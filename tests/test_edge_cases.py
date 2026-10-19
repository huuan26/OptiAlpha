"""
Edge case and coverage tests for OptiAlpha.
"""

import pytest
import numpy as np
import pandas as pd
from optialpha import (
    ConvexAlphaCombiner,
    convex_alpha_combination,
    equal_weight_combination,
    ic_weighted_combination,
    grinold_kahn_combination,
    OptimizationResult,
    solve_quadratic_program,
    solve_fista,
    ledoit_wolf_shrinkage,
    alpha_risk_contributions,
    effective_number_of_alphas,
    diversification_ratio,
    information_ratio,
    t_statistic_ic,
    ExponentiatedGradientCombiner,
)


def test_optimization_result_summary():
    res = OptimizationResult(
        weights=np.array([0.5, 0.5]),
        converged=True,
        iterations=15,
        objective_value=-0.0421234,
        predicted_ic=0.045678,
        predicted_variance=0.001234,
        turnover=0.123456,
        composite_alpha=np.array([1.0, 2.0]),
    )
    summary = res.summary()
    assert summary["converged"] is True
    assert summary["iterations"] == 15
    assert summary["objective_value"] == -0.042123
    assert summary["predicted_ic"] == 0.045678


def test_numpy_array_inputs_without_dataframe():
    np.random.seed(42)
    factors_arr = np.random.randn(40, 3)
    ic_arr = np.array([0.05, 0.03, 0.01])
    cov_arr = np.eye(3)

    comp_eq, w_eq = equal_weight_combination(factors_arr, normalize_factors=False)
    assert isinstance(comp_eq, np.ndarray)
    assert len(comp_eq) == 40

    comp_ic, w_ic = ic_weighted_combination(factors_arr, ic_arr, normalize_factors=False)
    assert isinstance(comp_ic, np.ndarray)

    res = convex_alpha_combination(
        factors=factors_arr,
        ic_vector=ic_arr,
        covariance=cov_arr,
        normalize_factors=False,
    )
    assert isinstance(res.composite_alpha, np.ndarray)


def test_mismatched_dimensions_in_convex():
    factors = np.random.randn(30, 4)
    cov = np.eye(4)
    # IC length mismatch
    with pytest.raises(ValueError, match="does not match"):
        convex_alpha_combination(factors, np.array([0.1, 0.2]), cov)

    # Covariance shape mismatch
    with pytest.raises(ValueError, match="Covariance shape"):
        convex_alpha_combination(factors, np.array([0.1, 0.2, 0.3, 0.4]), np.eye(3))


def test_zero_or_negative_ic_in_linear():
    factors = np.random.randn(20, 3)
    # All negative ICs with long_only=True should fallback to uniform weights
    all_neg_ic = np.array([-0.05, -0.02, -0.01])
    _, w = ic_weighted_combination(factors, all_neg_ic, long_only=True)
    np.testing.assert_allclose(w, [1/3, 1/3, 1/3])

    # All zeros in unconstrained
    zero_ic = np.array([0.0, 0.0, 0.0])
    _, w_zero = ic_weighted_combination(factors, zero_ic, long_only=False)
    np.testing.assert_allclose(w_zero, [1/3, 1/3, 1/3])


def test_quadratic_solver_zero_turnover_penalty():
    ic = np.array([0.05, 0.03])
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    w, conv, iters, obj = solve_quadratic_program(
        ic=ic,
        covariance=cov,
        risk_aversion=1.0,
        turnover_penalty=0.0,
        w_prev=np.array([0.9, 0.1]),
        s=1.0,
    )
    assert conv
    assert np.isclose(np.sum(w), 1.0)


def test_quadratic_solver_unbalanced_w_prev():
    ic = np.array([0.05, 0.03])
    cov = np.eye(2)
    # w_prev whose sum is far from s=1.0
    w_prev = np.array([0.0, 0.0])
    w, conv, _, _ = solve_quadratic_program(
        ic=ic,
        covariance=cov,
        turnover_penalty=0.01,
        w_prev=w_prev,
        s=1.0,
    )
    assert conv
    assert np.isclose(np.sum(w), 1.0)

    # Also test for turnover_penalty=0.0 branch
    w0, conv0, _, _ = solve_quadratic_program(
        ic=ic,
        covariance=cov,
        turnover_penalty=0.0,
        w_prev=w_prev,
        s=1.0,
    )
    assert conv0
    assert np.isclose(np.sum(w0), 1.0)


def test_grinold_kahn_unconstrained_and_zero_sum():
    factors = np.random.randn(20, 2)
    cov = np.eye(2)
    # Unconstrained positive weights
    _, w = grinold_kahn_combination(factors, np.array([0.05, 0.05]), cov, long_only=False)
    assert np.isclose(np.sum(np.abs(w)), 1.0)

    # All zeros IC -> raw_weights == [0, 0], abs_sum == 0
    _, w_zero = grinold_kahn_combination(factors, np.array([0.0, 0.0]), cov, long_only=False)
    np.testing.assert_allclose(w_zero, [0.5, 0.5])


def test_ledoit_wolf_single_factor_and_identity_branches():
    # K=1 constant correlation
    data_1d = np.random.randn(20, 1)
    cov_1d, delta_1d = ledoit_wolf_shrinkage(data_1d, target="constant_correlation")
    assert cov_1d.shape == (1, 1)

    # Constant data where d2 == 0
    constant_data = np.ones((20, 2)) * 5.0 + np.random.randn(20, 2) * 1e-10
    cov_c, delta_c = ledoit_wolf_shrinkage(constant_data, target="identity")
    assert cov_c.shape == (2, 2)


def test_risk_contributions_zero_variance():
    # Covariance all zeros
    cov_zero = np.zeros((2, 2))
    w = np.array([0.5, 0.5])
    mrc, prc = alpha_risk_contributions(w, cov_zero)
    np.testing.assert_allclose(prc, [0.5, 0.5])


def test_metrics_zero_variance_branches():
    # IC series with zero variance (identical elements)
    identical_ics = [0.05, 0.05, 0.05, 0.05]
    assert information_ratio(identical_ics) == 0.0
    assert t_statistic_ic(identical_ics) == 0.0

    # Diversification ratio with zero portfolio vol
    cov_zero = np.zeros((2, 2))
    dr = diversification_ratio([0.5, 0.5], cov_zero)
    assert dr == 1.0


def test_fista_zero_lipschitz():
    ic = np.array([0.0, 0.0])
    cov = np.zeros((2, 2))
    w, conv, _, _ = solve_fista(ic=ic, covariance=cov, risk_aversion=0.0, l2_reg=0.0)
    assert np.isclose(np.sum(w), 1.0)


def test_exponentiated_gradient_underflow():
    eg = ExponentiatedGradientCombiner(n_factors=2, learning_rate=1.0, exploration=0.0)
    # Set weights near zero
    eg.weights = np.array([1e-20, 1e-20])
    w = eg.update(np.array([-10.0, -10.0]))
    np.testing.assert_allclose(w, [0.5, 0.5])
