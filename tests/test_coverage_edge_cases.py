"""
Additional targeted edge cases for 100% test coverage.
"""

from unittest.mock import patch
import pytest
import numpy as np
from optialpha import (
    equal_weight_combination,
    icir_weighted_combination,
    inverse_variance_combination,
    grinold_kahn_combination,
    ledoit_wolf_shrinkage,
    exponential_covariance,
    diversification_ratio,
    project_box_simplex,
    solve_quadratic_program,
    ExponentiatedGradientCombiner,
)


def test_ndim_checks():
    with pytest.raises(ValueError, match="2D matrix"):
        equal_weight_combination(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="2D matrix"):
        icir_weighted_combination(np.random.randn(10, 2), np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="2D matrix"):
        ledoit_wolf_shrinkage(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="2D matrix"):
        exponential_covariance(np.array([1.0, 2.0, 3.0]))


def test_inverse_variance_numpy_array():
    comp, w = inverse_variance_combination(np.random.randn(20, 2), np.eye(2))
    assert isinstance(comp, np.ndarray)
    assert len(w) == 2


def test_grinold_kahn_singular_covariance():
    factors = np.random.randn(20, 2)
    # Singular rank-1 covariance matrix
    cov_singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    comp, w = grinold_kahn_combination(factors, np.array([0.05, 0.05]), cov_singular)
    assert np.isclose(np.sum(w), 1.0)


def test_grinold_kahn_linalg_error_mock():
    factors = np.random.randn(20, 2)
    with patch("numpy.linalg.solve", side_effect=np.linalg.LinAlgError("Singular matrix")):
        comp, w = grinold_kahn_combination(factors, np.array([0.05, 0.05]), np.eye(2))
        assert np.isclose(np.sum(w), 1.0)


def test_project_box_simplex_upper_bound_infeasible():
    with pytest.raises(ValueError, match="Infeasible budget"):
        project_box_simplex(
            np.array([0.1, 0.2]),
            lower_bounds=np.array([0.0, 0.0]),
            upper_bounds=np.array([0.2, 0.2]),
            s=1.0,
        )


def test_quadratic_solver_w_prev_provided():
    w, conv, _, _ = solve_quadratic_program(
        ic=np.array([0.05, 0.02]),
        covariance=np.eye(2),
        turnover_penalty=0.0,
        w_prev=np.array([0.5, 0.5]),
    )
    assert conv
    assert np.isclose(np.sum(w), 1.0)


def test_online_combiner_with_exploration():
    eg = ExponentiatedGradientCombiner(n_factors=3, exploration=0.1)
    w = eg.update(np.array([1.0, 0.0, -1.0]))
    assert np.all(w >= 0.0)
    assert np.isclose(np.sum(w), 1.0)


def test_diversification_ratio_zero_vol():
    cov_zero = np.zeros((2, 2))
    dr = diversification_ratio([0.0, 0.0], cov_zero)
    assert dr == 1.0
