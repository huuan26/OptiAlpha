"""
Unit tests for proximal algorithms, Euclidean projections, and FISTA solver.
"""

import pytest
import numpy as np
from optialpha.solvers.proximal import (
    project_simplex,
    project_box_simplex,
    prox_l1_shifted,
    solve_fista,
)
from optialpha.solvers.quadratic import solve_quadratic_program


def test_project_simplex_basic():
    v = np.array([0.4, 0.8, -0.2, 0.5])
    w = project_simplex(v, s=1.0)

    assert len(w) == 4
    assert np.all(w >= 0.0)
    assert np.isclose(np.sum(w), 1.0, atol=1e-10)

    # Idempotence: projecting an already feasible point should return itself
    w_idemp = project_simplex(w, s=1.0)
    np.testing.assert_allclose(w, w_idemp, atol=1e-10)


def test_project_simplex_scale():
    v = np.array([1.0, 2.0, 3.0])
    w = project_simplex(v, s=2.5)
    assert np.isclose(np.sum(w), 2.5, atol=1e-10)
    assert np.all(w >= 0.0)


def test_project_simplex_errors():
    with pytest.raises(ValueError, match="strictly positive"):
        project_simplex(np.array([1.0, 2.0]), s=-1.0)
    assert len(project_simplex(np.array([]))) == 0


def test_project_box_simplex():
    v = np.array([0.5, 0.5, 0.5])
    lb = np.array([0.1, 0.2, 0.0])
    ub = np.array([0.4, 0.6, 0.5])

    w = project_box_simplex(v, lb, ub, s=1.0)
    assert np.isclose(np.sum(w), 1.0, atol=1e-9)
    assert np.all(w >= lb - 1e-9)
    assert np.all(w <= ub + 1e-9)


def test_project_box_simplex_infeasible():
    lb = np.array([0.4, 0.4, 0.4])
    ub = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="Infeasible budget"):
        project_box_simplex(np.array([0.4, 0.4, 0.4]), lb, ub, s=1.0)

    with pytest.raises(ValueError, match="must not exceed"):
        project_box_simplex(np.array([0.4, 0.4]), np.array([0.6, 0.2]), np.array([0.5, 0.3]))


def test_prox_l1_shifted():
    v = np.array([1.0, 2.0, 3.0])
    center = np.array([0.5, 2.0, 4.0])
    # diff: [0.5, 0.0, -1.0]
    # with threshold 0.2: diff shrinks to [0.3, 0.0, -0.8]
    # result: [0.8, 2.0, 3.2]
    res = prox_l1_shifted(v, center, threshold=0.2)
    expected = np.array([0.8, 2.0, 3.2])
    np.testing.assert_allclose(res, expected, atol=1e-10)

    # Threshold <= 0 returns copy of v
    np.testing.assert_allclose(prox_l1_shifted(v, center, threshold=0.0), v)


def test_fista_vs_slsqp_agreement():
    np.random.seed(42)
    K = 4
    ic = np.array([0.08, 0.04, 0.01, -0.02])
    # Generate random positive-definite covariance matrix
    A = np.random.randn(K, K)
    cov = A.T @ A + 0.1 * np.eye(K)

    # Solve via FISTA
    w_fista, conv_fista, _, obj_fista = solve_fista(
        ic=ic,
        covariance=cov,
        risk_aversion=1.5,
        turnover_penalty=0.02,
        w_prev=np.full(K, 0.25),
        l2_reg=1e-4,
        s=1.0,
        tol=1e-9,
        max_iter=1000,
    )

    # Solve via SLSQP
    w_slsqp, conv_slsqp, _, obj_slsqp = solve_quadratic_program(
        ic=ic,
        covariance=cov,
        risk_aversion=1.5,
        turnover_penalty=0.02,
        w_prev=np.full(K, 0.25),
        l2_reg=1e-4,
        s=1.0,
        tol=1e-9,
        max_iter=1000,
    )

    assert conv_fista
    assert conv_slsqp
    # Both solvers should yield virtually identical objective values and weights
    np.testing.assert_allclose(obj_fista, obj_slsqp, atol=1e-4)
    np.testing.assert_allclose(w_fista, w_slsqp, atol=1e-2)
