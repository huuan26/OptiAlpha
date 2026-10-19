"""
Sequential Quadratic Programming (SQP) solver for convex alpha optimization.

Provides exact gradient and Hessian evaluations for constrained convex quadratic programs.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from scipy.optimize import minimize


def solve_quadratic_program(
    ic: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.0,
    w_prev: Optional[np.ndarray] = None,
    l2_reg: float = 1e-5,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    s: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> Tuple[np.ndarray, bool, int, float]:
    """
    Solve regularized alpha combination via SLSQP with exact analytic gradients.

    If turnover_penalty > 0, the objective uses auxiliary slack variables
    u_k >= |w_k - w_{prev, k}| to cast the non-smooth L1 norm into a smooth QP.

    Parameters
    ----------
    ic : np.ndarray
        Factor Information Coefficient (IC) vector of shape (K,).
    covariance : np.ndarray
        Alpha covariance matrix of shape (K, K).
    risk_aversion : float, default=1.0
        Risk aversion parameter gamma.
    turnover_penalty : float, default=0.0
        Turnover penalty lambda_turnover >= 0.
    w_prev : Optional[np.ndarray], default=None
        Previous weight vector for turnover control.
    l2_reg : float, default=1e-5
        L2 regularization parameter.
    lower_bounds : Optional[np.ndarray], default=None
        Lower bounds on weights. Defaults to 0.
    upper_bounds : Optional[np.ndarray], default=None
        Upper bounds on weights. Defaults to 1.
    s : float, default=1.0
        Sum constraint sum(w) == s.
    tol : float, default=1e-8
        Convergence tolerance.
    max_iter : int, default=500
        Maximum iterations.

    Returns
    -------
    Tuple[np.ndarray, bool, int, float]
        (optimal_weights, converged, iterations, final_objective)
    """
    ic = np.asarray(ic, dtype=np.float64).ravel()
    k = len(ic)
    cov = np.asarray(covariance, dtype=np.float64)

    if w_prev is None:
        w_prev = np.full(k, s / k, dtype=np.float64)
    else:
        w_prev = np.asarray(w_prev, dtype=np.float64).ravel()

    if lower_bounds is None:
        lower_bounds = np.zeros(k, dtype=np.float64)
    else:
        lower_bounds = np.asarray(lower_bounds, dtype=np.float64).ravel()

    if upper_bounds is None:
        upper_bounds = np.ones(k, dtype=np.float64)
    else:
        upper_bounds = np.asarray(upper_bounds, dtype=np.float64).ravel()

    hessian_w = risk_aversion * cov + l2_reg * np.eye(k)

    if turnover_penalty <= 1e-12:
        # Standard smooth QP over w
        def objective(w: np.ndarray) -> float:
            smooth = -np.dot(ic, w) + 0.5 * float(w.T @ hessian_w @ w)
            return float(smooth)

        def gradient(w: np.ndarray) -> np.ndarray:
            return -ic + hessian_w @ w

        bounds = list(zip(lower_bounds, upper_bounds))
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - s, "jac": lambda w: np.ones(k)}]

        w0 = np.clip(w_prev, lower_bounds, upper_bounds)
        if np.abs(np.sum(w0) - s) > 1e-6:
            w0 = np.full(k, s / k)

        res = minimize(
            fun=objective,
            x0=w0,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": max_iter, "ftol": tol},
        )

        w_opt = res.x
        return w_opt, bool(res.success), int(res.nit), float(res.fun)

    else:
        # Reformulate with slack variables u in R^K: u_i >= w_i - w_prev_i, u_i >= w_prev_i - w_i
        # Variable vector x = [w, u] of length 2K
        def objective(x: np.ndarray) -> float:
            w = x[:k]
            u = x[k:]
            smooth = -np.dot(ic, w) + 0.5 * float(w.T @ hessian_w @ w)
            turnover = turnover_penalty * float(np.sum(u))
            return float(smooth + turnover)

        def gradient(x: np.ndarray) -> np.ndarray:
            w = x[:k]
            grad_w = -ic + hessian_w @ w
            grad_u = np.full(k, turnover_penalty, dtype=np.float64)
            return np.concatenate([grad_w, grad_u])

        # Bounds: l_i <= w_i <= u_i, u_i >= 0
        bounds = list(zip(lower_bounds, upper_bounds)) + [(0.0, None)] * k

        constraints = [
            # sum(w) == s
            {
                "type": "eq",
                "fun": lambda x: np.sum(x[:k]) - s,
                "jac": lambda x: np.concatenate([np.ones(k), np.zeros(k)]),
            },
            # u - (w - w_prev) >= 0 => u - w + w_prev >= 0
            {
                "type": "ineq",
                "fun": lambda x: x[k:] - x[:k] + w_prev,
                "jac": lambda x: np.hstack([-np.eye(k), np.eye(k)]),
            },
            # u - (w_prev - w) >= 0 => u + w - w_prev >= 0
            {
                "type": "ineq",
                "fun": lambda x: x[k:] + x[:k] - w_prev,
                "jac": lambda x: np.hstack([np.eye(k), np.eye(k)]),
            },
        ]

        w0 = np.clip(w_prev, lower_bounds, upper_bounds)
        if np.abs(np.sum(w0) - s) > 1e-6:
            w0 = np.full(k, s / k)
        u0 = np.abs(w0 - w_prev)
        x0 = np.concatenate([w0, u0])

        res = minimize(
            fun=objective,
            x0=x0,
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"maxiter": max_iter, "ftol": tol},
        )

        w_opt = res.x[:k]
        return w_opt, bool(res.success), int(res.nit), float(res.fun)
