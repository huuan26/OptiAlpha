"""
Proximal algorithms and Euclidean projections for convex alpha optimization.

References:
    - Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008).
      Efficient projections onto the l1-ball for predictors in high dimensions. ICML.
    - Beck, A., & Teboulle, M. (2009).
      A fast iterative shrinkage-thresholding algorithm for linear inverse problems.
      SIAM Journal on Imaging Sciences, 2(1), 183-202.
    - Wang, W., & Carreira-Perpinán, M. A. (2013).
      Projection onto the probability simplex: An efficient algorithm with a simple proof.
"""

from typing import Optional, Tuple
import numpy as np


def project_simplex(v: np.ndarray, s: float = 1.0) -> np.ndarray:
    """
    Project vector v onto the probability simplex:
        {w in R^K : w >= 0, sum(w) == s}

    Parameters
    ----------
    v : np.ndarray
        Input vector of shape (K,).
    s : float, default=1.0
        Simplex scale constraint sum(w) == s.

    Returns
    -------
    np.ndarray
        Projected vector on the simplex.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    k = len(v)
    if k == 0:
        return np.array([], dtype=np.float64)
    if s <= 0:
        raise ValueError("Simplex sum constraint s must be strictly positive.")

    # Sort descending
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    indices = np.arange(1, k + 1)
    cond = u + (s - cssv) / indices > 0
    rho = np.nonzero(cond)[0][-1]
    theta = (s - cssv[rho]) / (rho + 1)
    w = np.maximum(v + theta, 0.0)
    return w


def project_box_simplex(
    v: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    s: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Project vector v onto the box-constrained simplex:
        {w in R^K : l_k <= w_k <= u_k, sum(w) == s}

    Parameters
    ----------
    v : np.ndarray
        Input vector of shape (K,).
    lower_bounds : np.ndarray
        Lower bound vector (K,).
    upper_bounds : np.ndarray
        Upper bound vector (K,).
    s : float, default=1.0
        Budget sum constraint.
    tol : float, default=1e-12
        Tolerance for root finding on dual variable.
    max_iter : int, default=100
        Maximum bisection iterations.

    Returns
    -------
    np.ndarray
        Projected vector satisfying box and budget constraints.
    """
    v = np.asarray(v, dtype=np.float64).ravel()
    lb = np.asarray(lower_bounds, dtype=np.float64).ravel()
    ub = np.asarray(upper_bounds, dtype=np.float64).ravel()

    if np.any(lb > ub):
        raise ValueError("Lower bounds must not exceed upper bounds.")
    if s < np.sum(lb) - 1e-9 or s > np.sum(ub) + 1e-9:
        raise ValueError(
            f"Infeasible budget: sum of bounds [{np.sum(lb):.4f}, {np.sum(ub):.4f}] cannot satisfy sum={s:.4f}"
        )

    # Dual problem is monotonic in shift parameter lambda: w_i(lambda) = clip(v_i - lambda, lb_i, ub_i)
    # Find bounding bracket [lambda_min, lambda_max]
    lambda_low = np.min(v - ub)
    lambda_high = np.max(v - lb)

    for _ in range(max_iter):
        lambda_mid = 0.5 * (lambda_low + lambda_high)
        w_mid = np.clip(v - lambda_mid, lb, ub)
        diff = np.sum(w_mid) - s

        if np.abs(diff) < tol:
            return w_mid
        if diff > 0:
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid

    return np.clip(v - (0.5 * (lambda_low + lambda_high)), lb, ub)


def prox_l1_shifted(
    v: np.ndarray,
    center: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Proximal operator for shifted L1 penalty:
        arg min_w 0.5 * ||w - v||_2^2 + threshold * ||w - center||_1

    Parameters
    ----------
    v : np.ndarray
        Current candidate vector.
    center : np.ndarray
        Reference center vector (e.g. previous period weights w_prev).
    threshold : float
        Threshold parameter tau = step_size * lambda_turnover.

    Returns
    -------
    np.ndarray
        Proximal result after soft-thresholding relative to center.
    """
    if threshold <= 0.0:
        return np.copy(v)
    diff = v - center
    shrink = np.sign(diff) * np.maximum(np.abs(diff) - threshold, 0.0)
    return center + shrink


def solve_fista(
    ic: np.ndarray,
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.0,
    w_prev: Optional[np.ndarray] = None,
    l2_reg: float = 1e-5,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None,
    s: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, bool, int, float]:
    """
    Solve the regularized alpha combination problem via FISTA:
        min_w  -ic^T w + 0.5 * gamma * w^T Omega w + 0.5 * l2_reg * ||w||_2^2
               + lambda_turnover * ||w - w_prev||_1
        s.t.   l_k <= w_k <= u_k,  sum(w) == s

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
        Previous weight vector for turnover control. If None, defaults to uniform weights.
    l2_reg : float, default=1e-5
        L2 regularization parameter to ensure strict convexity.
    lower_bounds : Optional[np.ndarray], default=None
        Lower bounds on weights. Defaults to 0 (long-only).
    upper_bounds : Optional[np.ndarray], default=None
        Upper bounds on weights. Defaults to 1.
    s : float, default=1.0
        Sum constraint sum(w) == s.
    max_iter : int, default=500
        Maximum FISTA iterations.
    tol : float, default=1e-8
        Convergence tolerance on weight change norm.

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

    # Hessian of smooth part: H = gamma * Omega + l2_reg * I
    hessian = risk_aversion * cov + l2_reg * np.eye(k)
    # Estimate Lipschitz constant L via spectral norm (largest eigenvalue)
    eigenvalues = np.linalg.eigvalsh(hessian)
    lipschitz = float(np.max(eigenvalues))
    if lipschitz <= 0:
        lipschitz = 1.0
    step_size = 1.0 / lipschitz

    # Initialize at feasible point
    w = project_box_simplex(w_prev, lower_bounds, upper_bounds, s=s)
    y = np.copy(w)
    t = 1.0

    def compute_objective(weights: np.ndarray) -> float:
        smooth_cost = (
            -np.dot(ic, weights)
            + 0.5 * risk_aversion * float(weights.T @ cov @ weights)
            + 0.5 * l2_reg * float(np.sum(weights**2))
        )
        non_smooth_cost = turnover_penalty * float(np.sum(np.abs(weights - w_prev)))
        return float(smooth_cost + non_smooth_cost)

    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        # Gradient of smooth part evaluated at y
        grad_f = -ic + (hessian @ y)

        # Gradient descent step
        z = y - step_size * grad_f

        # Apply proximal operator for L1 turnover penalty if lambda > 0
        if turnover_penalty > 0.0:
            z = prox_l1_shifted(z, w_prev, step_size * turnover_penalty)

        # Project onto constrained simplex
        w_next = project_box_simplex(z, lower_bounds, upper_bounds, s=s)

        # Check convergence
        diff_norm = float(np.linalg.norm(w_next - w))
        if diff_norm < tol:
            converged = True
            w = w_next
            break

        # Nesterov momentum acceleration update
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = w_next + ((t - 1.0) / t_next) * (w_next - w)

        w = w_next
        t = t_next

    final_obj = compute_objective(w)
    return w, converged, iteration, final_obj
