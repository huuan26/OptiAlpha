"""
Regularized convex alpha combination engine with turnover control and box constraints.

Combines multiple alpha signals into an optimal composite signal by solving a
strictly convex optimization problem trading off expected predictive power (IC),
factor variance risk, L1 turnover transaction costs, and factor concentration.
"""

from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np
import pandas as pd

from optialpha.solvers.proximal import solve_fista
from optialpha.solvers.quadratic import solve_quadratic_program
from optialpha.combination.linear import _normalize_factors


@dataclass
class OptimizationResult:
    """Structured container for alpha combination optimization output."""
    weights: np.ndarray
    converged: bool
    iterations: int
    objective_value: float
    predicted_ic: float
    predicted_variance: float
    turnover: float
    composite_alpha: Union[np.ndarray, pd.Series]

    def summary(self) -> dict:
        """Return diagnostic dictionary of the optimization run."""
        return {
            "converged": self.converged,
            "iterations": self.iterations,
            "objective_value": round(self.objective_value, 6),
            "predicted_ic": round(self.predicted_ic, 6),
            "predicted_variance": round(self.predicted_variance, 6),
            "turnover": round(self.turnover, 6),
        }


class ConvexAlphaCombiner:
    """
    Convex Regularized Alpha Combination Engine.

    Parameters
    ----------
    risk_aversion : float, default=1.0
        Risk aversion penalty gamma on factor portfolio variance w^T Omega w.
    turnover_penalty : float, default=0.0
        L1 penalty parameter lambda_turnover on weight delta ||w - w_prev||_1.
    l2_reg : float, default=1e-5
        L2 Ridge regularization parameter on ||w||_2^2 for strict convexity.
    lower_bound : float, default=0.0
        Minimum weight per factor (0.0 enforces long-only alphas).
    upper_bound : float, default=1.0
        Maximum allowable weight per factor to control concentration.
    budget : float, default=1.0
        Budget sum constraint sum(w) == budget.
    solver : str, default="fista"
        Optimization solver: 'fista' (fast proximal gradient) or 'slsqp' (SciPy SQP).
    tol : float, default=1e-8
        Convergence tolerance.
    max_iter : int, default=500
        Maximum solver iterations.
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        turnover_penalty: float = 0.0,
        l2_reg: float = 1e-5,
        lower_bound: float = 0.0,
        upper_bound: float = 1.0,
        budget: float = 1.0,
        solver: str = "fista",
        tol: float = 1e-8,
        max_iter: int = 500,
    ):
        if risk_aversion < 0:
            raise ValueError("Risk aversion gamma must be non-negative.")
        if turnover_penalty < 0:
            raise ValueError("Turnover penalty must be non-negative.")
        if l2_reg < 0:
            raise ValueError("L2 regularization parameter must be non-negative.")
        if lower_bound > upper_bound:
            raise ValueError("lower_bound cannot exceed upper_bound.")
        if solver not in ("fista", "slsqp"):
            raise ValueError(f"Unknown solver '{solver}'. Supported: 'fista', 'slsqp'.")

        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.l2_reg = l2_reg
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.budget = budget
        self.solver = solver
        self.tol = tol
        self.max_iter = max_iter

    def optimize(
        self,
        factors: Union[np.ndarray, pd.DataFrame],
        ic_vector: Union[np.ndarray, pd.Series],
        covariance: np.ndarray,
        w_prev: Optional[np.ndarray] = None,
        normalize_factors: bool = True,
    ) -> OptimizationResult:
        """
        Compute optimal alpha weights and generate composite alpha series.

        Parameters
        ----------
        factors : Union[np.ndarray, pd.DataFrame]
            Cross-sectional factor matrix of shape (N_assets, K_factors).
        ic_vector : Union[np.ndarray, pd.Series]
            Vector of factor ICs (K_factors,).
        covariance : np.ndarray
            Factor covariance matrix (K_factors, K_factors).
        w_prev : Optional[np.ndarray], default=None
            Previous allocation weights (K_factors,) for turnover penalty.
        normalize_factors : bool, default=True
            Whether to cross-sectionally standardize factor exposures.

        Returns
        -------
        OptimizationResult
            Container with weights, composite alpha, and optimization telemetry.
        """
        raw_arr, cols, idx = _normalize_factors(factors) if normalize_factors else (
            factors.to_numpy(dtype=np.float64) if isinstance(factors, pd.DataFrame) else np.asarray(factors, dtype=np.float64),
            list(factors.columns) if isinstance(factors, pd.DataFrame) else None,
            list(factors.index) if isinstance(factors, pd.DataFrame) else None,
        )

        ic = np.asarray(ic_vector, dtype=np.float64).ravel()
        K = raw_arr.shape[1]
        if len(ic) != K:
            raise ValueError(f"Length of ic_vector ({len(ic)}) does not match number of factors ({K}).")

        cov = np.asarray(covariance, dtype=np.float64)
        if cov.shape != (K, K):
            raise ValueError(f"Covariance shape {cov.shape} does not match factor count {K}.")

        lb = np.full(K, self.lower_bound, dtype=np.float64)
        ub = np.full(K, self.upper_bound, dtype=np.float64)

        if w_prev is not None:
            w_prev_arr = np.asarray(w_prev, dtype=np.float64).ravel()
        else:
            w_prev_arr = np.full(K, self.budget / K, dtype=np.float64)

        if self.solver == "fista":
            w_opt, conv, iters, obj = solve_fista(
                ic=ic,
                covariance=cov,
                risk_aversion=self.risk_aversion,
                turnover_penalty=self.turnover_penalty,
                w_prev=w_prev_arr,
                l2_reg=self.l2_reg,
                lower_bounds=lb,
                upper_bounds=ub,
                s=self.budget,
                max_iter=self.max_iter,
                tol=self.tol,
            )
        else:
            w_opt, conv, iters, obj = solve_quadratic_program(
                ic=ic,
                covariance=cov,
                risk_aversion=self.risk_aversion,
                turnover_penalty=self.turnover_penalty,
                w_prev=w_prev_arr,
                l2_reg=self.l2_reg,
                lower_bounds=lb,
                upper_bounds=ub,
                s=self.budget,
                max_iter=self.max_iter,
                tol=self.tol,
            )

        pred_ic = float(np.dot(ic, w_opt))
        pred_var = float(w_opt.T @ cov @ w_opt)
        turnover = float(np.sum(np.abs(w_opt - w_prev_arr)))
        composite_arr = raw_arr @ w_opt

        if idx is not None:
            composite_alpha = pd.Series(composite_arr, index=idx, name="composite_alpha")
        else:
            composite_alpha = composite_arr

        return OptimizationResult(
            weights=w_opt,
            converged=conv,
            iterations=iters,
            objective_value=obj,
            predicted_ic=pred_ic,
            predicted_variance=pred_var,
            turnover=turnover,
            composite_alpha=composite_alpha,
        )


def convex_alpha_combination(
    factors: Union[np.ndarray, pd.DataFrame],
    ic_vector: Union[np.ndarray, pd.Series],
    covariance: np.ndarray,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.0,
    w_prev: Optional[np.ndarray] = None,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    solver: str = "fista",
    normalize_factors: bool = True,
) -> OptimizationResult:
    """
    Functional wrapper for convex regularized alpha combination.

    Parameters
    ----------
    factors : Union[np.ndarray, pd.DataFrame]
        Factor exposure matrix (N, K).
    ic_vector : Union[np.ndarray, pd.Series]
        Factor IC vector (K,).
    covariance : np.ndarray
        Alpha covariance matrix (K, K).
    risk_aversion : float, default=1.0
        Risk aversion parameter gamma.
    turnover_penalty : float, default=0.0
        Turnover penalty lambda_turnover.
    w_prev : Optional[np.ndarray], default=None
        Previous weight vector for turnover control.
    lower_bound : float, default=0.0
        Minimum weight per factor.
    upper_bound : float, default=1.0
        Maximum weight per factor.
    solver : str, default='fista'
        Optimization solver ('fista' or 'slsqp').
    normalize_factors : bool, default=True
        Whether to standardize factors cross-sectionally.

    Returns
    -------
    OptimizationResult
        Optimization result object.
    """
    combiner = ConvexAlphaCombiner(
        risk_aversion=risk_aversion,
        turnover_penalty=turnover_penalty,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        solver=solver,
    )
    return combiner.optimize(
        factors=factors,
        ic_vector=ic_vector,
        covariance=covariance,
        w_prev=w_prev,
        normalize_factors=normalize_factors,
    )
