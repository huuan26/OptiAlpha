"""
OptiAlpha: Convex Regularized Alpha Combination and Dynamic Multi-Factor Allocation Engine.

Author: Phan Bùi Hữu Ân (huuan26)
License: MIT
"""

__version__ = "0.1.0"

# Combination APIs
from optialpha.combination import (
    equal_weight_combination,
    ic_weighted_combination,
    icir_weighted_combination,
    inverse_variance_combination,
    grinold_kahn_combination,
    OptimizationResult,
    ConvexAlphaCombiner,
    convex_alpha_combination,
    ExponentiatedGradientCombiner,
)

# Covariance & Shrinkage APIs
from optialpha.covariance import (
    sample_covariance,
    nearest_spd,
    ledoit_wolf_shrinkage,
    exponential_covariance,
)

# Solvers
from optialpha.solvers import (
    project_simplex,
    project_box_simplex,
    prox_l1_shifted,
    solve_fista,
    solve_quadratic_program,
)

# Metrics & Diagnostics
from optialpha.metrics import (
    information_coefficient,
    information_ratio,
    t_statistic_ic,
    turnover_ratio,
    max_drawdown,
    alpha_risk_contributions,
    effective_number_of_alphas,
    diversification_ratio,
)

# Synthetic Data
from optialpha.data import (
    generate_single_period_factors,
    generate_synthetic_alpha_panel,
)

__all__ = [
    # Combination
    "equal_weight_combination",
    "ic_weighted_combination",
    "icir_weighted_combination",
    "inverse_variance_combination",
    "grinold_kahn_combination",
    "OptimizationResult",
    "ConvexAlphaCombiner",
    "convex_alpha_combination",
    "ExponentiatedGradientCombiner",
    # Covariance
    "sample_covariance",
    "nearest_spd",
    "ledoit_wolf_shrinkage",
    "exponential_covariance",
    # Solvers
    "project_simplex",
    "project_box_simplex",
    "prox_l1_shifted",
    "solve_fista",
    "solve_quadratic_program",
    # Metrics
    "information_coefficient",
    "information_ratio",
    "t_statistic_ic",
    "turnover_ratio",
    "max_drawdown",
    "alpha_risk_contributions",
    "effective_number_of_alphas",
    "diversification_ratio",
    # Data
    "generate_single_period_factors",
    "generate_synthetic_alpha_panel",
]
