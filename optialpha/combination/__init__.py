"""
Alpha combination algorithms and engines.
"""

from optialpha.combination.linear import (
    equal_weight_combination,
    ic_weighted_combination,
    icir_weighted_combination,
    inverse_variance_combination,
    grinold_kahn_combination,
)
from optialpha.combination.convex import (
    OptimizationResult,
    ConvexAlphaCombiner,
    convex_alpha_combination,
)
from optialpha.combination.online import ExponentiatedGradientCombiner

__all__ = [
    "equal_weight_combination",
    "ic_weighted_combination",
    "icir_weighted_combination",
    "inverse_variance_combination",
    "grinold_kahn_combination",
    "OptimizationResult",
    "ConvexAlphaCombiner",
    "convex_alpha_combination",
    "ExponentiatedGradientCombiner",
]
