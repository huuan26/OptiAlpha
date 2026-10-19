"""
Online adaptive alpha allocation and multiplicative weight updates.

References:
    - Freund, Y., & Schapire, R. E. (1997).
      A decision-theoretic generalization of on-line learning and an application to boosting.
      Journal of Computer and System Sciences, 55(1), 119-139.
    - Kivinen, J., & Warmuth, M. K. (1997).
      Additive versus exponentiated gradient updates for linear predictors.
      Information and Computation, 132(1), 1-64.
"""

from typing import Optional, List, Union
import numpy as np
import pandas as pd


class ExponentiatedGradientCombiner:
    """
    Online Alpha Allocation via Exponentiated Gradient / Multiplicative Weights.

    Maintains adaptive factor weights over time with sublinear regret bounds
    O(sqrt(T * log(K))) against the best static alpha combination in hindsight.

    Parameters
    ----------
    n_factors : int
        Number of candidate alpha factors K.
    learning_rate : float, default=0.1
        Learning rate eta for exponential reward scaling.
    exploration : float, default=0.01
        Exploration mixing parameter epsilon to prevent premature factor abandonment.
    """

    def __init__(
        self,
        n_factors: int,
        learning_rate: float = 0.1,
        exploration: float = 0.01,
    ):
        if n_factors <= 0:
            raise ValueError("n_factors must be strictly positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be strictly positive.")
        if not (0.0 <= exploration <= 1.0):
            raise ValueError("exploration must be in [0, 1].")

        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.exploration = exploration

        # Initialize uniform weights on simplex
        self.weights = np.full(n_factors, 1.0 / n_factors, dtype=np.float64)
        self.history: List[np.ndarray] = [np.copy(self.weights)]

    def get_weights(self) -> np.ndarray:
        """Return current alpha allocation weights."""
        return np.copy(self.weights)

    def update(self, rewards: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Update alpha weights based on realized factor performance (e.g. IC or returns).

        Parameters
        ----------
        rewards : Union[np.ndarray, pd.Series]
            Performance vector of shape (K,) observed in the most recent period.

        Returns
        -------
        np.ndarray
            Updated weight vector for the subsequent period.
        """
        r = np.asarray(rewards, dtype=np.float64).ravel()
        if len(r) != self.n_factors:
            raise ValueError(f"Expected reward vector of length {self.n_factors}, got {len(r)}.")

        # Clip extreme values for numerical stability
        scaled_r = self.learning_rate * np.clip(r, -10.0, 10.0)

        # Multiplicative exponentiated update: w_i * exp(eta * r_i)
        unnorm_w = self.weights * np.exp(scaled_r - np.max(scaled_r))
        sum_w = np.sum(unnorm_w)
        if sum_w < 1e-15:
            new_weights = np.full(self.n_factors, 1.0 / self.n_factors)
        else:
            new_weights = unnorm_w / sum_w

        # Exploration mixing: (1 - eps) * w + eps / K
        if self.exploration > 0.0:
            uniform = 1.0 / self.n_factors
            new_weights = (1.0 - self.exploration) * new_weights + self.exploration * uniform

        self.weights = new_weights
        self.history.append(np.copy(self.weights))
        return np.copy(self.weights)

    def get_history(self) -> np.ndarray:
        """Return full trajectory of weights across all time steps as array of shape (T, K)."""
        return np.array(self.history, dtype=np.float64)
