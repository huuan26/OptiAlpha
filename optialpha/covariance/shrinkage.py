"""
Covariance matrix estimation and shrinkage for alpha signals.

References:
    - Ledoit, O., & Wolf, M. (2004).
      A well-conditioned estimator for large-dimensional covariance matrices.
      Journal of Multivariate Analysis, 88(2), 365-411.
    - Ledoit, O., & Wolf, M. (2003).
      Honey, I shrunk the sample covariance matrix.
      The Journal of Portfolio Management, 30(4), 110-119.
    - Higham, N. J. (1988).
      Computing a nearest symmetric positive semidefinite matrix.
      Linear Algebra and its Applications, 103, 103-118.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd


def sample_covariance(
    data: Union[np.ndarray, pd.DataFrame],
    ddof: int = 1,
) -> np.ndarray:
    """
    Compute sample covariance matrix.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame]
        Observation matrix of shape (T, K) with T periods and K factors.
    ddof : int, default=1
        Delta degrees of freedom.

    Returns
    -------
    np.ndarray
        Sample covariance matrix (K, K).
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Input data must be a 2D matrix of shape (T, K).")
    T, K = X.shape
    if T <= ddof:
        raise ValueError(f"Number of observations T={T} must exceed ddof={ddof}.")
    return np.cov(X, rowvar=False, ddof=ddof)


def nearest_spd(matrix: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Compute the nearest Symmetric Positive Definite (SPD) matrix (Higham, 1988).

    Parameters
    ----------
    matrix : np.ndarray
        Input square matrix of shape (K, K).
    eps : float, default=1e-7
        Minimum eigenvalue floor.

    Returns
    -------
    np.ndarray
        Symmetric positive-definite matrix.
    """
    A = np.asarray(matrix, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")

    # Symmetrize
    B = 0.5 * (A + A.T)
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    # Clip negative or too small eigenvalues
    eigvals_clipped = np.maximum(eigvals, eps)
    # Reconstruct
    A_spd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    return 0.5 * (A_spd + A_spd.T)


def ledoit_wolf_shrinkage(
    data: Union[np.ndarray, pd.DataFrame],
    target: str = "constant_correlation",
) -> Tuple[np.ndarray, float]:
    """
    Analytical Ledoit-Wolf optimal shrinkage estimator for covariance matrix.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame]
        Observation matrix of shape (T, K), e.g. historical factor returns or ICs.
    target : str, default="constant_correlation"
        Shrinkage target:
        - "constant_correlation": Sharper target preserving individual factor variances while
          shrinking pairwise correlations to their cross-sectional mean (Ledoit & Wolf, 2004).
        - "identity": Scaled identity target (mu * I_K) shrinking towards spherical noise.

    Returns
    -------
    Tuple[np.ndarray, float]
        (shrunk_covariance_matrix, optimal_shrinkage_intensity_delta)
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Input data must be a 2D matrix of shape (T, K).")
    T, K = X.shape
    if T < 4:
        raise ValueError(f"Ledoit-Wolf shrinkage requires at least T=4 observations, got T={T}.")

    # De-mean observations
    X_c = X - np.mean(X, axis=0, keepdims=True)
    sample_cov = (X_c.T @ X_c) / T  # Unbiased MLE covariance (T divisor)

    var = np.diag(sample_cov)
    std = np.sqrt(np.maximum(var, 1e-12))

    if target == "identity":
        mu = float(np.trace(sample_cov) / K)
        target_matrix = mu * np.eye(K)

        # Frobenius distance d^2 = ||S - T||_F^2
        d2 = float(np.sum((sample_cov - target_matrix) ** 2))

        # Asymptotic variance sum b^2
        b2_sum = 0.0
        for t in range(T):
            diff_t = np.outer(X_c[t], X_c[t]) - sample_cov
            b2_sum += float(np.sum(diff_t**2))
        b2 = b2_sum / (T * T)

        delta = min(1.0, max(0.0, b2 / d2)) if d2 > 1e-12 else 0.0
        shrunk_cov = delta * target_matrix + (1.0 - delta) * sample_cov
        return (T / (T - 1)) * shrunk_cov, float(delta)

    elif target == "constant_correlation":
        # Correlation matrix
        corr = sample_cov / np.outer(std, std)
        np.fill_diagonal(corr, 1.0)

        # Average correlation r_bar
        r_bar = float((np.sum(corr) - K) / (K * (K - 1))) if K > 1 else 0.0
        target_matrix = r_bar * np.outer(std, std)
        np.fill_diagonal(target_matrix, var)

        # Compute pi_mat: sample variance of sample covariance elements
        # pi_ij = 1/T sum_t (x_{ti} x_{tj} - S_{ij})^2
        pi_mat = np.zeros((K, K), dtype=np.float64)
        for t in range(T):
            diff_t = np.outer(X_c[t], X_c[t]) - sample_cov
            pi_mat += diff_t**2
        pi_mat /= T
        pi_hat = float(np.sum(pi_mat))

        # Gamma / target variance terms
        d2 = float(np.sum((sample_cov - target_matrix) ** 2))
        if d2 <= 1e-12:
            return (T / (T - 1)) * sample_cov, 0.0

        # Rho terms across cross-moments
        # Asymptotic covariance term rho_hat
        rho_diag = np.sum(np.diag(pi_mat))
        rho_off = 0.0
        for i in range(K):
            for j in range(K):
                if i != j:
                    term1 = float(np.sum((X_c[:, i] ** 2 - sample_cov[i, i]) * (X_c[:, i] * X_c[:, j] - sample_cov[i, j]))) / T
                    term2 = float(np.sum((X_c[:, j] ** 2 - sample_cov[j, j]) * (X_c[:, i] * X_c[:, j] - sample_cov[i, j]))) / T
                    rho_off += r_bar * 0.5 * (std[j] / std[i] * term1 + std[i] / std[j] * term2)

        rho_hat = rho_diag + rho_off
        kappa = (pi_hat - rho_hat) / d2
        delta = min(1.0, max(0.0, kappa / T))

        shrunk_cov = delta * target_matrix + (1.0 - delta) * sample_cov
        # Scale back to unbiased degree-of-freedom T/(T-1)
        unbiased_shrunk_cov = (T / (T - 1)) * shrunk_cov
        return nearest_spd(unbiased_shrunk_cov), float(delta)

    else:
        raise ValueError(f"Unknown shrinkage target '{target}'. Choose 'constant_correlation' or 'identity'.")


def exponential_covariance(
    data: Union[np.ndarray, pd.DataFrame],
    half_life: float = 60.0,
) -> np.ndarray:
    """
    Compute exponentially weighted covariance matrix with half-life decay.

    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame]
        Observation matrix of shape (T, K).
    half_life : float, default=60.0
        Decay half-life in periods (e.g. trading days).

    Returns
    -------
    np.ndarray
        Exponentially weighted covariance matrix (K, K).
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("Input data must be a 2D matrix of shape (T, K).")
    T, K = X.shape
    if T < 2:
        raise ValueError("Exponential covariance requires at least T=2 observations.")
    if half_life <= 0:
        raise ValueError("Half-life must be strictly positive.")

    decay_factor = np.exp(-np.log(2.0) / half_life)
    weights = np.array([decay_factor ** (T - 1 - t) for t in range(T)], dtype=np.float64)
    weights /= np.sum(weights)

    # Weighted mean
    weighted_mean = np.sum(weights[:, np.newaxis] * X, axis=0, keepdims=True)
    X_centered = X - weighted_mean

    # Weighted covariance
    cov = (X_centered.T * weights) @ X_centered
    # Normalization factor for unbiased weighted covariance
    scale = 1.0 / (1.0 - np.sum(weights**2))
    return nearest_spd(cov * scale)
