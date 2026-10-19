"""
Risk attribution and diversification metrics for multi-factor combinations.

References:
    - Meucci, A. (2009). Managing diversification. Risk, 22(5), 74-79.
    - Choueifaty, Y., & Coignard, Y. (2008).
      Toward maximum diversification. The Journal of Portfolio Management, 35(1), 40-51.
"""

from typing import Optional, Union, Tuple
import numpy as np


def alpha_risk_contributions(
    weights: Union[np.ndarray, list],
    covariance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute marginal and percentage risk contributions of each alpha factor.

    Total Portfolio Variance: sigma_p^2 = w^T Omega w
    Marginal Risk Contribution (MRC_k): (Omega w)_k / sigma_p
    Percentage Risk Contribution (PRC_k): w_k * (Omega w)_k / sigma_p^2
    Sum(PRC) == 1.0 (Euler's decomposition of homogeneous degree 1 functions).

    Parameters
    ----------
    weights : Union[np.ndarray, list]
        Factor weights vector of shape (K,).
    covariance : np.ndarray
        Factor covariance matrix of shape (K, K).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (marginal_risk_contributions, percentage_risk_contributions)
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    cov = np.asarray(covariance, dtype=np.float64)
    K = len(w)

    total_var = float(w.T @ cov @ w)
    total_vol = np.sqrt(max(total_var, 1e-14))

    mrc = (cov @ w) / total_vol

    if total_var > 1e-14:
        prc = (w * (cov @ w)) / total_var
    else:
        prc = np.full(K, 1.0 / K)

    return mrc, prc


def effective_number_of_alphas(
    weights: Union[np.ndarray, list],
    covariance: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Effective Number of Alphas (Meucci, 2009) via exponential Shannon entropy.

    If covariance is provided, uses Percentage Risk Contributions p_k.
    If covariance is None, uses normalized non-negative weights p_k = w_k / sum(w).

    Formula:
        ENF = exp( - sum_{k=1}^K p_k * ln(p_k) )

    Ranges from 1.0 (concentrated in a single factor) to K (perfectly balanced diversification).

    Parameters
    ----------
    weights : Union[np.ndarray, list]
        Factor weights vector (K,).
    covariance : Optional[np.ndarray], default=None
        Factor covariance matrix (K, K).

    Returns
    -------
    float
        Effective number of alphas in [1.0, K].
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    K = len(w)
    if K == 0:
        return 0.0

    if covariance is not None:
        _, p = alpha_risk_contributions(w, covariance)
        # Handle small negative numerical artifacts
        p = np.maximum(p, 0.0)
        s = np.sum(p)
        p = p / s if s > 1e-12 else np.full(K, 1.0 / K)
    else:
        p_w = np.maximum(w, 0.0)
        s = np.sum(p_w)
        p = p_w / s if s > 1e-12 else np.full(K, 1.0 / K)

    # Shannon entropy
    entropy = 0.0
    for pk in p:
        if pk > 1e-12:
            entropy -= pk * np.log(pk)

    return float(np.exp(entropy))


def diversification_ratio(
    weights: Union[np.ndarray, list],
    covariance: np.ndarray,
) -> float:
    """
    Compute Choueifaty Diversification Ratio (DR):
        DR = (sum_{k=1}^K w_k * sigma_k) / sqrt(w^T Omega w)

    For perfectly correlated assets, DR == 1.0. Higher DR indicates stronger diversification benefit.

    Parameters
    ----------
    weights : Union[np.ndarray, list]
        Factor weights vector (K,).
    covariance : np.ndarray
        Covariance matrix (K, K).

    Returns
    -------
    float
        Diversification ratio >= 1.0.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    cov = np.asarray(covariance, dtype=np.float64)
    raw_var = float(w.T @ cov @ w)
    if raw_var <= 1e-14:
        return 1.0
    vol_p = np.sqrt(raw_var)
    individual_vols = np.sqrt(np.maximum(np.diag(cov), 0.0))
    weighted_vol_sum = float(np.sum(np.abs(w) * individual_vols))
    return float(weighted_vol_sum / vol_p)
