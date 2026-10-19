"""
Classical and closed-form linear alpha combination models.

References:
    - Grinold, R. C., & Kahn, R. N. (2000).
      Active Portfolio Management: A Quantitative Approach for Producing Superior Returns
      and Controlling Risk. McGraw-Hill.
    - Qian, E. E., & Hua, R. H. (2004).
      Active risk and information ratio in multi-factor models.
      The Journal of Portfolio Management, 30(3), 20-34.
"""

from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from optialpha.solvers.proximal import project_simplex


def _normalize_factors(factors: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, Optional[list], Optional[list]]:
    """Convert input factors to 2D numpy array and standardize cross-sectionally (mean 0, std 1)."""
    if isinstance(factors, pd.DataFrame):
        columns = list(factors.columns)
        index = list(factors.index)
        arr = factors.to_numpy(dtype=np.float64)
    else:
        columns = None
        index = None
        arr = np.asarray(factors, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError("Factors must be a 2D matrix of shape (N_assets, K_factors).")

    # Cross-sectional z-score per factor
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    norm_arr = (arr - mean) / std
    return norm_arr, columns, index


def equal_weight_combination(
    factors: Union[np.ndarray, pd.DataFrame],
    normalize_factors: bool = True,
) -> Tuple[Union[np.ndarray, pd.Series], np.ndarray]:
    """
    Combine factors with equal weighting: w_k = 1 / K.

    Parameters
    ----------
    factors : Union[np.ndarray, pd.DataFrame]
        Factor exposure matrix of shape (N, K).
    normalize_factors : bool, default=True
        Whether to z-score standardize factors cross-sectionally.

    Returns
    -------
    Tuple[Union[np.ndarray, pd.Series], np.ndarray]
        (composite_alpha, weights_vector)
    """
    raw_arr, cols, idx = _normalize_factors(factors) if normalize_factors else (
        factors.to_numpy(dtype=np.float64) if isinstance(factors, pd.DataFrame) else np.asarray(factors, dtype=np.float64),
        list(factors.columns) if isinstance(factors, pd.DataFrame) else None,
        list(factors.index) if isinstance(factors, pd.DataFrame) else None,
    )

    N, K = raw_arr.shape
    weights = np.full(K, 1.0 / K, dtype=np.float64)
    composite = raw_arr @ weights

    if idx is not None:
        return pd.Series(composite, index=idx, name="composite_alpha"), weights
    return composite, weights


def ic_weighted_combination(
    factors: Union[np.ndarray, pd.DataFrame],
    ic_vector: Union[np.ndarray, pd.Series],
    long_only: bool = True,
    normalize_factors: bool = True,
) -> Tuple[Union[np.ndarray, pd.Series], np.ndarray]:
    """
    Combine factors proportional to their historical or estimated Information Coefficients.

    Parameters
    ----------
    factors : Union[np.ndarray, pd.DataFrame]
        Factor exposure matrix (N, K).
    ic_vector : Union[np.ndarray, pd.Series]
        IC vector of shape (K,).
    long_only : bool, default=True
        If True, non-positive IC factors receive 0 weight and remaining weights sum to 1.
        If False, weights are normalized by sum of absolute values.
    normalize_factors : bool, default=True
        Whether to z-score standardize factors cross-sectionally.

    Returns
    -------
    Tuple[Union[np.ndarray, pd.Series], np.ndarray]
        (composite_alpha, weights_vector)
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

    if long_only:
        pos_ic = np.maximum(ic, 0.0)
        s = np.sum(pos_ic)
        if s > 1e-12:
            weights = pos_ic / s
        else:
            weights = np.full(K, 1.0 / K, dtype=np.float64)
    else:
        abs_sum = np.sum(np.abs(ic))
        if abs_sum > 1e-12:
            weights = ic / abs_sum
        else:
            weights = np.full(K, 1.0 / K, dtype=np.float64)

    composite = raw_arr @ weights
    if idx is not None:
        return pd.Series(composite, index=idx, name="composite_alpha"), weights
    return composite, weights


def icir_weighted_combination(
    factors: Union[np.ndarray, pd.DataFrame],
    ic_history: Union[np.ndarray, pd.DataFrame],
    long_only: bool = True,
    normalize_factors: bool = True,
) -> Tuple[Union[np.ndarray, pd.Series], np.ndarray]:
    """
    Combine factors weighted by their Information Ratio (mean(IC) / std(IC)).

    Parameters
    ----------
    factors : Union[np.ndarray, pd.DataFrame]
        Factor exposure matrix (N, K).
    ic_history : Union[np.ndarray, pd.DataFrame]
        Historical IC time series of shape (T, K).
    long_only : bool, default=True
        If True, negative IR factors receive 0 weight.
    normalize_factors : bool, default=True
        Whether to z-score standardize factors cross-sectionally.

    Returns
    -------
    Tuple[Union[np.ndarray, pd.Series], np.ndarray]
        (composite_alpha, weights_vector)
    """
    ic_hist = np.asarray(ic_history, dtype=np.float64)
    if ic_hist.ndim != 2:
        raise ValueError("ic_history must be a 2D matrix of shape (T, K).")
    mean_ic = np.mean(ic_hist, axis=0)
    std_ic = np.std(ic_hist, axis=0, ddof=1)
    std_ic = np.where(std_ic < 1e-12, 1.0, std_ic)
    ir_vector = mean_ic / std_ic

    return ic_weighted_combination(factors, ir_vector, long_only=long_only, normalize_factors=normalize_factors)


def inverse_variance_combination(
    factors: Union[np.ndarray, pd.DataFrame],
    covariance: np.ndarray,
    normalize_factors: bool = True,
) -> Tuple[Union[np.ndarray, pd.Series], np.ndarray]:
    r"""
    Combine factors inversely proportional to their variance: w_k \propto 1 / sigma_k^2.

    Parameters
    ----------
    factors : Union[np.ndarray, pd.DataFrame]
        Factor exposure matrix (N, K).
    covariance : np.ndarray
        Alpha covariance matrix (K, K).
    normalize_factors : bool, default=True
        Whether to z-score standardize factors cross-sectionally.

    Returns
    -------
    Tuple[Union[np.ndarray, pd.Series], np.ndarray]
        (composite_alpha, weights_vector)
    """
    cov = np.asarray(covariance, dtype=np.float64)
    var = np.diag(cov)
    inv_var = 1.0 / np.maximum(var, 1e-12)
    weights = inv_var / np.sum(inv_var)

    raw_arr, cols, idx = _normalize_factors(factors) if normalize_factors else (
        factors.to_numpy(dtype=np.float64) if isinstance(factors, pd.DataFrame) else np.asarray(factors, dtype=np.float64),
        list(factors.columns) if isinstance(factors, pd.DataFrame) else None,
        list(factors.index) if isinstance(factors, pd.DataFrame) else None,
    )
    composite = raw_arr @ weights
    if idx is not None:
        return pd.Series(composite, index=idx, name="composite_alpha"), weights
    return composite, weights


def grinold_kahn_combination(
    factors: Union[np.ndarray, pd.DataFrame],
    ic_vector: Union[np.ndarray, pd.Series],
    covariance: np.ndarray,
    long_only: bool = True,
    normalize_factors: bool = True,
) -> Tuple[Union[np.ndarray, pd.Series], np.ndarray]:
    """
    Grinold-Kahn analytical optimal alpha combination:
        w^* = Omega^{-1} * ic

    Under orthonormal factors (OrthoQuant: Omega = I_K), this simplifies
    elegantly to w^* = ic.

    Parameters
    ----------
    factors : Union[np.ndarray, pd.DataFrame]
        Factor exposure matrix (N, K).
    ic_vector : Union[np.ndarray, pd.Series]
        Information coefficient vector (K,).
    covariance : np.ndarray
        Covariance matrix of factors or ICs (K, K).
    long_only : bool, default=True
        Whether to project resulting weights onto the probability simplex {w >= 0, sum(w) = 1}.
    normalize_factors : bool, default=True
        Whether to z-score standardize factors cross-sectionally.

    Returns
    -------
    Tuple[Union[np.ndarray, pd.Series], np.ndarray]
        (composite_alpha, weights_vector)
    """
    ic = np.asarray(ic_vector, dtype=np.float64).ravel()
    cov = np.asarray(covariance, dtype=np.float64)
    K = len(ic)
    if cov.shape != (K, K):
        raise ValueError(f"Covariance shape {cov.shape} does not match IC length {K}.")

    # Solve Omega * w = ic via pseudo-inverse / Cholesky for stability
    try:
        raw_weights = np.linalg.solve(cov + 1e-7 * np.eye(K), ic)
    except np.linalg.LinAlgError:
        raw_weights = np.linalg.pinv(cov) @ ic

    if long_only:
        weights = project_simplex(raw_weights, s=1.0)
    else:
        abs_sum = np.sum(np.abs(raw_weights))
        weights = raw_weights / abs_sum if abs_sum > 1e-12 else np.full(K, 1.0 / K)

    raw_arr, cols, idx = _normalize_factors(factors) if normalize_factors else (
        factors.to_numpy(dtype=np.float64) if isinstance(factors, pd.DataFrame) else np.asarray(factors, dtype=np.float64),
        list(factors.columns) if isinstance(factors, pd.DataFrame) else None,
        list(factors.index) if isinstance(factors, pd.DataFrame) else None,
    )
    composite = raw_arr @ weights
    if idx is not None:
        return pd.Series(composite, index=idx, name="composite_alpha"), weights
    return composite, weights
