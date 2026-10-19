"""
Synthetic multi-factor market generator for alpha combination research.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def generate_single_period_factors(
    n_assets: int = 100,
    n_factors: int = 5,
    base_correlation: float = 0.35,
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Generate synthetic cross-sectional factor panel and forward returns for a single period.

    Parameters
    ----------
    n_assets : int, default=100
        Number of assets in universe.
    n_factors : int, default=5
        Number of alpha factors.
    base_correlation : float, default=0.35
        Average correlation between candidate factors.
    random_seed : Optional[int], default=42
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'factors': pd.DataFrame of shape (N, K)
        - 'forward_returns': pd.Series of shape (N,)
        - 'true_weights': np.ndarray of shape (K,)
        - 'covariance': np.ndarray of shape (K, K)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Construct equicorrelated covariance matrix among factors
    corr = np.full((n_factors, n_factors), base_correlation)
    np.fill_diagonal(corr, 1.0)
    cholesky = np.linalg.cholesky(corr)

    # Generate correlated factor scores
    raw_z = np.random.randn(n_assets, n_factors)
    factors_mat = raw_z @ cholesky.T

    # Normalize cross-sectionally
    factors_mat = (factors_mat - np.mean(factors_mat, axis=0)) / np.std(factors_mat, axis=0)

    # True latent weights generating returns
    raw_true_w = np.linspace(0.4, 0.05, n_factors)
    true_weights = raw_true_w / np.sum(raw_true_w)

    # True composite signal + idiosyncratic noise
    true_signal = factors_mat @ true_weights
    noise = np.random.randn(n_assets) * 1.5
    forward_returns = 0.05 * true_signal + 0.08 * noise

    asset_ids = [f"ASSET_{i:04d}" for i in range(n_assets)]
    factor_ids = [f"ALPHA_{k+1}" for k in range(n_factors)]

    factors_df = pd.DataFrame(factors_mat, index=asset_ids, columns=factor_ids)
    returns_series = pd.Series(forward_returns, index=asset_ids, name="forward_returns")

    return {
        "factors": factors_df,
        "forward_returns": returns_series,
        "true_weights": true_weights,
        "covariance": corr,
    }


def generate_synthetic_alpha_panel(
    n_assets: int = 100,
    n_factors: int = 5,
    n_periods: int = 120,
    base_correlation: float = 0.35,
    random_seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Generate multi-period panel of factors, forward returns, and historical IC trajectories.

    Parameters
    ----------
    n_assets : int, default=100
        Number of assets.
    n_factors : int, default=5
        Number of alpha factors.
    n_periods : int, default=120
        Number of time steps (e.g. trading days).
    base_correlation : float, default=0.35
        Average inter-factor correlation.
    random_seed : Optional[int], default=42
        Random seed.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'factors_panel': List of pd.DataFrame (one per period)
        - 'returns_panel': List of pd.Series (one per period)
        - 'ic_history': pd.DataFrame of historical factor ICs of shape (n_periods, n_factors)
        - 'factor_covariance': np.ndarray of shape (n_factors, n_factors)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    corr = np.full((n_factors, n_factors), base_correlation)
    np.fill_diagonal(corr, 1.0)
    cholesky = np.linalg.cholesky(corr)

    asset_ids = [f"ASSET_{i:04d}" for i in range(n_assets)]
    factor_ids = [f"ALPHA_{k+1}" for k in range(n_factors)]
    dates = pd.date_range("2026-01-01", periods=n_periods, freq="B")

    # True latent predictive strengths
    base_signal_strength = np.linspace(0.08, 0.01, n_factors)

    factors_list = []
    returns_list = []
    ic_records = []

    for t in range(n_periods):
        raw_z = np.random.randn(n_assets, n_factors)
        f_t = raw_z @ cholesky.T
        f_t = (f_t - np.mean(f_t, axis=0)) / np.std(f_t, axis=0)

        # Dynamic return generation
        signal = f_t @ base_signal_strength
        ret_t = 0.06 * signal + 0.10 * np.random.randn(n_assets)

        f_df = pd.DataFrame(f_t, index=asset_ids, columns=factor_ids)
        r_sr = pd.Series(ret_t, index=asset_ids, name=f"ret_{t}")

        factors_list.append(f_df)
        returns_list.append(r_sr)

        # Record realized ICs for each factor
        ic_t = [float(np.corrcoef(f_df[col], r_sr)[0, 1]) for col in factor_ids]
        ic_records.append(ic_t)

    ic_history_df = pd.DataFrame(ic_records, index=dates, columns=factor_ids)

    return {
        "factors_panel": factors_list,
        "returns_panel": returns_list,
        "ic_history": ic_history_df,
        "factor_covariance": corr,
    }
