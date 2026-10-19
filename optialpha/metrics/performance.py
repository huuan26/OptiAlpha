"""
Performance and predictive evaluation metrics for combined alpha signals.
"""

from typing import Union
import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(
    alpha: Union[np.ndarray, pd.Series],
    forward_returns: Union[np.ndarray, pd.Series],
    method: str = "spearman",
) -> float:
    """
    Compute cross-sectional Information Coefficient (IC) between alpha and forward returns.

    Parameters
    ----------
    alpha : Union[np.ndarray, pd.Series]
        Alpha signal scores of shape (N,).
    forward_returns : Union[np.ndarray, pd.Series]
        Realized forward returns of shape (N,).
    method : str, default="spearman"
        Correlation metric: 'spearman' (Rank IC) or 'pearson' (Normal IC).

    Returns
    -------
    float
        Information coefficient in [-1, 1].
    """
    x = np.asarray(alpha, dtype=np.float64).ravel()
    y = np.asarray(forward_returns, dtype=np.float64).ravel()

    if len(x) != len(y):
        raise ValueError(f"Length mismatch: alpha has {len(x)} elements, returns has {len(y)} elements.")
    if len(x) < 3:
        return 0.0

    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    if len(x_clean) < 3 or np.all(x_clean == x_clean[0]) or np.all(y_clean == y_clean[0]):
        return 0.0

    if method == "spearman":
        corr, _ = stats.spearmanr(x_clean, y_clean)
    elif method == "pearson":
        corr, _ = stats.pearsonr(x_clean, y_clean)
    else:
        raise ValueError(f"Unknown correlation method '{method}'. Choose 'spearman' or 'pearson'.")

    return float(corr) if np.isfinite(corr) else 0.0


def information_ratio(
    ic_series: Union[np.ndarray, pd.Series, list],
    annualization_factor: float = np.sqrt(252.0),
) -> float:
    """
    Compute annualized Information Ratio (ICIR) of an alpha strategy:
        ICIR = mean(IC) / std(IC) * sqrt(periods_per_year)

    Parameters
    ----------
    ic_series : Union[np.ndarray, pd.Series, list]
        Time series of IC values across T periods.
    annualization_factor : float, default=sqrt(252)
        Annualization multiplier.

    Returns
    -------
    float
        Annualized Information Ratio.
    """
    ics = np.asarray(ic_series, dtype=np.float64).ravel()
    ics = ics[np.isfinite(ics)]
    if len(ics) < 2:
        return 0.0
    mean_ic = np.mean(ics)
    std_ic = np.std(ics, ddof=1)
    if std_ic < 1e-12:
        return 0.0
    return float((mean_ic / std_ic) * annualization_factor)


def t_statistic_ic(ic_series: Union[np.ndarray, pd.Series, list]) -> float:
    """
    Compute Student's t-statistic for the null hypothesis H0: E[IC] == 0.

    Parameters
    ----------
    ic_series : Union[np.ndarray, pd.Series, list]
        Historical IC time series.

    Returns
    -------
    float
        T-statistic value.
    """
    ics = np.asarray(ic_series, dtype=np.float64).ravel()
    ics = ics[np.isfinite(ics)]
    n = len(ics)
    if n < 2:
        return 0.0
    mean_ic = np.mean(ics)
    std_err = stats.sem(ics)
    if std_err < 1e-12:
        return 0.0
    return float(mean_ic / std_err)


def turnover_ratio(weights_trajectory: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Compute average turnover per period:
        Turnover = 1 / (T - 1) sum_{t=2}^T 0.5 * ||w_t - w_{t-1}||_1

    Parameters
    ----------
    weights_trajectory : Union[np.ndarray, pd.DataFrame]
        Trajectory array of shape (T_periods, K_factors).

    Returns
    -------
    float
        Average two-sided turnover ratio per period in [0, 1].
    """
    W = np.asarray(weights_trajectory, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("weights_trajectory must be a 2D matrix of shape (T, K).")
    T = W.shape[0]
    if T < 2:
        return 0.0
    deltas = np.abs(W[1:] - W[:-1])
    turnovers = 0.5 * np.sum(deltas, axis=1)
    return float(np.mean(turnovers))


def max_drawdown(cumulative_series: Union[np.ndarray, pd.Series]) -> float:
    """
    Compute maximum peak-to-trough drawdown of a cumulative return or PnL series.

    Parameters
    ----------
    cumulative_series : Union[np.ndarray, pd.Series]
        Cumulative values or equity curve.

    Returns
    -------
    float
        Maximum drawdown (positive decimal, e.g. 0.15 = 15% drawdown).
    """
    s = np.asarray(cumulative_series, dtype=np.float64).ravel()
    if len(s) < 2:
        return 0.0
    running_max = np.maximum.accumulate(s)
    drawdowns = (running_max - s) / np.where(running_max > 0, running_max, 1.0)
    return float(np.max(drawdowns))
