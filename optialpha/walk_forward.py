"""Polars & polars-anta Walk-Forward Multi-Factor Allocation Pipeline.

Bridges `polars-anta` (Rust-accelerated factor extraction, cross-sectional winsorization/z-score,
Rank IC computation, and Newey-West HAC evaluation) with `OptiAlpha` (analytical Ledoit-Wolf
covariance shrinkage and FISTA $L_1$-turnover-penalized convex combination) across
multi-period Polars panel DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np
import polars as pl
import polars_anta as pa  # noqa: F401 - registers .anta and .ta namespaces

from optialpha.combination.convex import ConvexAlphaCombiner
from optialpha.covariance.shrinkage import (
    exponential_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
)


@dataclass
class WalkForwardAllocationResult:
    """Container for walk-forward Polars multi-factor combination outputs."""

    panel_df: pl.DataFrame
    weights_history: pl.DataFrame
    ic_matrix_df: pl.DataFrame
    composite_ic_summary: Dict[str, float]
    equal_weight_ic_summary: Dict[str, float]
    mean_turnover: float
    mean_shrinkage_delta: float

    def summary(self) -> Dict[str, float]:
        """Return a diagnostic summary comparing FISTA composite vs. equal-weight baseline."""
        return {
            "fista_ic_mean": round(self.composite_ic_summary["ic_mean"], 6),
            "fista_icir": round(self.composite_ic_summary["icir"], 6),
            "fista_nw_t_stat": round(self.composite_ic_summary["nw_t_stat"], 4),
            "ew_ic_mean": round(self.equal_weight_ic_summary["ic_mean"], 6),
            "ew_icir": round(self.equal_weight_ic_summary["icir"], 6),
            "ew_nw_t_stat": round(self.equal_weight_ic_summary["nw_t_stat"], 4),
            "mean_turnover": round(self.mean_turnover, 6),
            "mean_shrinkage_delta": round(self.mean_shrinkage_delta, 6),
            "n_oos_periods": int(self.composite_ic_summary["n_periods"]),
        }


def extract_polars_anta_panel_factors(
    df: pl.DataFrame,
    symbol_col: str = "symbol",
    time_col: str = "timestamp",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    period: int = 10,
    forward_horizon: int = 1,
    forward_return_col: str = "fwd_ret",
) -> Tuple[pl.DataFrame, List[str]]:
    """Extract standardized cross-sectional alpha factors and forward returns via `polars-anta`.

    Parameters
    ----------
    df : pl.DataFrame
        Input OHLCV Polars panel DataFrame.
    symbol_col : str, default 'symbol'
        Asset identifier column.
    time_col : str, default 'timestamp'
        Cross-sectional timestamp/date column.
    open_col, high_col, low_col, close_col, volume_col : str
        OHLCV column names.
    period : int, default 10
        Lookback window for technical and microstructure primitives.
    forward_horizon : int, default 1
        Forward return horizon in bars if `forward_return_col` is not already present.
    forward_return_col : str, default 'fwd_ret'
        Name of the forward return column.

    Returns
    -------
    Tuple[pl.DataFrame, List[str]]
        (Polars DataFrame with standardized factor columns and forward returns, factor column names).
    """
    required = {symbol_col, time_col, open_col, high_col, low_col, close_col, volume_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    sorted_df = df.sort([symbol_col, time_col])
    factor_cols = ["alpha_rsi", "alpha_roc", "alpha_vpin", "alpha_101", "alpha_12"]

    enriched = sorted_df.with_columns(
        pl.col(close_col).anta.rsi(period=period).over(symbol_col).alias("alpha_rsi"),
        pl.col(close_col).anta.roc(period=period).over(symbol_col).alias("alpha_roc"),
        pa.vpin(pl.col(close_col), pl.col(volume_col), period=period)
        .over(symbol_col)
        .alias("alpha_vpin"),
        pa.alpha_101(
            pl.col(open_col),
            pl.col(high_col),
            pl.col(low_col),
            pl.col(close_col),
        )
        .over(symbol_col)
        .alias("alpha_101"),
        pa.alpha_012(pl.col(close_col), pl.col(volume_col))
        .over(symbol_col)
        .alias("alpha_12"),
    )

    if forward_return_col not in enriched.columns:
        enriched = enriched.with_columns(
            ((pl.col(close_col).shift(-forward_horizon).over(symbol_col) / pl.col(close_col)) - 1.0).alias(
                forward_return_col
            )
        )

    enriched = enriched.with_columns(
        [
            pa.cs_zscore(
                pa.cs_winsorize(pl.col(c), over=time_col, method="mad", threshold=3.0),
                over=time_col,
            ).alias(c)
            for c in factor_cols
        ]
    )

    return enriched.sort([time_col, symbol_col]), factor_cols


class PolarsWalkForwardCombiner:
    """Walk-forward multi-factor allocation engine powered by `polars-anta` and FISTA.

    At each out-of-sample timestamp $t \\ge \\text{lookback}$, estimates the expected
    factor Information Coefficient vector $\\hat{\\mathbf{ic}}_t$ and shrunk factor covariance
    matrix $\\hat{\\mathbf{\\Omega}}_{\\alpha, t}$ strictly from trailing realized cross-sectional
    Rank ICs over $[t - \\text{lookback}, t - 1]$, then solves the FISTA convex optimization
    problem with $L_1$ turnover control against $\\mathbf{w}_{t-1}$.
    """

    def __init__(
        self,
        lookback: int = 20,
        risk_aversion: float = 2.0,
        turnover_penalty: float = 0.05,
        l2_reg: float = 1e-4,
        lower_bound: float = 0.0,
        upper_bound: float = 0.6,
        budget: float = 1.0,
        cov_estimator: Literal["ledoit_wolf", "exponential", "sample"] = "ledoit_wolf",
        shrinkage_target: Literal["constant_correlation", "identity"] = "constant_correlation",
        ic_ema_halflife: Optional[float] = None,
        solver: str = "fista",
    ):
        if lookback < 5:
            raise ValueError("lookback window must be at least 5 periods.")
        if cov_estimator not in ("ledoit_wolf", "exponential", "sample"):
            raise ValueError(
                f"Unknown cov_estimator '{cov_estimator}'. Choose 'ledoit_wolf', 'exponential', or 'sample'."
            )

        self.lookback = lookback
        self.risk_aversion = risk_aversion
        self.turnover_penalty = turnover_penalty
        self.l2_reg = l2_reg
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.budget = budget
        self.cov_estimator = cov_estimator
        self.shrinkage_target = shrinkage_target
        self.ic_ema_halflife = ic_ema_halflife
        self.solver = solver

        self._combiner = ConvexAlphaCombiner(
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
            l2_reg=l2_reg,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            budget=budget,
            solver=solver,
        )

    def _compute_ic_matrix(
        self,
        df: pl.DataFrame,
        factor_cols: Sequence[str],
        return_col: str,
        time_col: str,
        method: Literal["spearman", "pearson"] = "spearman",
    ) -> pl.DataFrame:
        """Compute cross-sectional IC time series for all K factors using `polars-anta`."""
        ic_frames: List[pl.DataFrame] = []
        for col in factor_cols:
            ic_df = (
                pa.information_coefficient(
                    df,
                    factor_col=col,
                    return_col=return_col,
                    group_by=time_col,
                    method=method,
                )
                .select([pl.col(time_col), pl.col("ic").alias(col)])
                .sort(time_col)
            )
            ic_frames.append(ic_df)

        merged = ic_frames[0]
        for nxt in ic_frames[1:]:
            merged = merged.join(nxt, on=time_col, how="inner")
        return merged.drop_nulls().sort(time_col)

    def fit_transform(
        self,
        df: pl.DataFrame,
        factor_cols: Sequence[str],
        return_col: str = "fwd_ret",
        time_col: str = "timestamp",
        symbol_col: str = "symbol",
        ic_method: Literal["spearman", "pearson"] = "spearman",
        nw_lags: int = 5,
        composite_col: str = "composite_alpha",
        equal_weight_col: str = "equal_weight_alpha",
    ) -> WalkForwardAllocationResult:
        """Run walk-forward factor allocation and generate out-of-sample composite signals.

        Parameters
        ----------
        df : pl.DataFrame
            Polars panel DataFrame containing `factor_cols` and `return_col`.
        factor_cols : Sequence[str]
            Names of the $K$ candidate alpha factor columns.
        return_col : str, default 'fwd_ret'
            Forward return column used to compute realized cross-sectional ICs.
        time_col : str, default 'timestamp'
            Timestamp/date grouping column.
        symbol_col : str, default 'symbol'
            Asset identifier column.
        ic_method : {'spearman', 'pearson'}, default 'spearman'
            Correlation method for cross-sectional IC estimation.
        nw_lags : int, default 5
            Newey-West lags for `polars-anta` out-of-sample IC evaluation.
        composite_col : str, default 'composite_alpha'
            Output column name for the FISTA composite signal.
        equal_weight_col : str, default 'equal_weight_alpha'
            Output column name for the 1/K equal-weight benchmark signal.

        Returns
        -------
        WalkForwardAllocationResult
            Walk-forward allocation results, weight history, and `polars-anta` IC diagnostics.
        """
        factor_list = list(factor_cols)
        K = len(factor_list)
        if K < 2:
            raise ValueError("At least 2 factor columns are required for multi-factor combination.")

        sorted_df = df.sort([time_col, symbol_col])
        ic_matrix_df = self._compute_ic_matrix(
            sorted_df,
            factor_cols=factor_list,
            return_col=return_col,
            time_col=time_col,
            method=ic_method,
        )

        if ic_matrix_df.height <= self.lookback:
            raise ValueError(
                f"Valid IC periods ({ic_matrix_df.height}) must exceed lookback ({self.lookback})."
            )

        ic_times = ic_matrix_df.get_column(time_col).to_list()
        ic_values = ic_matrix_df.select(factor_list).to_numpy().astype(np.float64)

        w_prev = np.full(K, self.budget / K, dtype=np.float64)
        weights_by_time: Dict[object, np.ndarray] = {}
        weight_rows: List[Dict[str, object]] = []
        turnovers: List[float] = []
        deltas: List[float] = []

        for idx in range(self.lookback, len(ic_times)):
            t_val = ic_times[idx]
            window_ic = ic_values[idx - self.lookback : idx]

            if self.ic_ema_halflife is not None and self.ic_ema_halflife > 0:
                decay = np.exp(-np.log(2.0) / self.ic_ema_halflife)
                ema_w = np.array([decay ** (self.lookback - 1 - i) for i in range(self.lookback)])
                ema_w /= np.sum(ema_w)
                expected_ic = np.sum(window_ic * ema_w[:, np.newaxis], axis=0)
            else:
                expected_ic = np.mean(window_ic, axis=0)

            if self.cov_estimator == "ledoit_wolf":
                cov_mat, delta = ledoit_wolf_shrinkage(window_ic, target=self.shrinkage_target)
                deltas.append(delta)
            elif self.cov_estimator == "exponential":
                cov_mat = exponential_covariance(window_ic, half_life=max(5.0, self.lookback / 2.0))
                deltas.append(0.0)
            else:
                cov_mat = sample_covariance(window_ic)
                deltas.append(0.0)

            # Dummy identity factor snapshot to solve strictly for weights w_opt
            dummy_F = np.eye(K, dtype=np.float64)
            opt_res = self._combiner.optimize(
                factors=dummy_F,
                ic_vector=expected_ic,
                covariance=cov_mat,
                w_prev=w_prev,
                normalize_factors=False,
            )

            w_opt = opt_res.weights
            weights_by_time[t_val] = w_opt
            turnovers.append(opt_res.turnover)
            w_prev = w_opt

            row_dict: Dict[str, object] = {
                time_col: t_val,
                "turnover": opt_res.turnover,
                "predicted_ic": opt_res.predicted_ic,
                "converged": opt_res.converged,
            }
            for k_idx, col_name in enumerate(factor_list):
                row_dict[f"w_{col_name}"] = float(w_opt[k_idx])
            weight_rows.append(row_dict)

        weights_history_df = pl.DataFrame(weight_rows)

        # Join time-varying optimal weights onto panel_df and compute composite_alpha in Polars
        w_cols = [f"w_{c}" for c in factor_list]
        joined_df = sorted_df.join(
            weights_history_df.select([time_col] + w_cols),
            on=time_col,
            how="inner",
        )

        comp_expr = pl.sum_horizontal([pl.col(c) * pl.col(f"w_{c}") for c in factor_list])
        ew_expr = pl.mean_horizontal([pl.col(c) for c in factor_list]) * float(self.budget)

        out_panel = joined_df.with_columns(
            pa.cs_zscore(comp_expr, over=time_col).alias(composite_col),
            pa.cs_zscore(ew_expr, over=time_col).alias(equal_weight_col),
        )

        comp_summary = pa.ic_summary(
            out_panel,
            factor_col=composite_col,
            return_col=return_col,
            group_by=time_col,
            method=ic_method,
            nw_lags=nw_lags,
        )
        ew_summary = pa.ic_summary(
            out_panel,
            factor_col=equal_weight_col,
            return_col=return_col,
            group_by=time_col,
            method=ic_method,
            nw_lags=nw_lags,
        )

        return WalkForwardAllocationResult(
            panel_df=out_panel,
            weights_history=weights_history_df,
            ic_matrix_df=ic_matrix_df,
            composite_ic_summary=comp_summary,
            equal_weight_ic_summary=ew_summary,
            mean_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
            mean_shrinkage_delta=float(np.mean(deltas)) if deltas else 0.0,
        )
