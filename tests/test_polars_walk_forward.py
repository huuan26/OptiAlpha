"""Tests for Polars & polars-anta Walk-Forward Multi-Factor Allocation in OptiAlpha."""

import numpy as np
import polars as pl
import pytest

from optialpha import (
    PolarsWalkForwardCombiner,
    extract_polars_anta_panel_factors,
)


def _make_ohlcv_panel(n_symbols: int = 25, n_bars: int = 35, seed: int = 123) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s_idx in range(n_symbols):
        sym = f"ASSET_{s_idx:02d}"
        price = 50.0 + s_idx
        for t in range(n_bars):
            ret = rng.normal(0.0005, 0.018)
            open_p = price
            close_p = price * (1.0 + ret)
            high_p = max(open_p, close_p) * (1.0 + abs(rng.normal(0.0, 0.004)))
            low_p = min(open_p, close_p) * (1.0 - abs(rng.normal(0.0, 0.004)))
            vol = float(rng.integers(5_000, 80_000))
            rows.append(
                {
                    "timestamp": t,
                    "symbol": sym,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": vol,
                }
            )
            price = close_p
    return pl.DataFrame(rows)


def test_polars_anta_walk_forward_ledoit_wolf():
    """Verify end-to-end polars-anta factor extraction + Ledoit-Wolf + FISTA walk-forward."""
    raw_panel = _make_ohlcv_panel(n_symbols=25, n_bars=35)
    enriched, factor_cols = extract_polars_anta_panel_factors(raw_panel, period=6)

    assert len(factor_cols) == 5
    assert "fwd_ret" in enriched.columns

    wf = PolarsWalkForwardCombiner(
        lookback=10,
        risk_aversion=2.0,
        turnover_penalty=0.05,
        lower_bound=0.0,
        upper_bound=0.6,
        budget=1.0,
        cov_estimator="ledoit_wolf",
        ic_ema_halflife=5.0,
    )
    res = wf.fit_transform(enriched, factor_cols=factor_cols, return_col="fwd_ret")

    assert "composite_alpha" in res.panel_df.columns
    assert "equal_weight_alpha" in res.panel_df.columns
    assert res.weights_history.height > 0
    assert res.ic_matrix_df.height > 10
    assert 0.0 <= res.mean_shrinkage_delta <= 1.0
    assert res.mean_turnover >= 0.0

    w_cols = [f"w_{c}" for c in factor_cols]
    weight_sums = res.weights_history.select(pl.sum_horizontal(w_cols)).to_numpy().ravel()
    np.testing.assert_allclose(weight_sums, 1.0, atol=1e-6)

    summary = res.summary()
    assert "fista_ic_mean" in summary
    assert "ew_ic_mean" in summary
    assert summary["n_oos_periods"] > 0


def test_polars_walk_forward_alternative_estimators_and_errors():
    """Verify exponential and sample covariance estimators and validation checks."""
    raw_panel = _make_ohlcv_panel(n_symbols=20, n_bars=30)
    enriched, factor_cols = extract_polars_anta_panel_factors(raw_panel, period=5)

    for est in ("exponential", "sample"):
        wf = PolarsWalkForwardCombiner(lookback=8, cov_estimator=est)
        res = wf.fit_transform(enriched, factor_cols=factor_cols[:3])
        assert res.weights_history.height > 0

    with pytest.raises(ValueError, match="lookback window must be at least 5"):
        PolarsWalkForwardCombiner(lookback=3)

    with pytest.raises(ValueError, match="Unknown cov_estimator"):
        PolarsWalkForwardCombiner(cov_estimator="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        extract_polars_anta_panel_factors(pl.DataFrame({"symbol": ["A"]}))

    wf_valid = PolarsWalkForwardCombiner(lookback=8)
    with pytest.raises(ValueError, match="At least 2 factor columns"):
        wf_valid.fit_transform(enriched, factor_cols=["alpha_rsi"])

    with pytest.raises(ValueError, match="must exceed lookback"):
        wf_long = PolarsWalkForwardCombiner(lookback=100)
        wf_long.fit_transform(enriched, factor_cols=factor_cols[:2])
