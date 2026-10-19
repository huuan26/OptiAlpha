"""
Unit tests for synthetic data generation module.
"""

import numpy as np
import pandas as pd
from optialpha.data.synthetic import (
    generate_single_period_factors,
    generate_synthetic_alpha_panel,
)


def test_generate_single_period_factors():
    data = generate_single_period_factors(n_assets=60, n_factors=4, base_correlation=0.4, random_seed=42)

    assert "factors" in data
    assert "forward_returns" in data
    assert "true_weights" in data
    assert "covariance" in data

    factors = data["factors"]
    returns = data["forward_returns"]

    assert isinstance(factors, pd.DataFrame)
    assert factors.shape == (60, 4)
    assert len(returns) == 60
    assert np.isclose(np.sum(data["true_weights"]), 1.0)


def test_generate_synthetic_alpha_panel():
    panel = generate_synthetic_alpha_panel(
        n_assets=50,
        n_factors=3,
        n_periods=40,
        base_correlation=0.3,
        random_seed=42,
    )

    assert len(panel["factors_panel"]) == 40
    assert len(panel["returns_panel"]) == 40
    assert panel["ic_history"].shape == (40, 3)
    assert panel["factor_covariance"].shape == (3, 3)
