"""
Unit tests for performance and attribution metrics.
"""

import pytest
import numpy as np
import pandas as pd
from optialpha.metrics.performance import (
    information_coefficient,
    information_ratio,
    t_statistic_ic,
    turnover_ratio,
    max_drawdown,
)
from optialpha.metrics.attribution import (
    alpha_risk_contributions,
    effective_number_of_alphas,
    diversification_ratio,
)


def test_information_coefficient():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.2, 1.9, 3.1, 4.2, 5.0])

    spearman_ic = information_coefficient(x, y, method="spearman")
    pearson_ic = information_coefficient(x, y, method="pearson")

    assert np.isclose(spearman_ic, 1.0)
    assert 0.95 <= pearson_ic <= 1.0


def test_information_coefficient_edge_cases():
    assert information_coefficient([1.0, 1.0], [2.0, 3.0]) == 0.0
    assert information_coefficient([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    with pytest.raises(ValueError, match="Length mismatch"):
        information_coefficient([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="Unknown correlation method"):
        information_coefficient([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], method="unsupported")


def test_information_ratio_and_t_stat():
    np.random.seed(42)
    ics = np.random.normal(loc=0.04, scale=0.02, size=100)
    ir = information_ratio(ics)
    t_stat = t_statistic_ic(ics)

    assert ir > 0
    assert t_stat > 10.0  # Significant t-stat for mean=0.04, se=0.02/sqrt(100)=0.002

    assert information_ratio([]) == 0.0
    assert t_statistic_ic([]) == 0.0


def test_turnover_ratio():
    # Constant weights trajectory -> zero turnover
    w_const = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    assert turnover_ratio(w_const) == 0.0

    # Flipping weights between [1, 0] and [0, 1] -> turnover 1.0
    w_flip = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert turnover_ratio(w_flip) == 1.0

    assert turnover_ratio(np.array([[0.5, 0.5]])) == 0.0
    with pytest.raises(ValueError, match="2D matrix"):
        turnover_ratio(np.array([0.5, 0.5]))


def test_max_drawdown():
    series = np.array([100.0, 120.0, 90.0, 110.0, 80.0, 130.0])
    # Peak at 120, trough at 80 => (120 - 80) / 120 = 40/120 = 0.333333
    dd = max_drawdown(series)
    assert np.isclose(dd, 40.0 / 120.0)
    assert max_drawdown([100.0]) == 0.0


def test_alpha_risk_contributions():
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    w = np.array([0.6, 0.4])

    mrc, prc = alpha_risk_contributions(w, cov)
    assert len(mrc) == 2
    assert len(prc) == 2
    # Euler's theorem: sum of percentage risk contributions must equal 1.0
    assert np.isclose(np.sum(prc), 1.0)


def test_effective_number_of_alphas():
    # 4 equal risk alphas -> ENF == 4.0
    w_equal = np.array([0.25, 0.25, 0.25, 0.25])
    cov_diag = np.eye(4)
    enf_equal = effective_number_of_alphas(w_equal, cov_diag)
    assert np.isclose(enf_equal, 4.0)

    # 1 dominant alpha -> ENF approaches 1.0
    w_concentrated = np.array([0.999, 0.001])
    enf_conc = effective_number_of_alphas(w_concentrated)
    assert 1.0 <= enf_conc <= 1.05

    assert effective_number_of_alphas([]) == 0.0


def test_diversification_ratio():
    cov_uncorr = np.eye(3)
    w = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    # For uncorrelated assets: DR = sqrt(3) ~ 1.732
    dr = diversification_ratio(w, cov_uncorr)
    assert np.isclose(dr, np.sqrt(3.0))

    # Perfectly correlated assets: DR == 1.0
    cov_corr = np.ones((3, 3))
    dr_corr = diversification_ratio(w, cov_corr)
    assert np.isclose(dr_corr, 1.0)
