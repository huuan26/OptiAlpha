"""
Unit tests for covariance estimation and shrinkage module.
"""

import pytest
import numpy as np
import pandas as pd
from optialpha.covariance.shrinkage import (
    sample_covariance,
    nearest_spd,
    ledoit_wolf_shrinkage,
    exponential_covariance,
)


def test_sample_covariance():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    cov = sample_covariance(X)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T)
    # Check against numpy cov
    np.testing.assert_allclose(cov, np.cov(X, rowvar=False, ddof=1))


def test_sample_covariance_errors():
    with pytest.raises(ValueError, match="2D matrix"):
        sample_covariance(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="must exceed ddof"):
        sample_covariance(np.random.randn(1, 4), ddof=1)


def test_nearest_spd():
    # Symmetric indefinite matrix
    A = np.array([[1.0, 2.0], [2.0, 1.0]])  # Eigenvalues are 3 and -1
    A_spd = nearest_spd(A, eps=1e-5)

    eigvals = np.linalg.eigvalsh(A_spd)
    assert np.all(eigvals >= 1e-5)
    assert np.allclose(A_spd, A_spd.T)


def test_nearest_spd_errors():
    with pytest.raises(ValueError, match="must be square"):
        nearest_spd(np.random.randn(3, 4))


def test_ledoit_wolf_constant_correlation():
    np.random.seed(42)
    T, K = 60, 5
    data = np.random.randn(T, K)
    shrunk_cov, delta = ledoit_wolf_shrinkage(data, target="constant_correlation")

    assert shrunk_cov.shape == (K, K)
    assert 0.0 <= delta <= 1.0
    eigvals = np.linalg.eigvalsh(shrunk_cov)
    assert np.all(eigvals > 0)


def test_ledoit_wolf_identity():
    np.random.seed(42)
    T, K = 80, 4
    data = np.random.randn(T, K)
    shrunk_cov, delta = ledoit_wolf_shrinkage(data, target="identity")

    assert shrunk_cov.shape == (K, K)
    assert 0.0 <= delta <= 1.0
    eigvals = np.linalg.eigvalsh(shrunk_cov)
    assert np.all(eigvals > 0)


def test_ledoit_wolf_invalid_target_and_length():
    with pytest.raises(ValueError, match="Unknown shrinkage target"):
        ledoit_wolf_shrinkage(np.random.randn(20, 3), target="unsupported")
    with pytest.raises(ValueError, match="at least T=4"):
        ledoit_wolf_shrinkage(np.random.randn(2, 3))


def test_exponential_covariance():
    np.random.seed(42)
    T, K = 100, 3
    data = np.random.randn(T, K)
    exp_cov = exponential_covariance(data, half_life=30.0)

    assert exp_cov.shape == (K, K)
    assert np.allclose(exp_cov, exp_cov.T)
    eigvals = np.linalg.eigvalsh(exp_cov)
    assert np.all(eigvals > 0)


def test_exponential_covariance_errors():
    with pytest.raises(ValueError, match="(?i)half-life must be strictly positive"):
        exponential_covariance(np.random.randn(20, 3), half_life=-10)
    with pytest.raises(ValueError, match="at least T=2"):
        exponential_covariance(np.random.randn(1, 3))
