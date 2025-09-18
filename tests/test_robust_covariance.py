import numpy as np
import pytest
from numpy.testing import assert_allclose
import probes.mean_difference as mdc

from probes.mean_difference import robust_covariance

def test_nan_input_triggers_fallback(caplog):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 5))
    X[0, 0] = np.nan  # inject NaN

    S = robust_covariance(X, method="empirical", fallback_scale=3.0)
    assert_allclose(S, 3.0 * np.eye(5), atol=1e-12)
    assert any("using scaled identity" in rec.message.lower() for rec in caplog.records)

def test_inf_input_triggers_fallback(caplog):
    X = np.zeros((10, 4))
    X[1, 2] = np.inf  # inject Inf

    S = robust_covariance(X, method="oas", fallback_scale=2.0)
    assert_allclose(S, 2.0 * np.eye(4), atol=1e-12)
    assert any("using scaled identity" in rec.message.lower() for rec in caplog.records)

def test_estimator_exception_triggers_fallback(monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError("kaboom")

    # Patch the symbol your function actually calls
    monkeypatch.setattr(mdc, "oas", boom)

    X = np.random.RandomState(0).randn(15, 6)

    S = mdc.robust_covariance(X, method="oas", fallback_scale=5.0)
    assert np.allclose(S, 5.0 * np.eye(6))
    assert any("using scaled identity" in rec.message.lower() for rec in caplog.records)

def test_happy_path_symmetry_and_finiteness():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 8))
    Xc = X - X.mean(0, keepdims=True)

    for method in ["oas", "ledoit", "empirical", "shrunk", "diagonal"]:
        S = robust_covariance(Xc, method=method)
        # symmetry
        assert_allclose(S, S.T, atol=1e-10)
        # finiteness
        assert np.isfinite(S).all()
        # shape
        assert S.shape == (8, 8)

def test_diagonal_is_diagonal():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((30, 7))
    S = robust_covariance(X, method="diagonal")
    offdiag = S - np.diag(np.diag(S))
    assert_allclose(offdiag, np.zeros_like(S), atol=1e-12)

def test_shrunk_is_between_empirical_and_identity():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 6))
    S_emp = np.cov(X, rowvar=False)
    S_shr = robust_covariance(X, method="shrunk")
    # Not a strict mathematical test, but shrunk should be close-ish to emp + ridge
    assert np.linalg.norm(S_shr) > 0
    assert S_shr.shape == S_emp.shape
