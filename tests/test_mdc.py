import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score

from probes.mean_difference import normalize, robust_covariance, MeanDifferenceClassifier


def rng(seed=0):
    return np.random.default_rng(seed)

def make_gaussian_binary(n=400, d=32, sep=1.0, cov_scale=1.0, seed=0, equal_cov=True):
    """
    Two Gaussians: class 1 mean shifted by 'sep' along e1.
    If equal_cov=False, make covariances slightly different to test robustness.
    """
    g = rng(seed)
    mu0 = np.zeros(d)
    mu1 = np.zeros(d); mu1[0] = sep
    A0 = g.standard_normal((d, d)); S0 = (A0 @ A0.T) / d
    A1 = g.standard_normal((d, d)); S1 = (A1 @ A1.T) / d
    if equal_cov:
        S1 = S0
    X0 = g.multivariate_normal(mu0, cov_scale*S0, size=n//2)
    X1 = g.multivariate_normal(mu1, cov_scale*S1, size=n - n//2)
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(len(X0), int), np.ones(len(X1), int)])
    return X, y, S0, S1, mu0, mu1

# ---------- normalize ----------

def test_normalize_vector_and_matrix():
    from probes.mean_difference import normalize  # replace with actual import
    v = np.array([3.0, 4.0])
    vm = normalize(v)
    assert_allclose(np.linalg.norm(vm), 1.0, atol=1e-8)

    M = np.array([[3.0, 4.0], [0.0, 5.0]])
    Mn = normalize(M)
    norms = np.linalg.norm(M, axis=1)
    assert_allclose(Mn[0], M[0] / norms[0], atol=1e-8)
    assert_allclose(Mn[1], M[1] / norms[1], atol=1e-8)

def test_normalize_handles_zero_vector():
    from probes.mean_difference import normalize
    z = np.zeros(5)
    out = normalize(z, tol=1e-9)
    # should not be NaN or inf; we accept zero vector back
    assert np.isfinite(out).all()

# ---------- robust_covariance ----------

@pytest.mark.parametrize("method", ["oas", "ledoit"])
def test_robust_covariance_symmetry_psd(method):
    from probes.mean_difference import robust_covariance
    X, _, _, _, _, _ = make_gaussian_binary(n=200, d=16, sep=0.0, seed=1)
    Xc = X - X.mean(0, keepdims=True)
    S = robust_covariance(Xc, method=method)
    # symmetry
    assert_allclose(S, S.T, atol=1e-8)
    # PSD-ish: all eigenvalues non-negative within small numerical tolerance
    w = np.linalg.eigvalsh(S)
    assert w.min() >= -1e-8

def test_robust_covariance_fallback_identity(monkeypatch):
    from probes.mean_difference import robust_covariance
    # Force an error by passing NaNs
    X = np.full((10, 4), np.nan)
    S = robust_covariance(X, method="oas", fallback_scale=3.0)
    assert_allclose(S, 3.0*np.eye(4), atol=1e-12)

# ---------- classifier core ----------

def test_fit_without_covariance_recovers_mean_direction():
    from probes.mean_difference import MeanDifferenceClassifier
    X, y, _, _, _, _ = make_gaussian_binary(n=500, d=8, sep=2.5, seed=0)
    clf = MeanDifferenceClassifier(with_covariance=False, fit_intercept=True).fit(X, y)
    w = clf.coef_.ravel()
    # direction should be aligned with e1
    cos = abs(w @ np.array([1, *([0]*(len(w)-1))])) / (np.linalg.norm(w) + 1e-12)
    assert cos > 0.95
    # accuracy should be well above chance
    acc = accuracy_score(y, clf.predict(X))
    assert acc > 0.85

def test_fit_with_covariance_matches_lda_when_equal_cov():
    from probes.mean_difference import MeanDifferenceClassifier
    X, y, _, _, _, _ = make_gaussian_binary(n=600, d=10, sep=1.2, seed=42, equal_cov=True)
    # Our classifier (Mahalanobis)
    mdc = MeanDifferenceClassifier(with_covariance=True, cov_reg=1e-8, fit_intercept=True, cov_type='empirical').fit(X, y)
    # sklearn LDA
    lda = LinearDiscriminantAnalysis(solver="svd").fit(X, y)
    # Compare decision scores up to affine transform (monotonicity via AUC)
    auc_mdc = roc_auc_score(y, mdc.predict_proba(X))
    auc_lda = roc_auc_score(y, lda.predict_proba(X)[:, 1])
    assert auc_mdc > 0.95 and auc_lda > 0.95
    # Directions roughly align
    w_mdc = mdc.coef_.ravel()
    w_lda = lda.coef_.ravel()
    cos = abs(np.dot(w_mdc, w_lda)) / (np.linalg.norm(w_mdc)*np.linalg.norm(w_lda))
    assert cos > 0.9

def test_intercept_equals_midpoint_projection():
    from probes.mean_difference import MeanDifferenceClassifier
    X, y, _, _, _, _ = make_gaussian_binary(n=300, d=6, sep=1.3, seed=3)
    clf = MeanDifferenceClassifier(with_covariance=False, fit_intercept=True).fit(X, y)
    w = clf.coef_.ravel()
    mu_p = X[y==1].mean(0)
    mu_n = X[y==0].mean(0)
    midpoint = -0.5 * ((mu_p + mu_n) @ w)
    assert_allclose(clf.intercept_, midpoint, atol=1e-6)

def test_predict_proba_in_0_1_and_monotone_with_score():
    from probes.mean_difference import MeanDifferenceClassifier
    X, y, _, _, _, _ = make_gaussian_binary(n=300, d=5, sep=1.0, seed=11)
    clf = MeanDifferenceClassifier(with_covariance=False).fit(X, y)
    proba = clf.predict_proba(X)
    assert ((0.0 <= proba) & (proba <= 1.0)).all()
    # monotonic: higher decision_function -> higher proba
    s = clf.decision_function(X)
    idx = np.argsort(s)
    assert np.all(np.diff(proba[idx]) >= -1e-10)

def test_external_mahalanobis_matrix_path():
    from probes.mean_difference import MeanDifferenceClassifier
    X, y, _, _, _, _ = make_gaussian_binary(n=400, d=8, sep=1.5, seed=9)
    # Build a simple diagonal M (pretend whitened)
    var = X.var(axis=0) + 1e-6
    M = np.diag(1.0 / var)
    clf = MeanDifferenceClassifier(with_covariance=True, fit_intercept=True).fit(X, y, M=M)
    # should produce sensible accuracy and not crash
    acc = accuracy_score(y, clf.predict(X))
    assert acc > 0.75

def test_handles_small_n_big_d_with_shrinkage():
    from probes.mean_difference import MeanDifferenceClassifier
    X, y, _, _, _, _ = make_gaussian_binary(n=50, d=128, sep=2.0, seed=7, equal_cov=True)
    # with_covariance=False should still work and be decent
    clf = MeanDifferenceClassifier(with_covariance=False).fit(X, y)
    acc = accuracy_score(y, clf.predict(X))
    assert acc > 0.7

def test_identical_classes_give_near_chance_prob():
    from probes.mean_difference import MeanDifferenceClassifier
    X, y, _, _, _, _ = make_gaussian_binary(n=300, d=12, sep=0.0, seed=5)  # identical means
    clf = MeanDifferenceClassifier(with_covariance=False).fit(X, y)
    p = clf.predict_proba(X)
    # Should cluster around 0.5; allow variance
    assert abs(p.mean() - 0.5) < 0.1

def test_fit_sets_attributes_and_shapes():
    from probes.mean_difference import MeanDifferenceClassifier
    X, y, _, _, _, _ = make_gaussian_binary(n=120, d=7, sep=1.0, seed=21)
    clf = MeanDifferenceClassifier().fit(X, y)
    assert clf.coef_.shape == (1, X.shape[1])
    assert np.isscalar(clf.intercept_)
    assert_equal(clf.classes_, np.array([0, 1]))

def test_decision_function_requires_fit():
    from probes.mean_difference import MeanDifferenceClassifier
    X, y, _, _, _, _ = make_gaussian_binary(n=40, d=4, sep=1.0, seed=0)
    clf = MeanDifferenceClassifier()
    with pytest.raises(Exception):
        _ = clf.decision_function(X)
