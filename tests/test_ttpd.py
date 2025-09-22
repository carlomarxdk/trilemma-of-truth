import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# ---- import the class & helpers under test ----
# from your_module import TTPD, is_centered
from probes.ttpd import TTPD, is_centered  # adjust to the real module path


# ------------------------- helpers -------------------------

def make_synthetic(n=400, d=16, rng=0):
    """
    Build a dataset where the targets are generated from a ground-truth
    'truth direction' and 'polarity direction' so the probe has signal
    to recover. We also create a binary polarity label p in {0,1}.
    """
    rs = np.random.RandomState(rng)
    X = rs.randn(n, d).astype(np.float64)

    # ground-truth directions
    w_t_true = rs.randn(d); w_t_true /= np.linalg.norm(w_t_true)
    w_p_true = rs.randn(d); w_p_true /= np.linalg.norm(w_p_true)

    # polarity label: 50/50 split
    p = (rs.rand(n) > 0.5).astype(int)

    # truth label: logistic in (X@w_t + 0.6 * (2*p-1) * X@w_p)
    # so polarity interacts with truth a bit
    logits = (X @ w_t_true) + 0.6 * ((2 * p - 1) * (X @ w_p_true))
    y_prob = 1 / (1 + np.exp(-logits))
    y = (y_prob > 0.5).astype(int)

    return X, y, p, w_t_true, w_p_true


# ------------------------- unit tests -------------------------

def test_is_centered_true_false():
    X = np.random.randn(100, 5)
    assert is_centered(X) is False
    Xc = X - X.mean(axis=0, keepdims=True)
    assert is_centered(Xc) is True

def test_labels_to_sign_mapping():
    y = np.array([0, 1, 1, 0, 0, 1])
    # access the protected method via the class (style only; OK in tests)
    y_sign = TTPD._labels_to_sign(y)
    assert y_sign.dtype.kind == 'f'
    assert np.all(y_sign[y == 1] == 1.0)
    assert np.all(y_sign[y == 0] == -1.0)

def test_get_truth_direction_ols_shape_and_signal():
    # Build X from a known linear model of the form Xc = A W + noise,
    # where A = [t, t*p] and W rows are the true directions.
    rs = np.random.RandomState(123)
    n, d = 300, 12
    t = rs.choice([-1.0, 1.0], size=n)
    p = (rs.rand(n) > 0.4).astype(float)  # in {0,1} with both classes present

    # build A
    A = np.column_stack([t, t * p])  # (n,2)
    # true directions:
    w_t_true = rs.randn(d); w_t_true /= np.linalg.norm(w_t_true)
    w_p_true = rs.randn(d); w_p_true /= np.linalg.norm(w_p_true)
    W_true = np.vstack([w_t_true, w_p_true])  # (2, d)

    # generate centered X from A W_true + small noise
    noise = 0.05 * rs.randn(n, d)
    Xc = A @ W_true + noise
    X = Xc + rs.randn(1, d)  # add a random mean so code must center

    # call private OLS
    w_t_hat, w_p_hat = TTPD._get_truth_direction(X, t, p)

    # check shapes and angle similarity
    assert w_t_hat.shape == (d,)
    assert w_p_hat is not None and w_p_hat.shape == (d,)

    # cosine similarity should be high
    def cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    assert cos(w_t_hat, w_t_true) > 0.9
    assert cos(w_p_hat, w_p_true) > 0.8

def test_get_truth_direction_zero_polarity_returns_single_direction():
    rs = np.random.RandomState(0)
    n, d = 200, 10
    t = rs.choice([-1.0, 1.0], size=n)
    p = np.zeros(n)  # all zero polarity

    # true single direction
    w_t_true = rs.randn(d); w_t_true /= np.linalg.norm(w_t_true)

    # Xc = (t) * w_t^T + noise
    A = t.reshape(-1, 1)
    Xc = A @ w_t_true.reshape(1, -1) + 0.05 * rs.randn(n, d)
    X = Xc + rs.randn(1, d)  # add non-zero mean to test centering

    w_t_hat, w_p_hat = TTPD._get_truth_direction(X, t, p)
    assert w_p_hat is None
    # cosine should be strong
    cos = np.dot(w_t_hat, w_t_true) / (np.linalg.norm(w_t_hat) * np.linalg.norm(w_t_true))
    assert cos > 0.9

def test_get_polarity_direction_requires_two_classes():
    rs = np.random.RandomState(7)
    X = rs.randn(100, 8)
    p_all_zero = np.zeros(100, dtype=int)
    # Should raise from sklearn LR because single class
    with pytest.raises(Exception):
        _ = TTPD._get_polarity_direction(X, p_all_zero)

def test_get_polarity_direction_returns_vector_when_valid():
    rs = np.random.RandomState(8)
    X = rs.randn(200, 6)
    p = (rs.rand(200) > 0.5).astype(int)
    w_p = TTPD._get_polarity_direction(X, p)
    assert w_p.shape == (X.shape[1],)
    assert np.isfinite(w_p).all()

def test_fit_predict_end_to_end_high_auc():
    X, y, p, *_ = make_synthetic(n=600, d=18, rng=1234)
    model = TTPD(base=LogisticRegression(penalty=None, fit_intercept=True, max_iter=2000), random_seed=123)
    model.fit(X, y, p)
    prob = model.predict_proba(X)
    assert prob.shape == (X.shape[0],)
    auc = roc_auc_score(y, prob)
    assert auc > 0.83

def test_decision_function_and_predict_shapes_and_types():
    X, y, p, *_ = make_synthetic(n=200, d=10, rng=3)
    model = TTPD()
    model.fit(X, y, p)
    df = model.decision_function(X)
    proba = model.predict_proba(X)
    pred = model.predict(X)
    assert df.shape == (X.shape[0],)
    assert proba.shape == (X.shape[0],)
    assert pred.shape == (X.shape[0],)
    assert set(np.unique(pred)).issubset({0, 1})

def test_dtype_upcast_accepts_float16_inputs():
    # ensure internals don’t choke on float16 (lstsq doesn’t support f16)
    X, y, p, *_ = make_synthetic(n=300, d=12, rng=11)
    X16 = X.astype(np.float16)
    model = TTPD()
    model.fit(X16, y, p)  # should not raise
    _ = model.predict(X16)

def test_fit_rejects_non_binary_labels_or_mismatched_lengths():
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, size=50)
    p = np.random.randint(0, 2, size=50)

    # mismatched lengths
    with pytest.raises(AssertionError):
        TTPD().fit(X[:40], y, p)

    # non-binary y
    y_bad = np.arange(50)
    with pytest.raises(AssertionError):
        TTPD().fit(X, y_bad, p)
