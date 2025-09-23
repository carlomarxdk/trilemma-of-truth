import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import ledoit_wolf, oas
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve, LinAlgError

import logging
log = logging.getLogger(__name__)

VALID_COVARIANCE_METHODS = ['oas', 'ledoit', 'empirical', 'shrunk', 'diagonal']

def normalize(X, tol: float = 1e-9) -> np.ndarray:
    """
    Normalize the rows of a matrix.
    Args:
        X: (d,) or (N, d) array
        tol: small constant to avoid division by zero
    """
    if X.ndim == 1:
        return X / (np.linalg.norm(X) + tol)
    return X / (np.linalg.norm(X, axis=1)[:, np.newaxis] + tol)


def robust_covariance(X: np.ndarray, method: str = "oas", fallback_scale: float = 1.0) -> np.ndarray:
    """
    Compute a robust covariance estimate with shrinkage, safe against ill-conditioning.

    Args:
        X (np.ndarray): Data matrix of shape (n_samples, n_features).
                        Assumed centered if method='oas' with assume_centered=True.
        method (str): Which shrinkage estimator to use: {"oas", "ledoit", "empirical", "shrunk", "diagonal"}.
        fallback_scale (float): Scale for the identity fallback if estimation fails.

    Returns:
        np.ndarray: Covariance matrix of shape (n_features, n_features).
    """
    assert method in VALID_COVARIANCE_METHODS, f"Unknown method: {method}, must be one of {VALID_COVARIANCE_METHODS}"
    X = np.asarray(X)
    _, d = X.shape

    try:
        if method == "oas":
            cov, _ = oas(X, assume_centered=True)
        elif method == "ledoit":
            cov, _ = ledoit_wolf(X, assume_centered=True)
        elif method == "empirical":
            cov = np.cov(X, rowvar=False)  
        elif method == 'diagonal':
            # Very fast - diagonal covariance only
            cov = np.diag(np.var(X, axis=0))
        elif method == 'shrunk':
            # Fast shrinkage estimator
            emp_cov = np.cov(X, rowvar=False, bias=False)
            trace = np.trace(emp_cov)
            shrinkage = 0.1  # Fixed shrinkage
            cov = (1 - shrinkage) * emp_cov + shrinkage * \
                (trace / X.shape[1]) * np.eye(X.shape[1])
        else:
            raise ValueError(f"Unknown method: {method}")

        # Quick NaN check
        if not np.isfinite(X).all():
            log.warning("Non-finite data detected; using scaled identity.")
            return fallback_scale * np.eye(X.shape[1])
        if np.isnan(cov).any():
            raise ValueError("Non-finite entries detected in covariance.")

    except (LinAlgError, ValueError, np.linalg.LinAlgError) as e:
        cov = fallback_scale * np.eye(X.shape[1])       
        log.warning(
            f"Covariance estimation failed ({e}); using scaled identity.")
        cov = fallback_scale * np.eye(d)

    # Regularize tiny negatives due to numerical error
    cov = 0.5 * (cov + cov.T)  # enforce symmetry
    return cov


class MeanDifferenceClassifier(ClassifierMixin, BaseEstimator):
    """
    Binary mean-difference classifier with optional pooled-covariance weighting (aka Mahalanobis whitening).
    The code is adapted from https://github.com/saprmarks/geometry-of-truth/
    """

    def __init__(self,
                 fit_intercept: bool = True,
                 with_covariance: bool = False,
                 cov_type: str = 'oas',
                 cov_reg: float = 1e-8,
                 tol: float = 1e-8,
                 verbose: bool = False) -> 'MeanDifferenceClassifier':
        '''
        Args:
            fit_intercept: If True, the decision boundary is at the midpoint between the projected class means.
            with_covariance: If True, the inverse pooled-covariance matrix is used to compute the score (aka Mahalanobis whitening). Turns it into the Fisher discriminant (LDA).
            cov_reg: Ridge regularization added to the covariance matrix when with_covariance=True (to ensure invertibility).
            tol: Tolerance for numerical stability when normalizing the weight vector.
            verbose: If True, prints additional information during fitting and scoring.
        '''
        super().__init__()
        assert cov_type in VALID_COVARIANCE_METHODS, f"cov_type must be one of {VALID_COVARIANCE_METHODS}"
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.with_covariance = with_covariance
        self.cov_reg = cov_reg
        self.cov_type = cov_type
        self.tol = tol
        self.M_ = None

        # Initialize the parameters
        self.intercept_ = None
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray, M: np.ndarray = None) -> 'MeanDifferenceClassifier':
        """
        Fit the model to the data.
        Args:
            X: (N, d) array of input data
            y: (N,) array of binary labels (0 or 1)
            M: Mahalanobis matrix (optional), used if with_covariance=True.
        Returns:
            self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        assert type_of_target(y) == "binary", "Labels should be binary."
        if M is not None:
            assert self.with_covariance, "If providing M, must have with_covariance=True"
            assert M.shape[0] == M.shape[1] == X.shape[1], "M must be square and match feature dimension."

        pos_acts, neg_acts = X[y == 1], X[y == 0]
        mu_pos, mu_neg = pos_acts.mean(0), neg_acts.mean(0)
        delta = mu_pos - mu_neg

        if self.with_covariance:
            if M is not None:
                # supplied mahalanobis matrix
                # self.M_ = M
                w = M @ delta
            else:
                S_pos = robust_covariance(pos_acts - mu_pos[None, :], method=self.cov_type, fallback_scale=1.0)
                S_neg = robust_covariance(neg_acts - mu_neg[None, :], method=self.cov_type, fallback_scale=1.0)
                n_pos, n_neg = pos_acts.shape[0], neg_acts.shape[0]
                # pooled covariance
                Sp = ((n_pos - 1) * S_pos + (n_neg - 1) * S_neg) / \
                    max(1, (n_pos + n_neg - 2))
                Sp = Sp + self.cov_reg * np.eye(Sp.shape[0], dtype=Sp.dtype)
                # solve pooled @ w = delta (does not require explicit inversion)
                c, lower = cho_factor(
                    Sp, overwrite_a=False, check_finite=False)
                w = cho_solve((c, lower), delta, check_finite=False)
        else:
            w = delta
        w = normalize(w, tol=self.tol)
        self.coef_ = w.reshape(1, -1)

        if self.fit_intercept:
            # Compute the intercept as the difference in the means of the projected features
            b_pos = (pos_acts @ self.coef_.T).mean()
            b_neg = (neg_acts @ self.coef_.T).mean()
            self.intercept_ = float(-0.5 * (b_pos + b_neg))
        else:
            self.intercept_ = 0.0

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class of each sample in X.
        """
        return self.predict_proba(X).round()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of each sample in X.
        """
        return expit(self.decision_function(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the decision function for each sample in X.
        Args:
            X: (N, d) array of input data
        Returns:
            scores: (N,) array of decision scores
        """
        check_is_fitted(self)
        X = np.asarray(X)
        return (X @ self.coef_.T).ravel() + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray, scorer, sample_weight=None) -> float:  # type: ignore
        """
        Compute the accuracy of the model.
        """
        assert type_of_target(y) == "binary", "Labels should be binary."
        try:
            return scorer(y, self.predict_proba(X))
        except:
            if self.verbose:
                print("Using discreet label instead of predict_proba.")
            return scorer(y, self.predict(X))
