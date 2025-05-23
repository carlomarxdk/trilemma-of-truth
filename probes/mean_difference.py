import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from scipy.special import expit

from sklearn.covariance import oas
import logging
log = logging.getLogger(__name__)


def normalize(X):
    """
    Normalize the rows of a matrix.
    """
    if X.ndim == 1:
        return X / np.linalg.norm(X)
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


def robust_covariance(X, fallback_scale=1.0):
    """
    Computes a robust covariance estimate using ledoit_wolf.
    If an error occurs or NaNs are detected, returns a fallback covariance.

    Args:
        X (np.ndarray): Input data.
        fallback_scale (float): Scale factor for the identity matrix fallback.

    Returns:
        np.ndarray: Covariance matrix.
    """
    try:
        cov, _ = oas(X, assume_centered=True)
        # Check for NaN values in the computed covariance matrix
        if np.isnan(cov).any():
            raise ValueError("NaN values detected in covariance estimate.")
    except Exception as e:
        log.warning(
            f"Error computing covariance: {e}. Using fallback covariance.")
        cov = fallback_scale * np.eye(X.shape[1])
    return cov


class MeanDifferenceClassifier(BaseEstimator, ClassifierMixin):
    # The code is adapted from https://github.com/saprmarks/geometry-of-truth/

    def __init__(self,
                 fit_intercept: bool = True,
                 with_covariance: bool = False,
                 cov_reg: float = 1e-6,
                 verbose: bool = False) -> None:
        super().__init__()
        # If True, the covariance matrix is used to compute the score
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        # Initialize the parameters
        self.intercept_ = None
        self.coef_ = None
        self.with_covariance = with_covariance
        self.cov_reg = cov_reg
        self.M_ = None

    def fit(self, X, y, M=None):
        """
        Fit the model to the data.
        Args:
            M: Mahalanobis matrix
        """
        assert type_of_target(y) == "binary", "Labels should be binary."

        pos_acts, neg_acts = X[y == 1], X[y == 0]
        pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)

        if self.with_covariance and (M is None):
            # Compute the Mahalanobis inverse covariance matrix (pooled)
            cov_pos = robust_covariance(pos_acts)
            if not np.array_equal(cov_pos, np.eye(cov_pos.shape[0])):
                cov_neg = robust_covariance(neg_acts)
                if np.array_equal(cov_neg, np.eye(cov_neg.shape[0])):
                    cov_neg = cov_pos
            else:
                cov_neg = cov_pos
            n_pos = pos_acts.shape[0]
            n_neg = neg_acts.shape[0]
            pooled_cov = ((n_pos - 1) * cov_pos + (n_neg - 1)
                          * cov_neg) / (n_pos + n_neg - 2)
            # Check for NaNs in the pooled covariance and substitute if needed.
            if np.isnan(pooled_cov).any():
                log.warning(
                    "NaN detected in pooled covariance; substituting with identity matrix.")
                pooled_cov = np.eye(pos_acts.shape[1])

            pooled_cov += self.cov_reg * np.eye(pooled_cov.shape[0])
            self.M_ = np.linalg.inv(pooled_cov)
        elif self.with_covariance and (M is not None):
            self.M_ = M

        self.coef_ = pos_mean - neg_mean
        if type(self.coef_) != np.ndarray:
            self.coef_ = self.coef_.cpu().numpy()
        if self.with_covariance:
            self.coef_ = self.M_ @ self.coef_
        self.coef_ = normalize(self.coef_).reshape(1, -1)

        if self.fit_intercept:
            # Compute the intercept as the difference in the means of the projected features
            pos_proj_mean = pos_acts @ self.coef_.T
            neg_proj_mean = neg_acts @ self.coef_.T
            self.intercept_ = -0.5 * \
                (pos_proj_mean.mean() + neg_proj_mean.mean())

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict the class of each sample in X.
        """
        return self.predict_proba(X).round()

    def predict_proba(self, X):
        """
        Predict the probability of each sample in X.
        """
        return expit(self.decision_function(X))

    def decision_function(self, X):
        """
        Compute the decision function for each sample in X.
        """
        check_is_fitted(self)
        score = X @ self.coef_.T

        if self.intercept_ is not None:
            return score + self.intercept_

        return score

    def score(self, X, y, scorer, sample_weight=None):  # type: ignore
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
