from __future__ import annotations
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from scipy.special import expit
import numpy as np
import logging
log = logging.getLogger(__name__)

class BinaryLinearProbe(BaseEstimator, ClassifierMixin):
    """
    A binary linear probe with pre-trained coefficients and intercept.
    This probe does not require fitting; it uses provided coefficients and intercept
    to make predictions.
    """
    def __init__(self, coef: np.ndarray, intercept: np.ndarray) -> BinaryLinearProbe:
        self.coef = np.asarray(coef)
        self.intercept = np.asarray(intercept).ravel()
        self.coef_ = self.coef
        self.intercept_ = self.intercept
        self.classes_ = [0, 1]
        self.is_fitted_ = True

    def fit(self, X: np.ndarray = None, y: np.ndarray=None) -> BinaryLinearProbe:
        log.warning("This probe is pre-trained and does not require fitting.")
        return self
    
    def decision_function(self, X:np.ndarray) -> np.ndarray:
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
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability of each sample in X.
        """
        return expit(self.decision_function(X))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class of each sample in X.
        """
        return self.predict_proba(X).round()
    
