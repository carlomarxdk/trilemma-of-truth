from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
import numpy as np
from typing import Union
from scipy.sparse.linalg import eigsh
from scipy.special import expit

from probes.mean_difference import VALID_COVARIANCE_METHODS, robust_covariance


class SupervisedPCA(BaseEstimator, ClassifierMixin):
    '''
    Supervised PCA (PCA + LR probe)
    '''
    def __init__(self,
                 n_components: Union[int, None] = None,
                 cov_type: str = 'oas',
                 cov_fallback_scale: float = 1.0,
                 base_model: ClassifierMixin = LogisticRegression(penalty='none', intercept=True),
                 verbose: bool = False,
                 random_seed: int = 42) -> 'SupervisedPCA':
        '''
        Initialize the SupervisedPCA model.
        Args:
            n_components: Number of components to keep
            cov_type: Type of covariance estimation. One of VALID_COVARIANCE_METHODS
            base_model: Base classifier to use after projection
            verbose: Whether to print progress.
            random_seed: Random seed for reproducibility.
            '''
        assert cov_type in VALID_COVARIANCE_METHODS, f"cov_method must be one of {VALID_COVARIANCE_METHODS}"
        self.random_seed = random_seed
        self.verbose = verbose
        self.n_components = n_components
        self.cov_type = cov_type
        self.cov_fallback_scale = cov_fallback_scale
        self.coef_ = None
        self.vals_ = None
        self.base_model = base_model
        self.is_fitted_ = False

    def fit(self, X: np.array, y: np.array) -> 'SupervisedPCA':
        # assert all(attr in df.columns for attr in attributes), "Attributes must be in the dataframe."
        np.random.seed(self.random_seed)
        X = np.asarray(X, dtype=np.float32)  # Use float32 for speed
        y = np.asarray(y).astype(int)

        X_pos = X[y == 1]
        X_neg = X[y == 0]

        mu_p = X_pos.mean(axis=0, keepdims=True)
        mu_n = X_neg.mean(axis=0, keepdims=True)
        
        Xp = X_pos - mu_p
        Xn = X_neg - mu_n
        
        # covariance matrices
        S_pos = robust_covariance(Xp, method=self.cov_method, fallback_scale=self.cov_fallback_scale)
        S_neg = robust_covariance(Xn, method=self.cov_method, fallback_scale=self.cov_fallback_scale)
        dm = (mu_p - mu_n).ravel()
        M = S_pos + S_neg + np.outer(dm, dm)
        vals, vecs = eigsh(M, k=self.n_components, which="LA")  # top-k only
        
        self.vals_ = vals
        self.coef_ = vecs
        Z = X @ self.coef_
        self.base_model.fit(Z, y)
        self.is_fitted_ = True
        return self

    def decision_function(self, X: np.array) -> np.array:
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X, dtype=np.float32)
        scores = X @ self.coef_
        return self.base_model.decision_function(scores)

    def predict_proba(self, X: np.array) -> np.array:
        scores = self.decision_function(X)
        probs = expit(scores)
        return probs

    def predict(self, X: np.array) -> np.array:
        scores = self.predict_proba(X)
        return  scores.round().astype(int)
    
    
    __all__ = ['SupervisedPCA']