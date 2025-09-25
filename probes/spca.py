from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
import numpy as np
from typing import Union
from scipy.sparse.linalg import eigsh
# from scipy.special import expit

from probes.mean_difference import VALID_COVARIANCE_METHODS, robust_covariance


class SupervisedPCA(ClassifierMixin, BaseEstimator):
    '''
    Supervised PCA (PCA + LR probe)
    '''
    def __init__(self,
                 n_components: Union[int, None] = None,
                 cov_type: str = 'oas',
                 cov_fallback_scale: float = 1.0,
                 cov_reg: float = 1e-8,
                 base_model: ClassifierMixin = LogisticRegression(penalty=None, fit_intercept=True, max_iter=2000),
                 verbose: bool = False,
                 random_seed: int = 42) -> 'SupervisedPCA':
        '''
        Initialize the SupervisedPCA model.
        Args:
            n_components: Number of components to keep
            cov_type: Type of covariance estimation. One of VALID_COVARIANCE_METHODS
            cov_fallback_scale: Scale for the identity fallback if covariance estimation fails
            cov_reg: Regularization added to the diagonal of the covariance matrix for numerical stability
            base_model: Base classifier to use after projection
            verbose: Whether to print progress.
            random_seed: Random seed for reproducibility.
            '''
        assert cov_type in VALID_COVARIANCE_METHODS, f"cov_type must be one of {VALID_COVARIANCE_METHODS}"
        self.random_seed = random_seed
        self.verbose = verbose
        self.n_components = n_components
        self.cov_type = cov_type
        self.cov_reg = cov_reg
        self.cov_fallback_scale = cov_fallback_scale
        self.coef_ = None
        self.vals_ = None
        self.base_model = base_model
        self.is_fitted_ = False
        self.classes_ = np.array([0, 1])

    def fit(self, X: np.array, y: np.array) -> 'SupervisedPCA':
        np.random.seed(self.random_seed)
        X = np.asarray(X, dtype=np.float32)  # Use float32 for speed
        y = np.asarray(y).astype(int)

        X_pos = X[y == 1]
        X_neg = X[y == 0]

        mu_p = X_pos.mean(axis=0, keepdims=True)
        mu_n = X_neg.mean(axis=0, keepdims=True)
        mu = X.mean(axis=0, keepdims=True)
        n_p = X_pos.shape[0]
        n_n = X_neg.shape[0]
        n = n_p + n_n
        d = X.shape[1]
        
        Xp = X_pos - mu_p
        Xn = X_neg - mu_n
        
        # covariance matrices
        S_pos = robust_covariance(Xp, method=self.cov_type, fallback_scale=self.cov_fallback_scale)
        S_neg = robust_covariance(Xn, method=self.cov_type, fallback_scale=self.cov_fallback_scale)
        # within-class covariance
        S_w = (n_p * S_pos + n_n * S_neg) / max(n, 1)

        # between-class covariance
        dp = (mu_p - mu).reshape(-1, 1)  # (d,1)
        dn = (mu_n - mu).reshape(-1, 1)  # (d,1)
        S_b = (n_p * (dp @ dp.T) + n_n * (dn @ dn.T)) / max(n, 1)
        M = S_b + S_w + self.cov_reg * np.eye(d)
        M = 0.5 * (M + M.T)  # enforce symmetry numerically
        if (M.shape[0] - 1) < self.n_components: # only for 'sparse' problems
            raise ValueError(f"n_components={self.n_components} is too large for data with {M.shape[0]} features.")
        vals, vecs = eigsh(M, k=self.n_components, which="LA")  # top-k only
        # sort in descending order
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        
        self.components_ = vecs
        self.explained_variance_ = vals
        Z = X @ self.components_
        self.base_model.fit(Z, y)
        self.is_fitted_ = True
        return self

    def decision_function(self, X: np.array) -> np.array:
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X, dtype=np.float32)
        scores = X @ self.components_
        return self.base_model.decision_function(scores)

    def predict_proba(self, X: np.array) -> np.array:
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X, dtype=np.float32)
        probs =  self.base_model.predict_proba(X @ self.components_)
        if probs.shape[1] == 2:
            return probs[:, 1].ravel() 
        return probs.ravel()
    def predict(self, X: np.array) -> np.array:
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X, dtype=np.float32)
        return self.base_model.predict(X @ self.components_)
        # scores = self.predict_proba(X)
        # return  scores.round().astype(int)
    
    
    __all__ = ['SupervisedPCA']
