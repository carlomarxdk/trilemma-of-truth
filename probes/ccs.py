# here is the variation of the CCS probe
from scipy.linalg import LinAlgError
from sklearn.covariance import LedoitWolf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
import numpy as np
from typing import Union
import pandas as pd
from scipy.sparse.linalg import eigsh


VALID_COVARIANCE_METHODS = ['empirical', 'ledoit_wolf', 'diagonal', 'shrunk']

def fast_covariance(X, method='empirical', fallback_scale=1.0):
    """
    Fast covariance estimation with multiple methods.

    Args:
        X: Input data
        method: 'empirical', 'ledoit_wolf', 'diagonal', 'shrunk'
        fallback_scale: Scale for fallback identity matrix
    """
    try:
        if method == 'empirical':
            # Fastest - simple sample covariance
            cov = np.cov(X, rowvar=False, bias=False)

        elif method == 'ledoit_wolf':
            # Good compromise between speed and robustness
            cov, _ = LedoitWolf().fit(X).covariance_, None

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

        # Quick NaN check
        if np.isnan(cov).any():
            raise ValueError("NaN detected")

    except (LinAlgError, ValueError, np.linalg.LinAlgError):
        cov = fallback_scale * np.eye(X.shape[1])

    return cov


class EigenCCS(BaseEstimator, ClassifierMixin):
    '''
    Contrast Consistent Search (CCS) via the eig
    '''

    def __init__(self,
                 n_components: Union[int, None] = None,
                 cov_method: str = 'ledoit_wolf',
                 verbose: bool = False,
                 random_seed: int = 42) -> 'EigenCCS':
        '''
        Initialize the EigenCCS model.
        Args:
            n_components: Number of components to keep
            verbose: Whether to print progress.
            random_seed: Random seed for reproducibility.
            '''
        assert cov_method in VALID_COVARIANCE_METHODS, f"cov_method must be one of {VALID_COVARIANCE_METHODS}"
        self.random_seed = random_seed
        self.verbose = verbose
        self.n_components = n_components
        self.cov_method = cov_method
        self.coef_ = None
        self.vals_ = None
        self.model = LogisticRegression()
        self.is_fitted_ = False

    def fit(self, X: np.array, y: np.array) -> 'EigenCCS':
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
        Cp = fast_covariance(Xp, method=self.cov_method, fallback_scale=1.0)
        Cn = fast_covariance(Xn, method=self.cov_method, fallback_scale=1.0)
        dm = (mu_p - mu_n).ravel()
        M = Cp + Cn + np.outer(dm, dm)
        vals, vecs = eigsh(M, k=self.n_components, which="LA")  # top-k only
        
        self.vals_ = vals
        self.coef_ = vecs
        Z = X @ self.coef_
        self.model.fit(Z, y)
        self.is_fitted_ = True
        return self

    # def find_pairs(self, df: pd.DataFrame, groupby: str = 'object_1', contrast_col: str = 'correct'):
    #     df = df.copy()
    #     df = df.reset_index(drop=False)  # need to keep ids to assemble pairs
    #     pairs = []
    #     for _, group in df.groupby(groupby):
    #         pos_rows = group[group[contrast_col] == True]
    #         neg_rows = group[group[contrast_col] == False]

    #         if pos_rows.empty or neg_rows.empty:
    #             continue

    #         for _, pos_row in pos_rows.iterrows():
    #             for _, neg_row in neg_rows.iterrows():
    #                 pairs.append((pos_row['index'], neg_row['index']))

    #     pairs = list(set(pairs))
    #     return pairs
    def decision_function(self, X: np.array, sum: bool = False) -> np.array:
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X, dtype=np.float32)
        scores = X @ self.coef_
        if sum:
            scores = scores.sum(axis=1)
        return scores
