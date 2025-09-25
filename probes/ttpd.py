import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import type_of_target


def is_centered(X: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Return True if each column of X has (near) zero mean.
    """
    return np.allclose(X.mean(axis=0), 0.0, atol=tol)


class TTPD(ClassifierMixin, BaseEstimator):
    '''
    Training of Truth and Polarity Direction (TTPD) probe introduced in:
        Bürger, Lennart, Fred A. Hamprecht, and Boaz Nadler. 
        'Truth is universal: Robust detection of lies in llms, 2024.' 
        URL https://arxiv. org/abs/2407.12831.
    The following code is adapted from the original implementation at:
        https://github.com/sciai-lab/Truth_is_Universal
    '''

    def __init__(self,
                 base: ClassifierMixin = LogisticRegression(
                     penalty=None, fit_intercept=True),
                 random_seed: int = 42,
                 verbose: bool = False) -> 'TTPD':
        '''
        Args:
            base: Base classifier to use after projection
            random_seed: Random seed for reproducibility.
            verbose: Whether to print progress.
        '''
        self.verbose = verbose
        self.base = base
        self.random_seed = random_seed
        self.is_fitted_ = False
        self.w_t = None # truth direction from OLS
        self.w_p = None # polarity direction from LR
        self.w_p_ = None  # polarity direction from OLS (not used downstream)
        self.mu_ = None  # global centering
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _labels_to_sign(y: np.ndarray) -> np.ndarray:
        # y assumed binary {0,1} -> {-1,+1}
        y = np.asarray(y).copy().astype(int).ravel()
        return np.where(y == 1, 1.0, -1.0)

    @staticmethod
    def _get_truth_direction(X: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        '''
        Get the truth direction using the Ordinary Least Squares (OLS) solution.
        Args:
            X: (N, d) array of input data
            t: (N,) array of truthfulness labels (-1 or +1)
            p: (N,) array of polarity labels (0 or 1)
        Returns:
            w_t: (d,) array, truth direction
            w_p: (d,) array, polarity direction (not used downstream)
        '''
        assert np.unique(t).tolist() == [-1, 1], "t must be in {-1, +1}"
        assert np.unique(p).tolist() == [0, 1] or np.unique(
            p).tolist() == [0], "p must be in {0, 1} or {0}"
        X = np.asarray(X, dtype=np.float64)     
        t_copy = np.asarray(t, dtype=np.float64).copy().ravel()
        p_copy = np.asarray(p, dtype=np.float64).copy().ravel()
        Xc = X if is_centered(X) else (X - X.mean(axis=0, keepdims=True))

        # design matrix (N,k)
        if np.all(p == 0):
            A = t.reshape(-1, 1)                            # (N,1)
        else:
            A = np.column_stack([t_copy, t_copy * p_copy]
                                )                 # (N,2)

        W, *_ = np.linalg.lstsq(A, Xc, rcond=None) # with centered X

        w_t = W[0, :].ravel()
        w_p = W[1, :].ravel() if W.shape[0] > 1 else None
        w_t = np.asarray(w_t, dtype=float)
        if w_p is not None:
            w_p = np.asarray(w_p, dtype=float).ravel()
        return w_t, w_p
    
    @staticmethod
    def _get_polarity_direction(X: np.ndarray, p: np.ndarray, seed: int=42) -> np.ndarray:
        '''
        Get the polarity directions
        Args:
            X: (N, d) array of input data
            p: (N,) array of polarity labels (0 or 1)
        Returns:
            w_p: (d,) array, polarity direction
        '''
        assert np.unique(p).tolist() == [0, 1] or np.unique(
            p).tolist() == [0], "p must be in {0, 1} or {0}"
        assert X.shape[0] == p.shape[0], "Mismatched number of samples."
        if np.all(p == 0):
            return np.zeros(X.shape[1], dtype=float)

        lr = LogisticRegression(penalty=None, fit_intercept=True)
        lr.fit(X, p.ravel().astype(int))
        return lr.coef_.ravel()

    def _project(self, X: np.ndarray) -> np.ndarray:
        '''
        Project data onto the truth and polarity directions.
        Args:
            X: (N, d) array of input data
        Returns:
            Z: (N, 2) array of projected data
        '''
        X = np.asarray(X, dtype=float)
        z_t = X @ self.w_t
        z_p = X @ self.w_p
        return np.column_stack([z_t, z_p])

    def fit(self, X: np.array, t_labels: np.array, p_labels: np.array) -> 'TTPD':
        '''
        Fit the TTPD probe.
        Args:
            X: (N, d) array of input data
            t_labels: (N,) array of truthfulness labels (0 or 1)
            p_labels: (N,) array of polarity labels (0 or 1)
        '''
        assert X.shape[0] == t_labels.shape[0] == p_labels.shape[0], "Mismatched number of samples."
        assert type_of_target(t_labels) == "binary", "Labels should be binary."
        assert type_of_target(p_labels) == "binary", "Labels should be binary."
        np.random.seed(self.random_seed)
        t_labels = np.asarray(t_labels).ravel().copy()
        p_labels = np.asarray(p_labels).ravel().copy()

        self.w_t, self.w_p_ = self._get_truth_direction(X, self._labels_to_sign(t_labels), p_labels)
        # self.w_p_ is not used downstream
        self.w_p = self._get_polarity_direction(X, p_labels, seed=self.random_seed)
        
        Xp = self._project(X)
        self.base.fit(Xp, self._labels_to_sign(t_labels))
        
        self.is_fitted_ = True
        return self
        
    def decision_function(self, X: np.array) -> np.array:
        '''
        Compute the decision function of the TTPD probe.
        Args:
            X: (N, d) array of input data
        Returns:
            scores: (N,) array of decision scores
        '''
        check_is_fitted(self, 'is_fitted_')
        Xp = self._project(X)
        return self.base.decision_function(Xp)
    
    def predict_proba(self, X: np.array) -> np.array:
        '''
        Predict the probability of the positive class.
        Args:
            X: (N, d) array of input data
        Returns:
            proba: (N,) array of probabilities
        '''
        check_is_fitted(self, 'is_fitted_')
        Xp = self._project(X)
        return self.base.predict_proba(Xp)[:, 1]
    
    def predict(self, X: np.array) -> np.array:
        '''
        Predict the class labels.
        Args:
            X: (N, d) array of input data
        Returns:
            labels: (N,) array of predicted labels (0 or 1)
        '''
        check_is_fitted(self, 'is_fitted_')
        Xp = self._project(X)
        return self.base.predict(Xp)