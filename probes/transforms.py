import torch.nn as nn
import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import functools
import math


def hellinger_fast(p, q):
    """Hellinger distance between two discrete distributions.
       In pure Python.
       Fastest version.
    """
    return sum([(math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p, q)])


def parallel_df(func, df, series, n_jobs):
    """
    Apply a function to each column of a DataFrame in parallel.
    """
    n_jobs = min(cpu_count(), len(df.columns)) if n_jobs == - \
        1 else min(cpu_count(), n_jobs)
    col_chunks = np.array_split(range(len(df.columns)), n_jobs)
    lst = Parallel(n_jobs=n_jobs)(
        delayed(func)(df.iloc[:, col_chunk], series)
        for col_chunk in col_chunks
    )
    return pd.concat(lst)


def chatterjee_cc(X, Y, ties=False, random_state=42):
    """
    Compute Chatterjee's Concordance Correlation Coefficient.
    """
    np.random.seed(random_state)
    n = len(X)
    order = np.argsort(X)
    ranks = np.argsort(Y[order])
    diff_ranks = np.abs(np.diff(ranks))

    if ties:
        counts = np.bincount(ranks)
        ranks += np.random.uniform(0, counts[ranks] - 1).astype(int)
        l = np.bincount(ranks).astype(float)
        return 1 - (n * np.sum(diff_ranks)) / (2 * np.sum(l * (n - l)))
    else:
        return 1 - (3 * np.sum(diff_ranks)) / (n**2 - 1)


# def compute_distance_covariance(x, y):
#    """
#    Compute the distance covariance between x and y.
#    """
#    return dcor.distance_covariance(x, y)


def _mi_classif(X, y):
    """
    Compute the mutual information between each column of X and y.
    """
    def _mi_classif_series(x, y, n_neighbors: int = 25):
        x_not_na = ~ x.isna()
        if x_not_na.sum() == 0:
            return 0
        return mutual_info_classif(x[x_not_na].to_frame(), y[x_not_na], n_neighbors=n_neighbors)[0]

    return X.apply(lambda col: _mi_classif_series(col, y)).fillna(0.0)

# Figure out how to set the number of neighbours


def mi_classif(X, y, n_jobs):
    """
    Compute the mutual information between each column of X and y in parallel.
    """
    return parallel_df(_mi_classif, X, y, n_jobs=n_jobs)


def random_forest_classif(X, y):
    forest = RandomForestClassifier(n_estimators=500, max_depth=10, criterion="gini", random_state=0, n_jobs=-1).fit(
        X.fillna(X.min().min() - 1), y)
    return pd.Series(forest.feature_importances_, index=X.columns)


def correlation(target_column, features, X, n_jobs, corr_type="spearman"):
    """
    Compute the correlation between each column of X and the target column.
    """
    def _correlation(X, y):
        if corr_type == "spearman":
            return X.corrwith(y, method="spearman").fillna(0.0)
        elif corr_type == "pearson":
            return X.corrwith(y, method="pearson").fillna(0.0)
        elif corr_type == "chatterjee":
            return X.corrwith(y, method=chatterjee_cc).fillna(0.0)
        elif corr_type == "distance_covariance":
            raise NotImplementedError()
            # return X.corrwith(y, method=compute_distance_covariance).fillna(0.0)
        else:
            raise ValueError(f"Unknown correlation type: {corr_type}")
    return parallel_df(_correlation, X.loc[:, features], X.loc[:, target_column], n_jobs=n_jobs)


def normalize(X):
    """
    Normalize the rows of a matrix.
    """
    if X.ndim == 1:
        return X / np.linalg.norm(X)
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


def get_direction(acts, labels):
    pos_acts, neg_acts = acts[labels == 1], acts[labels == 0]
    pos_mean, neg_mean = pos_acts.mean(0), neg_acts.mean(0)
    return (pos_mean - neg_mean)


def sparsify(x, num_features):
    _x = x.copy()
    x_abs = _x
    idx = np.argsort(x_abs)[:-num_features]
    _x[idx] = 0
    return _x


class SparseTransform(BaseEstimator, TransformerMixin):
    def __init__(self, max_k: int = 25, redundancy="spearman",
                 relevance='mi',
                 normalize: bool = True,
                 return_scores: bool = False,
                 show_progress: bool = True) -> None:
        """
        Sparse Transform
        Args:
            k: Number of features to select
            redundancy: Redundancy function
            relevance: Relevance function
            normalize: Normalize the data
        """
        super().__init__()
        self.normalize = normalize

        self.max_k = max_k
        self.selected_features = None
        self.scores = None

        self.return_scores = return_scores
        self.show_progress = show_progress
        self.redundancy = redundancy
        self.relevance = relevance

        if self.normalize:
            self.sc = StandardScaler()

        # Set Redundancy function
        if type(self.redundancy) == str:
            if self.redundancy in ['spearman', 'pearson', 'chatterjee', 'distance_covariance']:
                self.redundancy_fn = functools.partial(
                    correlation, n_jobs=cpu_count(), corr_type=self.redundancy)
            else:
                raise ValueError(
                    f"Unknown correlation type: {self.redundancy}. Choose from ['spearman', 'pearson', 'chatterjee', 'distance_covariance']")
        else:
            self.redundancy_fn = self.redundancy

        # Set Relevance function
        if type(self.relevance) == str:
            if self.relevance == 'mi':
                self.relevance_fn = functools.partial(
                    mi_classif, n_jobs=cpu_count())
            elif self.relevance in ["ks", 'f']:
                self.relevance_fn = self.relevance
            elif self.relevance == 'rf':
                self.relevance_fn = random_forest_classif
            else:
                raise ValueError(
                    f"Unknown relevance type: {self.relevance}. Choose from ['mi', 'ks', 'f', 'rf']")
        else:
            self.relevance_fn = self.relevance

        self.is_fitted = False

    def fit(self, X, y):
        if not self.is_fitted:
            if self.normalize:
                raise NotImplementedError()
                # X = self.sc.fit_transform(X)

            if self.return_scores:  # Return scores
                self.selected_features, self.scores = mrmr_classif(pd.DataFrame(X), pd.Series(y), K=self.max_k,
                                                                   redundancy=self.redundancy_fn,
                                                                   relevance=self.relevance_fn,
                                                                   return_scores=self.return_scores, show_progress=self.show_progress)
            else:
                self.selected_features = mrmr_classif(pd.DataFrame(X), pd.Series(y), K=self.max_k,
                                                      redundancy=self.redundancy_fn,
                                                      relevance=self.relevance_fn,
                                                      show_progress=self.show_progress,
                                                      return_scores=False)
            self.is_fitted = True

        return self

    def transform(self, X, k: int = None):
        check_is_fitted(self, 'is_fitted')
        if k is None:
            k = self.max_k
        else:
            assert k <= self.max_k, "k should be less than or equal to max_k"
        selected_features = self.selected_features[:k]

        if self.normalize:
            X = self.sc.transform(X)
        _X = np.zeros_like(X)
        _X[:, selected_features] = X[:, selected_features].copy()
        return _X

    def fit_transform(self, X, y, k: int = None):
        self.fit(X, y)
        return self.transform(X, k=k)
