import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lars
from scipy.stats import wasserstein_distance, energy_distance


def hellinger_fast(p, q):
    """Hellinger distance between two discrete distributions.
       In pure Python.
       Fastest version.
    """
    return sum([(math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p, q)])


class DistanceBasedFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 25, distance_metric="hellinger", base_model=Lars(),
                 normalize: bool = True) -> None:
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

        self.k = k
        self.selected_features = None
        self.scores = None
        self.base_model = base_model

        if distance_metric == "hellinger":
            self.distance_fn = hellinger_fast
        elif distance_metric == "wasserstein":
            self.distance_fn = wasserstein_distance
        elif distance_metric == "energy":
            self.distance_fn = energy_distance
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        if self.normalize:
            self.sc = StandardScaler()

    def fit(self, X, y):
        if self.normalize:
            X = self.sc.fit_transform(X)

        self.is_fitted_ = True

        return self

    def transform(self, X):
        if self.normalize:
            X = self.sc.transform(X)
        _X = np.zeros_like(X)
        _X[:, self.selected_features] = X[:, self.selected_features].copy()
        return _X

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _fit(self, X, y):
