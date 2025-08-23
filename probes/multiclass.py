# probes/ova_projector.py
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from misc.probe_data import ProbeData
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class MulticlassProbe(Pipeline):
    '''Multiclass probe for scoring bags against multiple classes.'''
    def __init__(self, projector, pool="max", max_iter=2000):
        self.projector = projector
        self.pool = pool
        self.max_iter = max_iter
        super().__init__(steps=[
            ("bag2scores", OvAScoresSkWrapper(projector, pool=pool)),
            ("scaler", StandardScaler()),
            ("predictor", LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                penalty=None,           
                multi_class="multinomial",
                max_iter=max_iter
            )),
        ])

class OvAProjector:
    '''One-vs-All Projector for scoring bags against multiple classes.'''
    def __init__(self, probe_data_dict: dict[int, "ProbeData"], layer_id: int):
        """
        This objects takes a instances of ProbeData 
        probe_data_by_class: {class_id: ProbeData}
        layer_id: which layer’s params to use
        """
        self.order = sorted(probe_data_dict.keys())
        # cache layer params once
        self.params = {c: probe_data_dict[c].load_layer(layer_id) for c in self.order}

    @staticmethod
    def _sanitize(X, val=1e4):
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=val, neginf=-val)
        return np.clip(X, -val, val)

    @staticmethod
    def _score_instances(X, direction, bias):
        # X: (n,d), direction: (d,), bias: scalar -> (n,)
        return X @ direction + bias

    def score_bag(self, bag: np.ndarray, pool: str = "max") -> np.ndarray:
        """
        bag: (n,d) -> (C,) scores (one per class)
        pool: 'last' | 'max' | 'mean'
        """
        bag = self._sanitize(bag)
        out = np.zeros(len(self.order), dtype=np.float64)

        for j, c in enumerate(self.order):
            lp = self.params[c]
            # If you saved a full model instead of direction/bias, you could branch here.
            Xs = lp.scaler.transform(bag) if lp.scaler is not None else bag
            s = self._score_instances(Xs, lp.direction, lp.bias)  # (n,)

            if pool == "last":
                v = s[-1]
            elif pool == "max":
                v = float(np.max(s))
            elif pool == "mean":
                v = float(np.mean(s))
            else:
                raise ValueError(f"Unknown pool={pool}")
            out[j] = v

        return out  # (C,)

    def score_bags(self, bags: list[np.ndarray], pool: str = "last") -> np.ndarray:
        return np.stack([self.score_bag(b, pool=pool) for b in bags], axis=0)  # (N,C)
    

class OvAScoresSkWrapper(BaseEstimator, TransformerMixin):
    '''Transform One-vs-All scores to a format suitable for scikit-learn.'''
    def __init__(self, projector, pool: str = "max"):
        self.projector = projector
        self.pool = pool

    def fit(self, X, y=None):
        return self  # stateless

    def transform(self, X):
        return self.projector.score_bags(X, pool=self.pool)