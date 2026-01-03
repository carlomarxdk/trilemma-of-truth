from __future__ import annotations

import json
import logging
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from probes.conformal import InductiveConformalPredictor, symmetric_nonconformity
from probes.ttpd import TTPD
from runners.base import BaseProbeRunner

log = logging.getLogger("SILRunner-TTPD")


class TTPD_Runner(BaseProbeRunner):
    """
    SIL Probe Runner for probes trained on the last instance of the bag (for Training Truth and Polarity Direction probe)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        # set random seed
        np.random.seed(getattr(cfg.probe, "seed", None))
        self.separator = None
        self.scaler = StandardScaler()
        self.calibrator = None
        self.transformer = None

    def return_target(self, y: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        In case of TTPD, just apply mask.
        Args:
            y: bag_labels or labels
            mask: mask for the task
        Returns:
            yy: masked bag_labels or labels
        """
        yy = deepcopy(y)
        if mask is not None:
            return yy[mask]
        return yy

    def single_training(
        self,
        X: list[np.ndarray],
        y: np.ndarray,
        mask: np.ndarray,
        neg: np.ndarray = None,
    ):
        """
        Train transformer and separator on the masked subset of bags.
        Returns dict with 'separator' and fitted 'transformer'.
        Args:
            X: an list of bags
            y: bag_labels
            mask: boolean mask array of length len(X) indicating which bags to train on
            neg: affirmative vs negated labels (if statement is negated = 1)
        Returns a dict with keys:
            'separator', 'scaler', 'transformer'
        """
        # 0) Get the bags and labels
        f_neg = deepcopy(neg) if neg is not None else None
        f_y = deepcopy(y)
        f_X = deepcopy(X)
        f_mask = np.array(mask, dtype=bool)

        assert (
            len(f_X) == len(f_y) == len(f_mask) == len(f_neg)
        ), "X, y and mask must have the same length"
        assert np.unique(f_y).size == 2, "y must be binary"

        ym = self.return_target(f_y, f_mask)
        negm = self.return_target(1 - f_neg, f_mask) if f_neg is not None else None
        # 1) Fit transformer on concatenated instances
        if self.cfg.probe.get("normalize_data", True):
            log.warning("\t\tNormalizing the data...")
            Xm = np.vstack([bag[-1] for bag, m in zip(f_X, f_mask) if m])
            self.scaler.fit(Xm)
        else:
            raise NotImplementedError(
                "Only a pipeline with the normalization is implemented"
            )

        Xm = np.vstack(
            [self.scaler.transform(bag)[-1] for bag, m in zip(f_X, f_mask) if m]
        )

        # 2) Transform each bag (take only the last element)
        cfg = self.cfg.probe
        limit = cfg.get("train_sample_limit", Xm.shape[0])
        log.warning("\t\tFit the data...")

        # 3) Fit the model
        self.separator = TTPD(
            verbose=cfg.init_params.get("verbose", False),
            random_seed=cfg.init_params.get("random_seed", 42),
        )

        self.separator.fit(Xm[:limit], t_labels=ym[:limit], p_labels=negm[:limit])

        return {
            "separator": self.separator,
            "scaler": self.scaler,
            "transformer": np.nan,
        }

    def parameter_search(
        self,
        X: list[np.ndarray],
        y: np.ndarray,
        mask: np.ndarray,
        neg: np.ndarray = None,
    ):
        """
        Training with hyperparameter search
        Args:
            X: an array of bags (Sequences, Lenghts, Hidden Size)
            y: labels
            mask: mask for the data
        """
        log.warning(
            "Running the hyperparameter search... (For SPCA Probe parameter_search == sigle_training)"
        )
        return self.single_training(X, y, mask, neg)

    def conformal_training(self, X_cal, y_cal, mask_cal):
        """
        Train the conformal predictor on the calibration set.
        Args:
            X_cal: array-like, shape (n_samples, n_features)
                The calibration set features.
            y_cal: array-like, shape (n_samples,)
                The calibration set true labels.
            mask_cal: array-like, shape (n_samples,)
                The mask for the calibration set.
        """
        f_X = deepcopy(X_cal)
        f_y = deepcopy(y_cal)
        f_mask = np.array(mask_cal, dtype=bool).copy()
        cfg = self.cfg.conformal_params

        if cfg["nc"] == "binary":
            nc = symmetric_nonconformity
        else:
            raise NotImplementedError(
                f"Nonconformity function {cfg['nc']} is not implemented."
            )
        self.calibrator = InductiveConformalPredictor(
            nonconformity_func=nc, alpha=cfg["alpha"], tie_breaking=cfg["tie_breaking"]
        )
        yh_cal = self.decision_function(f_X)
        self.calibrator.fit(y=f_y[f_mask], scores=yh_cal[f_mask])
        return self.calibrator

    def conformal_prediction(self, X: list[np.ndarray]) -> np.ndarray:
        """
        Compute the conformal prediction for the given bags.
        """
        # Transform the bags using the fitted scaler
        f_X = deepcopy(X)
        # Compute the decision function using the separator
        yh = self.decision_function(f_X)
        # Compute the conformal prediction
        return self.calibrator.predict(yh)

    def decision_function(self, X):
        """
        Compute the decision function for the given bags.
        """
        # Transform the bags using the fitted scaler
        Xt = self.process_input(X)
        yhat = self.separator.decision_function(Xt)
        return yhat.flatten()

    def predict_proba(self, X):
        Xt = self.process_input(X)
        proba = self.separator.predict_proba(Xt)
        if proba.ndim > 1 and proba.shape[1] == 2:
            proba = proba[:, 1]
        return proba.flatten()

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array(proba > 0.5)

    def _bags_to_instance(self, bags: list[np.ndarray]) -> np.ndarray:
        """
        Convert bags to instances by taking the last instance of each bag.
        Args:
            bags: list of bags (each is array-like of shape [ #instances × hidden_size ])
        Returns:
            instances: array-like of shape [ #bags × hidden_size ]
        """
        return np.vstack([bag[-1] for bag in bags])

    def process_input(self, X: list[np.ndarray] | np.ndarray) -> np.ndarray:
        if type(X) is np.ndarray:
            X = [X]
        return self.scaler.transform(self._bags_to_instance(X))

    def update_metric(self, metric_dict):
        """
        Add the metric items to the metric dictionary.
        """
        return metric_dict

    @property
    def direction(self):
        """
        Return the direction of the separator.
        """
        return None

    @property
    def bias(self):
        """
        Return the bias of the separator.
        """
        return None

    @property
    def direction_bias(self):
        """
        Return, BOTH, the direction and bias of the separator.
        """
        return None, None

    @property
    def estimator(self):
        """
        Return the estimator
        """
        return self.separator

    def process_bag(self, bag: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Process a single bag and return the transformed representation.
        Args:
            bag: (L, d) array of input data for a single bag
        Returns:
            transformed_bag: (d',) array of transformed representation
        """
        return self.scaler.transform(bag)

    def bag_decision_function(
        self, bags: list[np.ndarray], agg: str = "max"
    ) -> np.ndarray:
        """
        Compute the decision function for the given bags.
        """
        # Transform the bags using the fitted scaler
        yhat = []
        for bag in bags:
            bp = self.process_bag(bag)
            preds = self.separator.decision_function(bp)
            if agg == "max":
                yhat.append(np.max(preds))
            elif agg == "mean":
                yhat.append(np.mean(preds))
            else:
                raise ValueError(f"Unknown aggregation method: {agg}")

        return np.array(yhat).flatten()

    def bag_predict_proba(self, bags: list[np.ndarray], agg: str = "max") -> np.ndarray:
        """
        Compute the predicted probabilities for the given bags.
        """
        # Transform the bags using the fitted scaler
        proba = []
        for bag in bags:
            bp = self.process_bag(bag)
            preds = self.separator.predict_proba(bp)
            if agg == "max":
                # Probability of positive class
                proba.append(np.max(preds[1:]))
            elif agg == "mean":
                proba.append(np.mean(preds[1:]))
            else:
                raise ValueError(f"Unknown aggregation method: {agg}")
        return np.array(proba).flatten()

    def bag_predict(
        self, bags: list[np.ndarray], agg: str = "max", threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict the class labels for the given bags.
        """
        proba = self.bag_predict_proba(bags, agg=agg)
        return np.array(proba > threshold)

    def bag_conformal_prediction(
        self, bags: list[np.ndarray], agg: str = "max"
    ) -> list[set]:
        """
        Compute the conformal prediction for the given bags.
        """
        # Transform the bags using the fitted scaler
        # Compute the decision function using the separator
        yh = self.bag_decision_function(bags, agg=agg)
        # Compute the conformal prediction
        return self.calibrator.predict(yh)

    def inst_decision_function(self, X):
        """
        Predict raw scores for the LAST INSTANCE of each bag.
        """
        return self.decision_function(X)

    def inst_predict_proba(self, X):
        """
        Predict logits for the LAST INSTANCE of each bag.
        """
        return self.predict_proba(X)

    def inst_predict(self, X):
        """
        Predict classes for the LAST INSTANCE of each bag.
        """
        return self.predict(X)

    def inst_conformal_prediction(self, X):
        """
        Predict conformal classes for the LAST INSTANCE of each bag.
        """
        return self.conformal_prediction(X)

    def load(self, output_dir: str | Path, layer_id: int) -> TTPD_Runner:
        """
        Reload saved artifacts into this runner.
        Args:
            output_dir: path where save(...) stored things
            layer_id: integer id used in filenames
        """
        output_dir = Path(output_dir)

        manifest_path = output_dir / "manifests" / f"manifest_{layer_id}.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            paths = manifest["paths"]
        else:
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        if Path(paths["scaler"]).exists():
            self.scaler = joblib.load(paths["scaler"])
        else:
            raise FileNotFoundError(f"Scaler not found: {paths['scaler']}")

        if paths.get("estimator") and Path(paths["estimator"]).exists():
            self.separator = joblib.load(paths["estimator"])
        else:
            self.separator = None

        if paths.get("calibrator") and Path(paths["calibrator"]).exists():
            self.calibrator = joblib.load(paths["calibrator"])
        else:
            self.calibrator = None

        if paths.get("transformer") and Path(paths["transformer"]).exists():
            self.transformer = joblib.load(paths["transformer"])
        else:
            self.transformer = None

        self.is_fitted_ = True
        return self
