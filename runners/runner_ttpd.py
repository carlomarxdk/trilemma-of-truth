from runners.base import BaseProbeRunner
from probes.ttpd import TTPD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from probes.conformal import InductiveConformalPredictor, symmetric_nonconformity
from sklearn.metrics import (
    average_precision_score as mAP,
    matthews_corrcoef as mcc,
    make_scorer
)
import numpy as np
import logging
from typing import List
from copy import deepcopy

log = logging.getLogger("SILRunner-TTPD")


class TTPD_Runner(BaseProbeRunner):
    """
    SIL Probe Runner for probes trained on the last instance of the bag (for Training Truth and Polarity Direction probe)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        # set random seed
        np.random.seed(getattr(cfg.probe, 'seed', None))
        self.scaler = StandardScaler()
        self.calibrator = None
        self.separator = None
        self.transformer = None
        self.bag_processor = None

    def single_training(self, X: List[np.ndarray], y: np.ndarray, mask: np.ndarray, neg: np.ndarray = None):
        """
        Train transformer and separator on the masked subset of bags.
        Returns dict with 'separator' and fitted 'transformer'.
        Args:
            - X: an list of bags 
            - y: bag_abels
            - mask: mask for the task
        """
        # 0) Get the bags and labels
        f_mask = np.array(mask, dtype=bool)
        f_neg = deepcopy(neg) if neg is not None else None
        f_y = deepcopy(y)
        f_X = deepcopy(X)
        assert len(f_X) == len(f_y) == len(f_mask), "X, y and mask must have the same length"
        assert np.unique(f_y).size == 2, "y must be binary"

        ym = self.return_target(f_y, f_mask)
        negm = self.return_target(1 - f_neg, f_mask) if f_neg is not None else None
        # 1) Fit transformer on concatenated instances
        if self.cfg.probe.get('normalize_data', True):
            log.warning("\t\tNormalizing the data...")
            Xm = np.vstack([bag[-1] for bag, m in zip(f_X, f_mask) if m])
            self.scaler.fit(Xm)
        else:
            raise NotImplementedError(
                "Only a pipeline with the normalization is implemented")

        Xm = np.vstack([self.scaler.transform(bag)[-1]
                for bag, m in zip(f_X, f_mask) if m])

        # 2) Transform each bag (take only the last element)
        cfg = self.cfg.probe
        limit = cfg.get('train_sample_limit', Xm.shape[0])
        log.warning("\t\tFit the data...")

        self.separator = TTPD(
            verbose=cfg.init_params.get('verbose', False),
            random_seed=cfg.init_params.get('random_seed', 42)
        )

        self.separator.fit(
            Xm[:limit], t_labels=ym[:limit], p_labels=negm[:limit])

        return {'separator': self.separator,
                'scaler': self.scaler,
                'transformer': np.nan}

    def return_target(self, y, mask=None):
        yy = deepcopy(y)
        if mask is not None:
            return yy[mask]
        return yy

    def parameter_search(self, X: List[np.ndarray], y: np.ndarray, mask: np.ndarray, neg: np.ndarray = None):
        """
        Training with hyperparameter search
        Args:
            - X: an array of bags (Sequences, Lenghts, Hidden Size)
            - y: labels
            - mask: mask for the data
        """
        log.warning("Running the hyperparameter search... (For MD Probe parameter_search == sigle_training)")
        return self.single_training(X, y, mask, neg)
        
    def conformal_training(self, X_cal, y_cal, mask_cal):
        '''
        Train the conformal predictor on the calibration set.
        Args:
            X_cal: array-like, shape (n_samples, n_features)
                The calibration set features.
            y_cal: array-like, shape (n_samples,)
                The calibration set true labels.
            mask_cal: array-like, shape (n_samples,)
                The mask for the calibration set.
        '''
        f_X = deepcopy(X_cal)
        f_y = deepcopy(y_cal)
        f_mask = np.array(mask_cal, dtype=bool).copy()
        cfg = self.cfg.conformal_params

        if cfg['nc'] == 'binary':
            nc = symmetric_nonconformity
        else:
            raise NotImplementedError(
                f"Nonconformity function {cfg['nc']} is not implemented.")
        self.calibrator = InductiveConformalPredictor(nonconformity_func=nc,
                                                      alpha=cfg["alpha"],
                                                      tie_breaking=cfg["tie_breaking"])
        yh_cal = self.decision_function(f_X)
        self.calibrator.fit(y=f_y[f_mask], scores=yh_cal[f_mask])
        return self.calibrator

    def conformal_prediction(self, X):
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
    
    def process_input(self, X: List[np.ndarray]) -> np.ndarray:
        return np.vstack([self.scaler.transform(bag)[-1] for bag in X])

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