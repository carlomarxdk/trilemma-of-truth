from runners.runner_base import BaseProbeRunner
from probes.mean_difference import MeanDifferenceClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from probes.conformal import InductiveConformalPredictor, symmetric_nonconformity
from sklearn.metrics import (
    average_precision_score as mAP,
    matthews_corrcoef as mcc,
)
import numpy as np
import logging
from copy import deepcopy

log = logging.getLogger("SILRunner-MD")


class MDProbeRunner(BaseProbeRunner):
    """
    SIL Probe Runner for probes trained on the last instance of the bag (for mean difference probe)
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

    def single_training(self, X, y, mask):
        """
        Train transformer and separator on the masked subset of bags.
        Returns dict with 'separator' and fitted 'transformer'.
        Args:
            - X: an list of bags 
            - y: bag_abels
            - mask: mask for the task
        """
        # 0) Get the bags and labels
        assert len(X) == len(y), "X and y must have the same length"
        assert np.unique(y).size == 2, "y must be binary"
        
        mask = np.array(mask, dtype=bool)
        y = self.return_target(y, mask)
        # 1) Fit transformer on concatenated instances
        if self.cfg.probe.get('normalize_data', True):
            log.warning("\t\tNormalizing the data...")
            Xf = np.vstack([x[-1] for x, m in zip(X, mask) if m])
            self.scaler.fit(Xf)
        else:
            raise NotImplementedError(
                "Only a pipeline with the normalization is implemented")

        bags = np.vstack([self.scaler.transform(bag)[-1]
                for bag, m in zip(X, mask) if m])

        # 2) Transform each bag (take only the last element)
        cfg = self.cfg.probe
        limit = cfg.get('train_sample_limit', len(bags))
        self.separator = MeanDifferenceClassifier(with_covariance=cfg.init_params['with_covariance'],
                                                  fit_intercept=cfg.init_params['fit_intercept'],
                                                   verbose=cfg.init_params.get('verbose', True))
                                     
        self.separator.fit(
            bags[:limit], y[:limit])

        return {'separator': self.separator,
                'scaler': self.scaler,
                'transformer': np.nan}

    def return_target(self, y, mask=None):
        yy = deepcopy(y)
        if mask is not None:
            return yy[mask]
        return yy

    def parameter_search(self, X, y, mask):
        """
        Training with hyperparameter search
        Args:
            - X: an array of bags (Sequences, Lenghts, Hidden Size)
            - y: labels
            - mask: mask for the data
        """
        log.warning("Running the hyperparameter search... (For MD Probe parameter_search == sigle_training)")
        return self.single_training(X, y, mask)

    def conformal_training(self, X_cal, y_cal, mask_cal):
        '''
        Train the conformal predictor on the calibration set.
        '''
        X = deepcopy(X_cal)
        y = deepcopy(y_cal)
        mask = deepcopy(mask_cal)
        cfg = self.cfg.conformal_params
        mask_cal = np.array(mask, dtype=bool)

        if cfg['nc'] == 'binary':
            nc = symmetric_nonconformity
        else:
            raise NotImplementedError(
                f"Nonconformity function {cfg['nc']} is not implemented.")
        self.calibrator = InductiveConformalPredictor(nonconformity_func=nc,
                                                      alpha=cfg["alpha"],
                                                      tie_breaking=cfg["tie_breaking"])
        yh_cal = self.decision_function(X)
        self.calibrator.fit(y=y[mask], scores=yh_cal[mask])
        return self.calibrator

    def conformal_prediction(self, X):
        """
        Compute the conformal prediction for the given bags.
        """
        # Transform the bags using the fitted scaler
        X = deepcopy(X)
        # Compute the decision function using the separator
        yh = self.decision_function(X)
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
        return self.separator.predict_proba(Xt).flatten()
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array(proba > 0.5)    
    
    def process_input(self, X):
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
        return self.separator.coef_[0]

    @property
    def bias(self):
        """
        Return the bias of the separator.
        """
        return self.separator.intercept_ if self.separator.fit_intercept else 0.0

    @property
    def direction_bias(self):
        """
        Return, BOTH, the direction and bias of the separator.
        """
        return self.separator.direction, self.separator.bias
