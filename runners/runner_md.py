from runners.base import BaseProbeRunner
from probes.mean_difference import MeanDifferenceClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from probes.conformal import InductiveConformalPredictor, symmetric_nonconformity
import joblib
import json
from pathlib import Path
import numpy as np
import logging
import numpy as np
import logging
from typing import List
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

    def parameter_search(self, X: List[np.ndarray], y: np.ndarray, mask: np.ndarray, neg: np.ndarray = None):
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
        return np.asarray(self.separator.coef_[0])

    @property
    def bias(self):
        """
        Return the bias of the separator.
        """
        return np.asarray(self.separator.intercept_ if self.separator.fit_intercept else 0.0)

    @property
    def direction_bias(self):
        """
        Return, BOTH, the direction and bias of the separator.
        """
        return self.direction, self.bias
    
    @property
    def estimator(self):
        """
        Return the estimator
        """
        return self.separator

    def process_bag(self, bag: np.ndarray) -> np.ndarray:
        """
        Process a single bag and return the transformed representation.
        Args:
            bag: (L, d) array of input data for a single bag
        Returns:
            transformed_bag: (d',) array of transformed representation
        """
        return self.scaler.transform(bag)

    def bag_decision_function(self, bags: List[np.ndarray], agg: str = 'max') -> np.ndarray:
        """
        Compute the decision function for the given bags.
        """
        # Transform the bags using the fitted scaler
        yhat = []
        for bag in bags:
            bp = self.process_bag(bag)
            preds = self.separator.decision_function(bp)
            if agg == 'max':
                yhat.append(np.max(preds))
            elif agg == 'mean':
                yhat.append(np.mean(preds))
            else:
                raise ValueError(f"Unknown aggregation method: {agg}")

        return np.array(yhat).flatten()

    def bag_predict_proba(self, bags: List[np.ndarray], agg: str = 'max') -> np.ndarray:
        """
        Compute the predicted probabilities for the given bags.
        """
        # Transform the bags using the fitted scaler
        proba = []
        for bag in bags:
            bp = self.process_bag(bag)
            preds = self.separator.predict_proba(bp)
            if agg == 'max':
                # Probability of positive class
                proba.append(np.max(preds[1:]))
            elif agg == 'mean':
                proba.append(np.mean(preds[1:]))
            else:
                raise ValueError(f"Unknown aggregation method: {agg}")
        return np.array(proba).flatten()

    def bag_predict(self, bags: List[np.ndarray], agg: str = 'max', threshold: float = 0.5) -> np.ndarray:
        """
        Predict the class labels for the given bags.
        """
        proba = self.bag_predict_proba(bags, agg=agg)
        return np.array(proba > threshold)

    def bag_conformal_prediction(self, bags: List[np.ndarray], agg: str = 'max') -> List[set]:
        """
        Compute the conformal prediction for the given bags.
        """
        # Transform the bags using the fitted scaler
        # Compute the decision function using the separator
        yh = self.bag_decision_function(bags, agg=agg)
        # Compute the conformal prediction
        return self.calibrator.predict(yh)

    def load(self, output_dir: str | Path, layer_id: int) -> 'MDProbeRunner':
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
