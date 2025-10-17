from __future__ import annotations
import numpy as np
from copy import deepcopy
import logging
from typing import List, Dict
from runners.base import BaseProbeRunner
from scipy.special import expit
from misc.probe_data import ProbeData
from probes.multiclass import OvAProjector  # bags -> scores
from probes.multiclass import MulticlassProbe

from probes.conformal import MulticlassICP, probability_margin_nc

log = logging.getLogger("MulticlassMILRunner")


class MulticlassMILRunner(BaseProbeRunner):
    """
    MIL Multiclass using OvA method.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.separator = None  # sklearn Pipeline (MulticlassProbe)
        self.scaler = None
        self.transformer = None
        self.calibrator = None
        self.layer_id = None

        # hyperparams
        self.pool = "max"
        self.max_iter = 2000

        #  dict like {1: 'T', 0: 'F', 2: 'N'}
        self.tasks = cfg.multiclass_params.tasks

    def return_target(self, y: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        In case of Multiclass Sawmil, just apply mask.
        Args:
            y: bag_labels or labels
            mask: mask for the task
        Returns:
            yy: masked bag_labels or labels
        """
        assert np.unique(y).tolist() == [
            0, 1, 2], "MulticlassMILRunner expects y to have classes [0,1,2]"
        yy = deepcopy(y)
        if mask is not None:
            return yy[mask]
        return yy
    # ---- path helpers  ----

    def _probe_dir_for(self, task: str) -> str:
        """
        Build per-class ProbeData instances
        """
        probe_name = self.cfg.probe['name']
        model_name = self.cfg.model["name"]
        # keep base trial part before '-task'
        trial_name = self.cfg.trial_name.split('-')[0]
        # for OvA we had per-task suffix (e.g., .../{trial}-{task}/)
        return f"outputs/probes/{probe_name}/{model_name}/{trial_name}-{task}/"

    def _build_readers(self) -> dict[int, ProbeData]:
        # class id mapping: 0=False, 1=True, 2=Neither
        paths = {
            0: self._probe_dir_for(self.tasks['F']),
            1: self._probe_dir_for(self.tasks['T']),
            2: self._probe_dir_for(self.tasks['N']),
        }
        readers = {k: ProbeData(v) for k, v in paths.items()}
        return readers

    # ---- runner API (shape compatible with run_experiment.py) ----
    def single_training(self, X: List[np.ndarray], y: np.ndarray, mask: np.ndarray, layer_id: int | None = None, **kwargs) -> Dict:
        """
        Train a model without the hyperparameter search.
        Args:
            X: list of bags (each is array-like of shape [ #instances × hidden_size ])
            y: full array of bag-labels (0,1,2)
            mask: boolean mask array of length len(X) indicating which bags to train on
        Returns a dict with keys:
            'separator', 'scaler', 'transformer', 'eta'
        """
        self.layer_id = self.layer_id if layer_id is None else layer_id

        f_X = deepcopy(X)
        f_y = np.asarray(y)
        f_mask = np.asarray(mask, dtype=bool)

        readers = self._build_readers()  # {0,1,2} -> ProbeData
        projector = OvAProjector(readers, layer_id=self.layer_id)
        self.separator = MulticlassProbe(
            projector, pool=self.pool, max_iter=self.max_iter)

        bags = [b for b, m in zip(f_X, f_mask) if m]
        ym = self.return_target(f_y, f_mask)
        log.warning(
            f"Fitting MulticlassProbe on {len(bags)} bags (layer {self.layer_id}, pool={self.pool})")
        self.separator.fit(bags, ym)

        return {"separator": self.separator,
                "scaler": None,
                "transformer": None,
                "layer_id": self.layer_id}

    def parameter_search(self, X: List[np.ndarray], y: np.ndarray, mask: np.ndarray, layer_id: int | None = None, **kwargs) -> Dict:
        """
        Training with hyperparameter search
        Args:
            X: an array of bags (Sequences, Lenghts, Hidden Size)
            y: labels
            mask: mask for the data
            layer_id: which layer to use
        Returns a dict with keys:
            'separator', 'scaler', 'transformer', 'eta'
        """
        log.warning(
            "Parameter search not implemented for MulticlassMILRunner; doing single_training.")
        return self.single_training(X, y, mask, layer_id=layer_id, **kwargs)

    def conformal_training(self, X_cal: List[np.ndarray], y_cal: np.ndarray, mask_cal: np.ndarray) -> MulticlassICP:
        """
        Train the MultiClass ICP on the calibration split.
        Args:
            X_cal: list of bags for calibration
            y_cal: array of bag-labels for calibration
            mask_cal: boolean mask array for calibration
        Returns:
            the fitted MulticlassICP calibrator
        """
        probs_cal = self.predict_proba(
            [b for b, m in zip(X_cal, mask_cal) if m])
        y_c = np.asarray(y_cal)[mask_cal]
        self.calibrator = MulticlassICP(
            nonconformity_func=probability_margin_nc,
            alpha=self.cfg.conformal_params["alpha"],
            n_classes=3,
            tie_breaking=self.cfg.conformal_params["tie_breaking"]
        )
        self.calibrator.fit(y=y_c, scores=probs_cal)
        return self.calibrator

    def conformal_prediction(self, X: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute the conformal prediction for the given bags.
        """
        probs = self.predict_proba(X)
        return self.calibrator.predict(probs)

    def update_metric(self, metric_dict):
        return metric_dict

    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict logits for a new set of bags.
        Based on the FULL BAG (not just the last instance).
        """
        return self.separator.predict_proba(X)  # (N, C)

    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict the class for a new set of bags.
        Based on the FULL BAG (not just the last instance).
        """
        return self.separator.predict(X)        # (N,)

    def decision_function(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict raw scores for a new set of bags.
        Based on the FULL BAG (not just the last instance).
        Args:
            X: list of bags (each is array-like of shape [ #instances × hidden_size ])
        Returns:
            scores: array of shape (N,) where N is the number of bags
        """
        return self.separator.decision_function(X)
    
    # Adapter methods for BAG-LEVEL Predictions

    def bag_decision_function(self, bags: List[np.ndarray], agg: str = 'max') -> np.ndarray:
        """
        Predict raw scores for a new set of bags (based on FULL bag).
        """
        return self.decision_function(bags)

    def bag_predict_proba(self, bags: List[np.ndarray], agg: str = 'max') -> np.ndarray:
        """
        Predict logits for a new set of bags (based on FULL bag).
        """
        return self.predict_proba(bags)

    def bag_predict(self, bags: List[np.ndarray], agg: str = 'max', threshold: float = 0.5) -> np.ndarray:
        """
        Predict classes for a new set of bags (based on FULL bag).
        """
        return self.predict(bags)

    def bag_conformal_prediction(self, bags: List[np.ndarray], agg: str = 'max') -> np.ndarray:
        """
        Predict conformal classes for a new set of bags (based on FULL bag).
        """
        return self.conformal_prediction(bags)

    # Adapter methods for INSTANCE-LEVEL Predictions (last instance of each bag)
    def _process_bag_to_instances(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Transform a list of bags into an instance (aka take the last instance of each bag after scaling).
        Args:
            X: list of N bags (each bag is an array of shape (L, d))
        Returns:
            Xt: array of shape (N, d) where N is the number of bags and
                d is the feature dimension after scaling
        """
        return [[bag[-1]] for bag in X]
    
    def inst_decision_function(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict raw scores for the LAST INSTANCE of each bag.
        """
        if type(X) is np.ndarray:
            X = [X]

        Xt = self._process_bag_to_instances(X)
        return self.decision_function(Xt)

    def inst_predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict logits for the LAST INSTANCE of each bag.
        """
        if type(X) is np.ndarray:
            X = [X]

        Xt = self._process_bag_to_instances(X)
        return self.predict_proba(Xt)
    
    def inst_predict(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict classes for the LAST INSTANCE of each bag.
        """
        if type(X) is np.ndarray:
            X = [X]

        Xt = self._process_bag_to_instances(X)
        return self.predict(Xt)

    def inst_conformal_prediction(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict conformal classes for the LAST INSTANCE of each bag.
        """
        scores = self.predict_proba(X)
        return self.calibrator.predict(scores)