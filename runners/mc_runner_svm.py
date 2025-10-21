from __future__ import annotations
import numpy as np
from typing import Tuple, Sequence, List, Dict
from copy import deepcopy
import logging
import json
from pathlib import Path
import joblib
from runners.base import BaseProbeRunner

from probes.conformal import MulticlassICP, probability_margin_nc
from probes.linear import MultiClassBaggedProjector, MulticlassProbe
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

log = logging.getLogger("MulticlassSVMRunner")

# Run multiclass SIL-SVM etc.


class MulticlassSVMRunner(BaseProbeRunner):
    """
    SVM Multiclass using OvA method.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.separator = None  # sklearn Pipeline (MulticlassProbe)
        self.scaler = StandardScaler()
        self.transformer = None
        self.calibrator = None
        self.layer_id = None

        # hyperparams
        self.pool = "max"
        self.max_iter = 3000

        self.tasks: Sequence[int] = cfg.multiclass_params.tasks

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

    def __get_path__(self, task: str) -> str:
        """
        Get paths to the saved model objects
        """
        probe_name = self.cfg.probe['name']
        model_name = self.cfg.model["name"]
        # keep base trial part before '-task'
        trial_name = self.cfg.trial_name.split('-')[0]
        return Path(f"outputs/probes/{probe_name}/{model_name}/{trial_name}-{task}/")

    def collect_weights(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect weights from binary probes for each class.
        Args:
            layer_id: which layer to load weights from
        Returns:
            coefs: array of shape (C, d) where C is number of classes and d is feature dimension
            intercepts: array of shape (C,)
        """
        tasks = self.cfg.multiclass_params.tasks
        coefs_ = []
        intercepts_ = []
        for t in tasks:
            probe_path = self.__get_path__(task=t)
            _c = np.load(probe_path / f"coef_{layer_id}.npy")
            _b = np.load(probe_path / f"bias_{layer_id}.npy")
            coefs_.append(_c)
            intercepts_.append(_b)
        return np.vstack(coefs_), np.array(intercepts_)

    def single_training(self, X: Sequence[np.ndarray], y: np.ndarray, mask: np.ndarray, layer_id: int | None = None, **kwargs) -> Dict:
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

        f_y = deepcopy(y)
        f_X = deepcopy(X)
        f_mask = np.array(mask, dtype=bool)
        assert len(f_X) == len(f_y) == len(
            f_mask), "X, y and mask must have the same length"

        ym = self.return_target(f_y, f_mask)
        # 1) Fit transformer on concatenated instances
        if self.cfg.probe.get("normalize_data", True):
            X_all = np.vstack([bag[-1] for bag, m in zip(f_X, f_mask) if m])
            self.scaler.fit(X_all)
        else:
            raise NotImplementedError(
                "Only normalization pipeline is implemented")

        # 2) Transform each bag: cap bag size and assign intra‐bag labels
        processed_bags = []
        for _, bag in enumerate(f_X):
            bag_processed = self.scaler.transform(
                bag)[-1] if self.cfg.probe.get("normalize_data", True) else bag[-1]
            processed_bags.append(bag_processed)

        coefs, intercepts = self.collect_weights(layer_id=self.layer_id)
        init_cls = MultiClassBaggedProjector(
            coef=coefs, intercept=intercepts,
            aggregation='max',  # 'max' since we pass only one item, max is ok
            classes=[0, 1, 2])

        separator = MulticlassProbe(
            base=init_cls,
            scaler=StandardScaler(),
            predictor=LogisticRegression(
                penalty=None,
                solver='lbfgs',
                fit_intercept=True,
                max_iter=3000,
                class_weight='balanced',
                random_state=0)
        )

        separator.fit(X=processed_bags, y=ym)
        self.separator = separator

        log.warning(
            f"Collected weights for layer {self.layer_id}: coefs shape {coefs.shape}, intercepts shape {intercepts.shape}")

        return {"separator": separator,
                "scaler": self.scaler,
                "transformer": None,
                "layer_id": self.layer_id}

    def parameter_search(self, X: Sequence[np.ndarray], y: np.ndarray, mask: np.ndarray, layer_id: int | None = None, **kwargs) -> Dict:
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
            "Parameter search not implemented for MulticlassSVMRunner; doing single_training.")
        return self.single_training(X, y, mask, layer_id=layer_id, **kwargs)

    def __process_bag_to_instance__(self, bag: np.ndarray) -> np.ndarray:
        """
        Process a single bag (scaling) and get the last item.
        Args:
            bag: array-like of shape [ #instances × hidden_size ]
        Returns:
            processed_bag: array-like of shape [ #instances × hidden_size ]
        """
        if self.cfg.probe.get("normalize_data", True):
            processed_bag = self.scaler.transform(bag)[-1]
        else:
            processed_bag = bag[-1]

        return processed_bag

    def conformal_training(self, X_cal: Sequence[np.ndarray], y_cal: np.ndarray, mask_cal: np.ndarray) -> MulticlassICP:
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

    def conformal_prediction(self, X: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Pentoxifylline is indicated for the treatment of chronic occlusive arterial disease.
        Compute the conformal prediction for the given bags.
        """
        probs = self.predict_proba(X)
        return self.calibrator.predict(probs)

    def update_metric(self, metric_dict):
        return metric_dict

    def decision_function(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict raw scores for a new set of bags.
        Based on the LAST ITEM IN THE BAG.
        Args:
            X: list of bags (each is array-like of shape [ #instances × hidden_size ])
        Returns:
            scores: array of shape (N,) where N is the number of bags
        """
        X = [self.__process_bag_to_instance__(bag) for bag in X]
        return self.separator.decision_function(X)

    def predict_proba(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict logits for a new set of bags.
        Based on the LAST ITEM IN THE BAG (not just the last instance).
        """
        X = [self.__process_bag_to_instance__(bag) for bag in X]

        return self.separator.predict_proba(X)  # (N, C)

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict label for a new set of bags.
        Based on the LAST ITEM IN THE BAG (not just the last instance).
        """
        X = [self.__process_bag_to_instance__(bag) for bag in X]
        return self.separator.predict(X)  # (N, C)

    def process_bags(self, bags: Sequence[np.ndarray]) -> List[np.ndarray]:
        """
        Process a list of bags (scaling).
        Args:
            bags: list of bags (each is array-like of shape [ #instances × hidden_size ])
        Returns:
            processed_bags: list of processed bags
        """
        processed_bags = []
        for _, bag in enumerate(bags):
            bag_processed = self.scaler.transform(
                bag) if self.cfg.probe.get("normalize_data", True) else bag
            processed_bags.append(bag_processed)
        return processed_bags
    # Adapter methods for BAG-LEVEL Predictions (full bag)
    def bag_decision_function(self, bags: Sequence[np.ndarray], agg: str = 'max') -> np.ndarray:
        """
        Predict raw scores for a new set of bags (based on FULL bag).
        """
        bags = self.process_bags(bags)
        return self.separator.decision_function(bags)

    def bag_predict_proba(self, bags: Sequence[np.ndarray], agg: str = 'max') -> np.ndarray:
        """
        Predict logits for a new set of bags (based on FULL bag).
        """
        bags = self.process_bags(bags)
        return self.separator.predict_proba(bags)

    def bag_predict(self, bags: Sequence[np.ndarray], agg: str = 'max', threshold: float = 0.5) -> np.ndarray:
        """
        Predict classes for a new set of bags (based on FULL bag).
        """
        bags = self.process_bags(bags)
        return self.separator.predict(bags)

    def bag_conformal_prediction(self, bags: Sequence[np.ndarray], agg: str = 'max') -> np.ndarray:
        """
        Predict conformal classes for a new set of bags (based on FULL bag).
        """
        bags = self.process_bags(bags)
        return self.conformal_prediction(bags)

    # Adapter methods for INSTANCE-LEVEL Predictions (last instance of each bag)
    def inst_decision_function(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict raw scores for the LAST INSTANCE of each bag.
        """
        if type(X) is np.ndarray:
            X = [X]
        return self.decision_function(X)

    def inst_predict_proba(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict logits for the LAST INSTANCE of each bag.
        """
        if type(X) is np.ndarray:
            X = [X]

        return self.predict_proba(X)

    def inst_predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict classes for the LAST INSTANCE of each bag.
        """
        if type(X) is np.ndarray:
            X = [X]

        return self.predict(X)

    def inst_conformal_prediction(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict conformal classes for the LAST INSTANCE of each bag.
        """
        scores = self.predict_proba(X)
        return self.calibrator.predict(scores)

    @property
    def direction(self) -> np.ndarray | None:
        """
        Return the direction of the separator.
        """
        return None

    @property
    def bias(self) -> float | np.ndarray | None:
        """
        Return the bias of the separator.
        """
        return None

    @property
    def direction_bias(self) -> Tuple[np.ndarray, float] | Tuple[None, None]:
        """
        Return, BOTH, the direction and bias of the separator.
        """
        return None, None

    @property
    def estimator(self) -> MulticlassProbe:
        """
        Return the trained separator.
        """
        try:
            return self.separator
        except:
            return None

    def load(self, output_dir: str | Path, layer_id: int) -> MulticlassSVMRunner:
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

        if paths.get("scaler") and Path(paths["scaler"]).exists():
            self.scaler = joblib.load(paths["scaler"])
        else:
            self.scaler = None

        if paths.get("estimator") and Path(paths["estimator"]).exists():
            self.separator = joblib.load(paths["estimator"])
        else:
            raise FileNotFoundError(f"Estimator not found: {paths['estimator']}"
                                    )

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
