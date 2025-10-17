from runners.base import BaseProbeRunner
from probes.silSVM_patch import SVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from probes.conformal import InductiveConformalPredictor, symmetric_nonconformity
from sklearn.metrics import (
    average_precision_score as mAP,
)
import numpy as np
from typing import List
import logging
from copy import deepcopy
from probes.linear import BinaryLinearProbe
from pathlib import Path
import joblib
from scipy.special import expit
import json

log = logging.getLogger("SILRunner-SVM")


class SVMProbeRunner(BaseProbeRunner):
    """
    Single-instance SVM Runner for probes trained on the last instance of the bag.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        # set random seed
        np.random.seed(getattr(cfg.probe, 'seed', None))
        self.separator = None
        self.scaler = StandardScaler()
        self.calibrator = None
        self.transformer = None

    def return_target(self, y: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Turn {0,1} labels into {−1,+1} and apply mask.
        Args:
            y: bag_labels or labels
            mask: mask for the task
        Returns:
            yy: masked bag_labels or labels
        """
        arr = np.ones_like(y)
        arr[y == 0] = -1
        if mask is not None:
            return arr[mask]
        return arr

    def single_training(self, X: List[np.ndarray], y: np.ndarray, mask: np.ndarray, neg: np.ndarray = None):
        """
        Train a model without the hyperparameter search.
        Args:
            X: list of bags (each is array-like of shape [ #instances × hidden_size ])
            y: full array of bag-labels (0/1)
            mask: boolean mask array of length len(X) indicating which bags to train on
        Returns a dict with keys:
            'separator', 'scaler', 'transformer', 'eta'
        """

        f_y = deepcopy(y)
        f_X = deepcopy(X)
        f_mask = np.array(mask, dtype=bool)
        assert len(f_X) == len(f_y) == len(
            f_mask), "X, y and mask must have the same length"
        assert np.unique(f_y).size == 2, "y must be binary"

        ym = self.return_target(f_y, f_mask)

        # 1) Fit transformer on concatenated instances
        if self.cfg.probe.get('normalize_data', True):
            log.warning("\t\tNormalizing the data...")
            self.scaler.fit(self._bags_to_instance(f_X)[f_mask])
        else:
            raise NotImplementedError(
                "Only a pipeline with the normalization is implemented")

        bags = [bag for bag, m in zip(
            self._bags_to_single_instance(f_X), f_mask) if m]

        # 2) Transform each bag (take only the last element)
        cfg = self.cfg.probe
        limit = cfg.get('train_bag_limit', len(bags))
        self.separator = SVM(C=cfg.init_params['C'],
                             kernel=cfg.init_params['kernel'],
                             scale_C=cfg.init_params.get('scale_C', True),
                             verbose=cfg.init_params.get('verbose', True))
        self.separator.fit(
            bags[:limit], ym[:limit])

        return {'separator': self.separator,
                'scaler': self.scaler,
                'transformer': np.nan}

    def parameter_search(self, X: List[np.ndarray], y: np.ndarray, mask: np.ndarray, neg: np.ndarray = None):
        """
        Training with hyperparameter search
        Args:
            - X: an array of bags (Sequences, Lenghts, Hidden Size)
            - y: labels
            - mask: mask for the data
        """
        log.warning("Running the hyperparameter search...")
        param_grid = self.cfg.probe.param_grid['C']
        f_y = deepcopy(y)
        f_X = deepcopy(X)
        f_mask = np.array(mask, dtype=bool)
        ym = self.return_target(f_y, None)  # mask should be None

        assert len(f_X) == len(f_y) == len(
            f_mask), "X, y and mask must have the same length"
        assert np.unique(f_y).size == 2, "y must be binary"
        random_seed = self.cfg.get("random_seed", 42)

        # Cross-validation setup
        kf = KFold(n_splits=self.cfg.get("cv_n_folds", 3), shuffle=True,
                   random_state=random_seed)
        kf.get_n_splits(X)
        scores = []
        stds = []
        n_samples = len(X)

        for _, C in enumerate(param_grid):
            log.warning(f"\tRunning the iteration with C={C}...")
            _inner_scores = []
            for j, (train_index, test_index) in enumerate(kf.split(X)):
                # Initialize boolean masks
                tr_mask = np.zeros(n_samples, dtype=bool)
                te_mask = np.zeros(n_samples, dtype=bool)
                # Set True for the respective indices
                tr_mask[train_index] = True
                te_mask[test_index] = True
                tr_mask = tr_mask & f_mask
                te_mask = te_mask & f_mask
                X_train = [x for x, m in zip(f_X, tr_mask) if m]
                X_test = [x for x, m in zip(f_X, te_mask) if m]
                y_train = ym[tr_mask]
                y_test = ym[te_mask]
                np.random.seed(random_seed + j)

                # Step 1: Normalize the data
                if self.cfg.probe.get('normalize_data', True):
                    log.warning("\t\tNormalizing the data...")
                    Xt = self._bags_to_instance(X_train)
                    scaler = StandardScaler()
                    scaler.fit(Xt)
                    # Transform bags
                    bags = [scaler.transform(bag)[-1]
                            for bag in X_train]
                    bags_test = [scaler.transform(bag)[-1]
                                 for bag in X_test]
                else:
                    raise NotImplementedError(
                        "Only a pipeline with the normalization is implemented")
                limit = self.cfg.get('cv_bag_limit', len(bags))

                try:
                    separator = SVM(C=float(C),
                                    kernel=self.cfg.probe.get(
                                        'kernel', 'linear'),
                                    scale_C=self.cfg.probe.get(
                                        'scale_C', True),
                                    verbose=False)
                    separator.fit(
                        bags[:limit], y_train[:limit])
                    direction, bias = separator.linearize(normalize=True)

                    y_te = np.dot(bags_test, direction) + bias

                    _inner_scores.append(mAP(y_test, y_te))
                    log.warning(
                        f"\t\tmAP for {j}th fold: {_inner_scores[-1]}")
                except Exception as e:
                    log.error(f"Error: {e}")
                    log.warning(
                        "\t\tMoving to the next one...")
                    _inner_scores.append(0.1)

            scores.append(np.mean(_inner_scores))
            stds.append(np.std(_inner_scores))
            log.warning(f"\tMean mAP for {C}: {scores[-1]}")

        selected_C, _ = self._apply_se_rule(scores, stds)
        self.cfg.probe.init_params["C"] = selected_C
        log.warning(
            f"MODEL: Retraining with the best C: {self.cfg.probe.init_params['C']}...")
        result = self.single_training(f_X, f_y, f_mask)

        return {
            "separator": result["separator"],
            "scaler": result["scaler"],
            "transformer": result["transformer"],
            "best_C": self.cfg.probe["init_params"]["C"],
        }

    def _apply_se_rule(self, scores, stds):
        means = scores
        n_folds = self.cfg["cv_n_folds"]
        params = self.cfg.probe["param_grid"]

        best_idx = int(np.argmax(means))
        best_score = float(means[best_idx])
        best_se = float(stds[best_idx] / np.sqrt(n_folds))

        threshold = best_score - best_se
        selected_idx = None
        for i, (m, p) in enumerate(zip(means, params)):
            if m >= threshold:
                selected_idx = i
                break
        if selected_idx is None:
            selected_idx = best_idx  # fallback

        selected_score = float(means[selected_idx])
        log.warning(
            f"\tSelected via 1-SE: n_components={params['C'][selected_idx]} (mean AP={selected_score:.4f})")
        return params['C'][selected_idx], selected_score

    def conformal_training(self, X_cal, y_cal, mask_cal):
        '''
        Train the conformal predictor on the calibration set.
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
        # Compute the decision function using the separator
        return np.dot(Xt, self.direction) + self.bias

    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict logits for a new set of bags.
        Based on the FULL BAG (not just the last instance).
        """
        return expit(self.decision_function(X))

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.array(proba > 0.5)

    def update_metric(self, metric_dict):
        """
        Add the metric items to the metric dictionary.
        """
        metric_dict['C'] = self.separator.C
        metric_dict['kernel'] = self.separator.kernel
        metric_dict['scale_C'] = self.separator.scale_C
        return metric_dict


    @property
    def direction(self):
        """
        Return the direction of the separator.
        """
        try:
            return self.separator.linearize(normalize=True)[0]
        except:
            try:
                return self._direction
            except:
                return None

    @property
    def bias(self):
        """
        Return the bias of the separator.
        """
        try:
            return self.separator.linearize(normalize=True)[1]
        except:
            try:
                return self._bias
            except:
                return None

    @property
    def direction_bias(self):
        """
        Return, BOTH, the direction and bias of the separator.
        """
        try:
            return self.separator.linearize(normalize=True)
        except:
            try:
                return self._direction, self._bias
            except:
                return None, None

    @property
    def estimator(self):
        """
        Return the trained separator.
        """
        try:
            return self.separator
        except:
            dir, bias = self.direction_bias
            return BinaryLinearProbe(
                coef=dir.reshape(1, -1),
                intercept=bias
            )

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

    def _bags_to_instance(self, bags: List[np.ndarray]) -> np.ndarray:
        """
        Convert bags to instances by taking the last instance of each bag.
        Args:
            bags: list of bags (each is array-like of shape [ #instances × hidden_size ])
        Returns:
            instances: array-like of shape [ #bags × hidden_size ]
        """
        return np.vstack([bag[-1] for bag in bags])

    def _bags_to_single_instance(self, bags: List[np.ndarray]) -> np.ndarray:
        """
        Convert bags to instances by taking the last instance of each bag.
        Args:
            bags: list of bags (each is array-like of shape [ #instances × hidden_size ])
        Returns:
            instances: array-like of shape [ #bags × hidden_size ]
        """
        return [self.scaler.transform(bag)[-1] for bag in bags]

    def process_input(self, X: List[np.ndarray] | np.ndarray) -> np.ndarray:
        if type(X) is np.ndarray:
            X = [X]
        return self.scaler.transform(self._bags_to_instance(X))

    def load(self, output_dir: str | Path, layer_id: int) -> 'SVMProbeRunner':
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
            self.separator = BinaryLinearProbe(
                coef=np.load(paths["coef"]),
                intercept=np.load(paths["bias"])
            )


        self._bias = np.load(paths["bias"])
        self._direction = np.load(paths["coef"])
        
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
