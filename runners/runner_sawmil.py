from runners.base import BaseProbeRunner
from sklearn.metrics import (
    average_precision_score as mAP,
)
import numpy as np
import logging
from copy import deepcopy

from probes.sawmil import sAwMIL
from probes.conformal import InductiveConformalPredictor, symmetric_nonconformity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from utils_hydra import drop_rows_with_tail_keep
from typing import List
import joblib
import json
from scipy.special import expit
from pathlib import Path
from probes.linear import BinaryLinearProbe
log = logging.getLogger("MILRunner-Sawmil")


class SawmilProbeRunner(BaseProbeRunner):
    """
    Sparse Aware Multiple Instance Learning (MIL) SVM (sAwMIL)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        np.random.seed(getattr(cfg, "random_seed", 42))
        self.separator = None  # placeholder
        self.scaler = StandardScaler()
        self.transformer = None
        self.calibrator = None

        self.eta = None

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
        # 0) Get the bags and labels

        f_y = deepcopy(y)
        f_X = deepcopy(X)
        f_mask = np.array(mask, dtype=bool)
        assert len(f_X) == len(f_y) == len(
            f_mask), "X, y and mask must have the same length"
        assert np.unique(f_y).size == 2, "y must be binary"

        ym = self.return_target(f_y, f_mask)
        # 1) Fit transformer on concatenated instances
        if self.cfg.probe.get("normalize_data", True):
            # vstack all bags (including those we’ll later sparsify)
            X_all = np.vstack([bag for bag, m in zip(f_X, f_mask) if m])
            self.scaler.fit(X_all)
            # transform each bag
        else:
            raise NotImplementedError(
                "Only normalization pipeline is implemented")

        # 2) Transform each bag: cap bag size and assign intra‐bag labels
        processed_bags = []
        intra_bag_labels = []
        max_bag_size = self.cfg.probe["max_bag_size"]
        for i, bag in enumerate(f_X):
            bag_processed, intra_labels_for_this_bag = self._process_bag_for_training(
                bag=bag,
                max_bag_size=max_bag_size,
                rnd_seed_offset=i
            )
            processed_bags.append(bag_processed)
            intra_bag_labels.append(intra_labels_for_this_bag)

        # 2.1) Compute η (eta) ===========
        pos_lengths = [
            len(bag) for bag in processed_bags
        ]
        eta = sum([sum(lbl) for lbl in intra_bag_labels]) / sum(pos_lengths)
        self.eta = eta

        # 3) Fit the model
        log.warning("\t\tFit the data...")

        separator = sAwMIL(
            C=self.cfg.probe["init_params"]["C"],
            kernel=self.cfg.probe["init_params"]["kernel"],
            scale_C=self.cfg.probe["init_params"]["scale_C"],
            verbose=self.cfg.probe["init_params"]["verbose"],
            eta=self.eta,
        )
        limit = self.cfg.probe.get("train_bag_limit", len(processed_bags))

        separator.fit(
            bags=processed_bags[:limit],
            y=ym[:limit],
            in_bag_labels=intra_bag_labels[:limit]
        )
        self.separator = separator

        return {
            "separator": separator,
            "scaler": self.scaler,
            "transformer": self.transformer,
            "eta": eta,
        }

    def parameter_search(self, X: List[np.ndarray], y: np.ndarray, mask: np.ndarray, neg: np.ndarray = None):
        """
        Wraps the existing parameter_search(...) function from your script.
        Args:
            - X: list of bags
            - y: array of labels (0/1)
            - mask: boolean mask array
        Returns:
            same dict as single_training plus 'best_C'
        """
        log.warning("Running the hyperparameter search...")
        # You can redefine this based on your needs
        param_grid = self.cfg.probe.param_grid['C']
        f_X = deepcopy(X)
        f_y = deepcopy(y)
        f_mask = np.array(mask, dtype=bool)

        ym = self.return_targets(f_y, None)  # mask should be None

        assert len(f_X) == len(f_y) == len(
            f_mask), "X, y and mask must have the same length"
        assert np.unique(f_y).size == 2, "y must be binary"
        random_seed = self.cfg.get("random_seed", 42)

        kf = KFold(n_splits=self.cfg.get("cv_n_folds", 3), shuffle=True,
                   random_state=random_seed)
        kf.get_n_splits(X)
        scores = []
        stds = []
        n_samples = len(X)

        for i, C in enumerate(param_grid):
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

                # 2) Transform each bag: cap bag size and assign intra‐bag labels
                if self.cfg.probe.get("normalize_data", True):
                    log.warning("\t\tNormalizing the data...")
                    Xt = np.vstack([bag for bag in X_train])
                    scaler = StandardScaler()
                    scaler.fit(Xt)
                    # transform each bag
                else:
                    raise NotImplementedError(
                        "Only normalization pipeline is implemented")

                max_bag_size = self.cfg.probe["max_bag_size"]
                limit = self.cfg.get('cv_bag_limit', len(f_X))
                processed_bags = []
                intra_bag_labels = []

                for i, bag in enumerate(f_X):
                    bag_processed, intra_labels_for_this_bag = self._process_bag_for_training(
                        bag=bag,
                        max_bag_size=max_bag_size,
                        rnd_seed_offset=i
                    )
                    processed_bags.append(bag_processed)
                    intra_bag_labels.append(intra_labels_for_this_bag)

                # 2.1) Compute η (eta)
                pos_lengths = [
                    len(bag) for bag in processed_bags
                ]
                eta = sum([sum(lbl) for lbl in intra_bag_labels]) / \
                    sum(pos_lengths)
                try:
                    separator = sAwMIL(
                        C=float(C),
                        kernel=self.cfg.probe.get('kernel', 'linear'),
                        scale_C=self.cfg.probe.get('scale_C', True),
                        verbose=False,
                        eta=eta,
                    )
                    separator.fit(
                        bags=processed_bags[:limit],
                        y=y_train[:limit],
                        in_bag_labels=intra_bag_labels[:limit],
                    )
                    direction, bias = separator.linearize(normalize=True)
                    y_hat = self._decision_function_(
                        X_test, direction=direction, bias=bias, scaler=scaler)
                    _inner_scores.append(mAP(y_test, y_hat))
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
            f"\tSelected via 1-SE: n_components={params[selected_idx]['clf__n_components']} (mean AP={selected_score:.4f})")
        return params[selected_idx], selected_score

    def conformal_training(self, X_cal: List[np.ndarray], y_cal: np.ndarray, mask_cal: np.ndarray):
        """
        Train the InductiveConformalPredictor on the calibration split.
        Args:
            X_cal: list of bags for calibration
            y_cal: array of bag-labels for calibration
            mask_cal: boolean mask array for calibration
        Returns:
            the fitted InductiveConformalPredictor
        """
        f_X = deepcopy(X_cal)
        f_y = deepcopy(y_cal)
        f_mask = np.array(mask_cal, dtype=bool).copy()
        # keep original {0,1} labels—Conformal predictor takes raw scores + true labels
        cfg = self.cfg.conformal_params

        if cfg["nc"] == "binary":
            nc_func = symmetric_nonconformity
        else:
            raise NotImplementedError(f"NC {cfg['nc']} not implemented")

        # compute “scores” via current separator
        self.calibrator = InductiveConformalPredictor(
            nonconformity_func=nc_func,
            alpha=cfg["alpha"],
            tie_breaking=cfg["tie_breaking"],
        )
        yh_cal = self.decision_function(f_X)

        self.calibrator.fit(y=f_y[f_mask], scores=yh_cal[f_mask])
        return self.calibrator

    def conformal_prediction(self, X: List[np.ndarray]):
        """
        Compute the conformal prediction for the given bags.
        """
        # Transform the bags using the fitted scaler
        f_X = deepcopy(X)
        # Compute the decision function using the separator
        yh = self.decision_function(f_X)
        # Compute the conformal prediction
        return self.calibrator.predict(yh)

    def decision_function(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict raw scores for a new set of bags.
        Based on the FULL BAG (not just the last instance).
        Args:
            X: list of bags (each is array-like of shape [ #instances × hidden_size ])
        Returns:
            scores: array of shape (N,) where N is the number of bags
        """
        output = []
        if type(X) is np.ndarray:
            X = [X]

        for bag in X:
            bag = self.process_bag(bag)
            scores = self.separator.decision_function(bag)
            output.append(np.max(scores))
        return np.array(output)

    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict logits for a new set of bags.
        Based on the FULL BAG (not just the last instance).
        """
        return expit(self.decision_function(X))

    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict the class for a new set of bags.
        Based on the FULL BAG (not just the last instance).
        """
        return self.predict_proba(X).round()
    
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
    
    def inst_decision_function(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict raw scores for the LAST INSTANCE of each bag.
        """
        if type(X) is np.ndarray:
            X = [X]

        Xt = self._process_bag_to_instances(X)
        return self.estimator.decision_function(Xt)

    def inst_predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict logits for the LAST INSTANCE of each bag.
        """
        return expit(self.inst_decision_function(X))

    def inst_predict(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict classes for the LAST INSTANCE of each bag.
        """
        return self.inst_predict_proba(X).round()
    
    def inst_conformal_prediction(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict conformal classes for the LAST INSTANCE of each bag.
        """
        scores = self.inst_decision_function(X)
        return self.calibrator.predict(scores)

    def process_input(self, X: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError(
            "Use process_bag instead or process_instance")

    def process_bag(self, bag: np.ndarray) -> np.ndarray:
        """
        Process a single bag and return the transformed representation.
        Args:
            bag: (L, d) array of input data for a single bag
        Returns:
            transformed_bag: (d',) array of transformed representation
        """
        return self.scaler.transform(bag)

    def _process_bag_to_instances(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Transform a list of bags into an instance (aka take the last instance of each bag after scaling).
        Args:
            X: list of N bags (each bag is an array of shape (L, d))
        Returns:
            Xt: array of shape (N, d) where N is the number of bags and
                d is the feature dimension after scaling
        """
        return np.vstack([self.process_bag(bag)[-1] for bag in X])

    def update_metric(self, metric_dict):
        """
        Add probe‐specific hyperparameters to metrics (e.g., C, eta).
        """
        metric_dict["C"] = self.separator.C
        metric_dict["eta"] = self.eta
        return metric_dict

    @property
    def direction(self):
        """
        Return the direction of the separator.
        """
        return self.separator.linearize(normalize=True)[0]

    @property
    def bias(self):
        """
        Return the bias of the separator.
        """
        return self.separator.linearize(normalize=True)[1]

    @property
    def direction_bias(self):
        """
        Return, BOTH, the direction and bias of the separator.
        """
        return self.separator.linearize(normalize=True)

    @property
    def estimator(self):
        """
        Return the trained separator.
        """
        dir, bias = self.direction_bias
        return BinaryLinearProbe(
            coef=dir.reshape(1, -1),
            bias=bias
        )

    def _process_bag_for_training(self, bag: np.ndarray, max_bag_size: int = 100, rnd_seed_offset: int = 0):
        ''' 
        Process a single bag (use this during the training phase).
        Args:
            bag: np.array, shape [ #instances × hidden_size ]
            max_bag_size: int, maximum size of the bag after processing
            rnd_seed_offset: int, offset for the random seed (to ensure different random behavior for different bags)
        Returns:
            processed_bag: np.array, processed bag
            intra_labels_for_this_bag: np.array, intra bag labels used by the `sAwMIL` probe (for training)
        '''
        num_last_tokens_to_keep = self.cfg.probe["num_known_positives"]
        assume_known = self.cfg.probe.get("assume_known_positives", True)
        bag = self.process_bag(bag)

        L = bag.shape[0]
        # 1. Cap the bag size
        if L > max_bag_size:
            # drop FROM the tail but keep last `num_last_tokens_to_keep` items
            rng = self.cfg.random_seed + rnd_seed_offset
            processed_bag = drop_rows_with_tail_keep(
                bag, max_bag_size, num_last_tokens_to_keep, rng
            )
        else:
            processed_bag = bag
        # 2. Compute intra‐bag labels
        if assume_known:
            # last `num_last_tokens_to_keep` indices are “positive,” rest are “negative”
            intra_labels_for_this_bag = (
                [0] * (processed_bag.shape[0] - num_last_tokens_to_keep)
                + [1] * num_last_tokens_to_keep
            )
        else:
            intra_labels_for_this_bag = [1] * processed_bag.shape[0]
        return processed_bag, intra_labels_for_this_bag

    def update_metric(self, metric_dict):
        """
        Add the metric items to the metric dictionary.
        """
        metric_dict['C'] = self.separator.C
        metric_dict['kernel'] = self.separator.kernel
        metric_dict['scale_C'] = self.separator.scale_C
        return metric_dict

    def load(self, output_dir: str | Path, layer_id: int) -> 'SawmilProbeRunner':
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
                coef=np.load(paths["direction"]),
                bias=np.load(paths["bias"])
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
