from runners.runner_base import BaseProbeRunner
from probes.silSVM_patch import SVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from probes.conformal import InductiveConformalPredictor, symmetric_nonconformity
from sklearn.metrics import (
    average_precision_score as mAP,
)
import numpy as np
import logging
from copy import deepcopy

log = logging.getLogger("SILRunner-SVM")


class SVMProbeRunner(BaseProbeRunner):
    """
    SIL Probe Runner for probes trained on the last instance of the bag (for SVM probe).
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

        bags = [self.scaler.transform(bag)[-1]
                for bag, m in zip(X, mask) if m]

        # 2) Transform each bag (take only the last element)
        cfg = self.cfg.probe
        limit = cfg.get('train_bag_limit', len(bags))
        self.separator = SVM(C=cfg.init_params['C'],
                             kernel=cfg.init_params['kernel'],
                             scale_C=cfg.init_params.get('scale_C', True),
                             verbose=cfg.init_params.get('verbose', True))
        self.separator.fit(
            bags[:limit], y[:limit])

        return {'separator': self.separator,
                'scaler': self.scaler,
                'transformer': np.nan}

    def return_target(self, y, mask=None):
        yy = np.ones_like(y)
        yy[y == 0] = -1
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
        log.warning("Running the hyperparameter search...")
        param_grid = self.cfg.probe.param_grid['C']
        f_mask = deepcopy(mask)
        f_X = deepcopy(X)
        f_y = deepcopy(y)
        # 0) Get the bags and labels
        assert len(X) == len(y), "X and y must have the same length"
        assert np.unique(y).size == 2, "y must be binary"
        mask = np.array(mask, dtype=bool)
        y = self.return_target(y, None)
        random_seed = self.cfg.get("random_seed", 42)
        # 1) Fit transformer on concatenated instances

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
                tr_mask = tr_mask & mask
                te_mask = te_mask & mask
                X_train = [x for x, m in zip(X, tr_mask) if m]
                X_test = [x for x, m in zip(X, te_mask) if m]
                y_train = y[tr_mask]
                y_test = y[te_mask]

                np.random.seed(random_seed + j)
                # Step 1: Normalize the data
                if self.cfg.probe.get('normalize_data', True):
                    log.warning("\t\tNormalizing the data...")
                    Xt = np.vstack(X_train)
                    scaler = StandardScaler()
                    scaler.fit(Xt)
                    bags = [scaler.transform(bag)[-1]
                            for bag in X_train]
                    bags_test = np.vstack([scaler.transform(bag)[-1]
                                           for bag in X_test])
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
        se = np.array(stds) / np.sqrt(self.cfg.get("cv_n_folds", 3))
        scores = np.array(scores)
        best_index = np.argmax(scores)
        best_score = scores[best_index]
        best_C = list(param_grid)[best_index]
        best_se = se[best_index]
        # 1STD rule
        selected_index = None
        for idx, score in enumerate(scores):
            if score > (best_score - best_se):
                selected_index = idx
                break
        try:
            selected_C = list(param_grid)[selected_index]
            selected_score = scores[selected_index]
        except Exception as e:
            selected_C = best_C
            selected_score = best_score
            log.warning(
                f"\t\tCould not find a C within 1SE. Using the best C.")
            log.warning(f"\t\tScores: {scores}")
            log.warning(f"\t\tSE: {se}")

        log.warning(
            f"\t\tBest C: {best_C} with mAP: {best_score} (se {best_se})")
        log.warning(f"\t\tSelected C: {selected_C} with mAP: {selected_score}")

        self.cfg.probe["init_params"]["C"] = selected_C
        log.warning(
            f"MODEL: Retraining with the best C: {self.cfg.probe['init_params']['C']}...")
        result = self.single_training(f_X, f_y, f_mask)

        return {
            "separator": result["separator"],
            "scaler": result["scaler"],
            "transformer": result["transformer"],
            "best_C": self.cfg.probe["init_params"]["C"],
        }

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
    
    def process_input(self, X):
        return np.vstack([self.scaler.transform(bag)[-1] for bag in X])

    def decision_function(self, X):
        """
        Compute the decision function for the given bags.
        """
        # Transform the bags using the fitted scaler
        Xt = self.process_input(X)
        # Compute the decision function using the separator
        return np.dot(Xt, self.direction) + self.bias

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
