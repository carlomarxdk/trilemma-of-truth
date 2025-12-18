from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from probes.conformal import InductiveConformalPredictor, symmetric_nonconformity
from probes.spca import SupervisedPCA
from runners.base import BaseProbeRunner

log = logging.getLogger("SILRunner-SPCA")


class SPCA_Runner(BaseProbeRunner):
    """
    SIL Probe Runner for probes trained on the last instance of the bag (for mean difference probe)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        # set random seed
        np.random.seed(getattr(cfg.probe, "seed", None))
        self.scaler = StandardScaler()
        self.calibrator = None
        self.separator = None
        self.transformer = None

    def single_training(
        self, X: Sequence[np.ndarray], y: np.ndarray, mask: np.ndarray, **kwargs
    ) -> dict:
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
        f_y = deepcopy(y)
        f_X = deepcopy(X)
        assert (
            len(f_X) == len(f_y) == len(f_mask)
        ), "X, y and mask must have the same length"
        assert np.unique(f_y).size == 2, "y must be binary"

        ym = self.return_target(f_y, f_mask)
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

        self.separator = SupervisedPCA(
            n_components=cfg.init_params["n_components"],
            cov_type=cfg.init_params.get("cov_type", "oas"),
            cov_fallback_scale=cfg.init_params.get("cov_fallback_scale", 1.0),
            verbose=cfg.init_params.get("verbose", False),
            random_seed=cfg.init_params.get("random_seed", 42),
        )

        self.separator.fit(Xm[:limit], ym[:limit])

        return {
            "separator": self.separator,
            "scaler": self.scaler,
            "transformer": np.nan,
        }

    def return_target(self, y, mask=None):
        yy = deepcopy(y)
        if mask is not None:
            return yy[mask]
        return yy

    def parameter_search(
        self, X: Sequence[np.ndarray], y: np.ndarray, mask: np.ndarray, **kwargs
    ) -> dict:
        """
        Training with hyperparameter search
        Args:
            - X: an array of bags (Sequences, Lenghts, Hidden Size)
            - y: labels
            - mask: mask for the data
        """
        params = self.cfg.probe.param_grid
        param_grid = {f"clf__{k}": v for k, v in params.items() if v is not None}
        f_mask = np.array(mask, dtype=bool)
        f_y = deepcopy(y)
        f_X = deepcopy(X)
        assert (
            len(f_X) == len(f_y) == len(f_mask)
        ), "X, y and mask must have the same length"
        assert np.unique(f_y).size == 2, "y must be binary"

        Xm = np.vstack([x[-1] for x, m in zip(f_X, f_mask) if m])
        ym = self.return_target(f_y, f_mask)

        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", SupervisedPCA())])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            refit=False,
            scoring="average_precision",
            n_jobs=-1,
            verbose=1,
            error_score=0.0,
        )

        grid.fit(Xm, ym)
        best_params, _ = self._apply_se_rule(grid.cv_results_, n_folds=3)
        self.cfg.probe.init_params["n_components"] = best_params["clf__n_components"]
        # self.cfg.probe.init_params['cov_reg'] = best_params['clf__cov_reg']
        return self.single_training(X, y, mask, **kwargs)

    def _apply_se_rule(self, results: dict, n_folds: int = 3) -> tuple[dict, float]:
        means = results["mean_test_score"]
        stds = results["std_test_score"]
        params = results["params"]

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
            f"\tSelected via 1-SE: n_components={params[selected_idx]['clf__n_components']} (mean AP={selected_score:.4f})"
        )
        return params[selected_idx], selected_score

    def conformal_training(
        self, X_cal: Sequence[np.ndarray], y_cal: np.ndarray, mask_cal: np.ndarray
    ) -> InductiveConformalPredictor:
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

    def conformal_prediction(self, X: Sequence[np.ndarray]):
        """
        Compute the conformal prediction for the given bags.
        """
        # Transform the bags using the fitted scaler
        f_X = deepcopy(X)
        # Compute the decision function using the separator
        yh = self.decision_function(f_X)
        # Compute the conformal prediction
        return self.calibrator.predict(yh)

    def decision_function(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Compute the decision function for the given bags.
        """
        # Transform the bags using the fitted scaler
        Xt = self.process_input(X)
        yhat = self.separator.decision_function(Xt)
        return yhat.flatten()

    def predict_proba(self, X: Sequence[np.ndarray]) -> np.ndarray:
        Xt = self.process_input(X)
        return self.separator.predict_proba(Xt).flatten()

    def predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.array(proba > 0.5)

    def process_input(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Transform a list of bags into an instance (aka take the last instance of each bag after scaling).
        Args:
            X: list of N bags (each bag is an array of shape (L, d))
        Returns:
            Xt: array of shape (N, d) where N is the number of bags and
                d is the feature dimension after scaling
        """
        return np.vstack([self.scaler.transform(bag)[-1] for bag in X])

    def process_to_instances(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Transform a list of bags into an instance (aka take the last instance of each bag after scaling).
        Args:
            X: list of N bags (each bag is an array of shape (L, d))
        Returns:
            Xt: array of shape (N, d) where N is the number of bags and
                d is the feature dimension after scaling
        """
        return self.process_input(X)

    def update_metric(self, metric_dict: dict) -> dict:
        """
        Add the metric items to the metric dictionary.
        """
        return metric_dict

    @property
    def direction(self) -> np.ndarray | None:
        """
        Return the direction of the separator.
        """
        return None

    @property
    def bias(self) -> float | None:
        """
        Return the bias of the separator.
        """
        return None

    @property
    def direction_bias(self) -> tuple[np.ndarray | None, float | None]:
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

    def process_bag(self, bag: np.ndarray) -> np.ndarray:
        """
        Process a single bag and return the transformed representation.
        Args:
            bag: (L, d) array of input data for a single bag
        Returns:
            transformed_bag: (d',) array of transformed representation
        """
        return self.scaler.transform(bag)

    def bag_decision_function(
        self, bags: Sequence[np.ndarray], agg: str = "max"
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

    def bag_predict_proba(
        self, bags: Sequence[np.ndarray], agg: str = "max"
    ) -> np.ndarray:
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
        self, bags: Sequence[np.ndarray], agg: str = "max", threshold: float = 0.5
    ) -> np.ndarray:
        """
        Predict the class labels for the given bags.
        """
        proba = self.bag_predict_proba(bags, agg=agg)
        return np.array(proba > threshold)

    def bag_conformal_prediction(
        self, bags: Sequence[np.ndarray], agg: str = "max"
    ) -> list[set]:
        """
        Compute the conformal prediction for the given bags.
        """
        # Transform the bags using the fitted scaler
        # Compute the decision function using the separator
        yh = self.bag_decision_function(bags, agg=agg)
        # Compute the conformal prediction
        return self.calibrator.predict(yh)

    def inst_decision_function(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict raw scores for the LAST INSTANCE of each bag.
        """
        return self.decision_function(X)

    def inst_predict_proba(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict logits for the LAST INSTANCE of each bag.
        """
        return self.predict_proba(X)

    def inst_predict(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict classes for the LAST INSTANCE of each bag.
        """
        return self.predict(X)

    def inst_conformal_prediction(self, X: Sequence[np.ndarray]) -> np.ndarray:
        """
        Predict conformal classes for the LAST INSTANCE of each bag.
        """
        return self.conformal_prediction(X)

    def load(self, output_dir: str | Path, layer_id: int) -> SupervisedPCA:
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
