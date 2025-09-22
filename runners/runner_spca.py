from runners.base import BaseProbeRunner
from probes.spca import SupervisedPCA
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
from copy import deepcopy

log = logging.getLogger("SILRunner-SPCA")


class SPCA_Runner(BaseProbeRunner):
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
        f_mask = np.array(mask, dtype=bool)
        f_y = deepcopy(y)
        f_X = deepcopy(X)
        assert len(f_X) == len(f_y) == len(f_mask), "X, y and mask must have the same length"
        assert np.unique(f_y).size == 2, "y must be binary"

        ym = self.return_target(f_y, f_mask)
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

        self.separator = SupervisedPCA(n_components=cfg.init_params['n_components'],
                                        cov_type=cfg.init_params.get('cov_type', 'oas'),
                                        cov_fallback_scale=cfg.init_params.get('cov_fallback_scale', 1.0),
                                        verbose=cfg.init_params.get('verbose', False),
                                        random_seed=cfg.init_params.get('random_seed', 42))

        self.separator.fit(
            Xm[:limit], ym[:limit])

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
        params = self.cfg.probe.param_grid
        param_grid = {f'clf__{k}': v for k, v in params.items() if v is not None}
        f_mask = np.array(mask, dtype=bool)
        f_y = deepcopy(y)
        f_X = deepcopy(X)
        assert len(f_X) == len(f_y) == len(f_mask), "X, y and mask must have the same length"
        assert np.unique(f_y).size == 2, "y must be binary"

        Xm = np.vstack([x[-1] for x, m in zip(f_X, f_mask) if m])
        ym = self.return_target(f_y, f_mask)
        

        pipeline = Pipeline([("scaler", StandardScaler()),
                             ("clf", SupervisedPCA())])
        
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=KFold(n_splits=3, shuffle=True, random_state=42),
            refit=False,
            scoring='average_precision',
            n_jobs=-1,
            verbose=1,
            error_score=0.0
        )

        grid.fit(Xm, ym)
        best_params, _ = self._apply_se_rule(grid.cv_results_, n_folds=3)
        self.cfg.probe.init_params['n_components'] = best_params['clf__n_components']
        return self.single_training(X, y, mask)
    
    def _apply_se_rule(self, results, n_folds: int = 3):
        means = results['mean_test_score'] 
        stds = results['std_test_score'] 
        params = results['params']
        
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
        log.warning(f"\tSelected via 1-SE: n_components={params[selected_idx]['clf__n_components']} (mean AP={selected_score:.4f})")
        return params[selected_idx], selected_score
        
        
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
    
    def conformal_acc_rate(self, X):
        """
        Compute the conformal prediction acceptance rate for the given bags.
        """
        # Transform the bags using the fitted scaler
        f_X = deepcopy(X)
        # Compute the decision function using the separator
        yh = self.decision_function(f_X)
        # Compute the conformal prediction acceptance rate
        return self.calibrator.acceptance_rate(yh)

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
        return None

    @property
    def direction_bias(self):
        """
        Return, BOTH, the direction and bias of the separator.
        """
        return None, None

    @property
    def classifier(self):
        """
        Return the classifier of the separator.
        """
        return self.separator