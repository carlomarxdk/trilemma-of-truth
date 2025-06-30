from probes.runner_base import BaseProbeRunner
from probes.silSVM_patch import SVM
from probes.multiclass import MulticlassSIL
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from probes.conformal import MulticlassICP, probability_margin_nc
from sklearn.metrics import (
    average_precision_score as mAP,
    matthews_corrcoef as mcc,
)
import numpy as np
import logging
from copy import deepcopy
from misc.reader import MILProbeData

log = logging.getLogger("SILMC_Runner")


class SILMC_Runner(BaseProbeRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.probe["name"] == "SVM", "Probe name must be SIL"
        self.cfg = cfg
        # set random seed
        np.random.seed(getattr(cfg.probe, 'seed', None))
        self.scaler = None
        self.calibrator = None
        self.separator = None
        self.transformer = None
        self.bag_processor = None

        self.task_dict = cfg.task_dict
        reader_T = MILProbeData(output_dir=cfg.output_dir, task=self.task_dict[0], model_name=cfg.model["name"],
                                datapack=cfg.datapack['name'], trial_name=cfg.trial_name, probe_name=cfg.probe["name"])
        reader_F = MILProbeData(output_dir=cfg.output_dir, task=self.task_dict[1], model_name=cfg.model["name"],
                                datapack=cfg.datapack['name'], trial_name=cfg.trial_name, probe_name=cfg.probe["name"])
        reader_N = MILProbeData(output_dir=cfg.output_dir, task=self.task_dict[2], model_name=cfg.model["name"],
                                datapack=cfg.datapack['name'], trial_name=cfg.trial_name, probe_name=cfg.probe["name"])
        self.readers = {
            0: reader_F,
            1: reader_T,
            2: reader_N
        }

    def single_training(self, X, y, mask=None, **kwargs):
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
        self.separator = MulticlassSIL(readers=self.readers,
                                       max_bag_size=1,
                                       layer_id=kwargs.get('layer_id', np.nan))

        self.separator.fit(X, y)

        return {'separator': self.separator,
                'scaler': np.nan,
                'transformer': np.nan}

    def setup_from_pretrained(self, separator, scaler=None, transformer=None, calibrator=None):
        """
        Setup the runner from a pretrained separator.
        Args:
            - separator: a pretrained separator
            - scaler: a pretrained scaler
            - transformer: a pretrained transformer
        """
        self.separator = separator
        self.scaler = scaler
        self.transformer = transformer
        self.calibrator = calibrator

    def return_target(self, y, mask=None):
        return y

    def parameter_search(self, X, y, mask):
        raise NotImplementedError(
            "Parameter search is not implemented for SILMC_Runner. Use single_training instead.")

    def conformal_training(self, X_cal, y_cal, mask_cal=None):
        # CONFORMAL PREDICTION
        X = deepcopy(X_cal)
        y = deepcopy(y_cal)
        probs_cal = self.single_prediction(X)

        self.calibrator = MulticlassICP(nonconformity_func=probability_margin_nc,
                                        alpha=self.cfg.conformal_params["alpha"],
                                        n_classes=3,
                                        tie_breaking=self.cfg.conformal_params["tie_breaking"])
        self.calibrator.fit(y=y, scores=probs_cal)
        return self.calibrator

    def conformal_prediction(self, X, agg="last"):
        """
        Compute the conformal prediction for the given bags.
        Args:
            - X: a list of bags
            - agg: aggregation method, either 'last' or 'all' (all is based on the all tokens in the bag)
        """
        assert agg in ["last", "all"], "agg must be 'last' or 'all'"
        # Transform the bags using the fitted scaler
        X = deepcopy(X)
        if agg == "last":
            probs = self.single_prediction(X)
        elif agg == 'all':
            probs = self.bag_prediction(X)
        return self.calibrator.predict(probs)

    def _decision_function(self, X, per_instance=False):
        """
        Compute the decision function for the given bags.
        Args:
            - X: a list of bags
            - as_bag: if False, makes prediction based on the last token
        """
        # Transform the bags using the fitted scaler
        return self.separator.predict_proba(X, per_instance=per_instance)

    def single_prediction(self, X, per_instance=False):
        '''
        Decision function based on the last token only.
        Args:
            - X: a list of bags
        '''
        return self._decision_function(X, per_instance=per_instance)

    def bag_prediction(self, X):
        '''
        Decision function that aggregates the scores for each bag.
        Args:
            - X: a list of bags
        '''
        output = []
        bag_probs = self._decision_function(X, per_instance=True)
        # remove the first item in the bag, [SOS]/[CLS] token
        for probs in bag_probs:
            X = probs[1:]
            flat_idx = np.argmax(X)
            row_idx, _ = np.unravel_index(flat_idx, X.shape)
            output.append(X[row_idx])
        return np.array(output)

    def update_metric(self, metric_dict):
        """
        Add the metric items to the metric dictionary.
        """
        metric_dict['C'] = self.separator.C
        metric_dict['kernel'] = self.separator.kernel
        metric_dict['scale_C'] = self.separator.scale_C
        return metric_dict
