from runners.runner_base import BaseProbeRunner
from probes.silSVM_patch import SVM
from probes.multiclass import MulticlassMIL
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from utils_hydra import drop_rows_with_tail_keep
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


class MILMC_Runner(BaseProbeRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.probe["name"] == "sawmil", "Probe should belong to MIL"
        self.cfg = cfg
        # set random seed
        np.random.seed(getattr(cfg.probe, 'seed', None))
        self.scaler = None
        self.calibrator = None
        self.separator = None
        self.transformer = None
        self.bag_processor = None

        self.tasks = cfg.multiclass_params.tasks
        reader_T = MILProbeData(output_dir=cfg.output_dir, task=self.tasks['T'], model_name=cfg.model["name"],
                                datapack=cfg.datapack['name'], trial_name=cfg.trial_name, probe_name=cfg.probe["name"])
        reader_F = MILProbeData(output_dir=cfg.output_dir, task=self.tasks['F'], model_name=cfg.model["name"],
                                datapack=cfg.datapack['name'], trial_name=cfg.trial_name, probe_name=cfg.probe["name"])
        reader_N = MILProbeData(output_dir=cfg.output_dir, task=self.tasks['N'], model_name=cfg.model["name"],
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
        self.separator = MulticlassMIL(readers=self.readers,
                                       max_bag_size=self.cfg.probe.max_bag_size,
                                       layer_id=kwargs.get('layer_id', np.nan))
        bags = [self.process_single_bag(bag)[0] for bag in X]
        self.separator.fit(bags, y)

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

    def parameter_search(self, X, y, mask, **kwargs):
        log.warning(
            "Parameter search is not implemented for MILMC_Runner. Use single_training instead.")
        return self.single_training(X, y, mask=mask, **kwargs)

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
        """
        # Transform the bags using the fitted scaler
        return self.separator.predict_proba(X)

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
    
    def decision_function(self, X: list):
        """
        Compute raw bag‐scores for a new set of bags, using the trained separator.
        """
        output = []
        if type(X) is np.ndarray:
            X = [X]
            
        for bag in X:
            bag, _ = self.process_single_bag(bag)
            scores = self.separator.predict_proba([bag])
            output.append(scores)
        return np.array(output)

    def update_metric(self, metric_dict):
        """
        Add the metric items to the metric dictionary.
        """
        metric_dict['C'] = self.separator.C
        metric_dict['kernel'] = self.separator.kernel
        metric_dict['scale_C'] = self.separator.scale_C
        return metric_dict

    def process_single_bag(self, bag: np.ndarray, max_bag_size: int =100, rnd_seed_offset: int =0):
        ''' 
        Process a single bag
        bag: np.array, shape [ #instances × hidden_size ]
        max_bag_size: int, maximum size of the bag after processing
        rnd_seed_offset: int, offset for the random seed (to ensure different random behavior for different bags)
        Returns:
        output_bag: np.array, processed bag
        intra_bag_mask: np.array, intra bag labes used by the `sAwMIL` probe (for training)
        '''
        num_last_tokens_to_keep = self.cfg.probe["num_known_positives"]
        assume_known = self.cfg.probe.get("assume_known_positives", True)
            
        bag_size = bag.shape[0]    
        # 1. Cap the bag size
        if bag_size >  max_bag_size:
            # drop FROM the tail but keep last `num_last_tokens_to_keep` items
            rng = self.cfg.random_seed + rnd_seed_offset
            bag_processed = drop_rows_with_tail_keep(
                bag, max_bag_size, num_last_tokens_to_keep, rng
            )
        else:
            bag_processed = bag
        # 2. Compute intra‐bag labels
        if assume_known:
            # last `num_last_tokens_to_keep` indices are “positive,” rest are “negative”
            intra_labels_for_this_bag = (
                [0] * (bag_processed.shape[0] - num_last_tokens_to_keep)
                + [1] * num_last_tokens_to_keep
                )
        else:
            intra_labels_for_this_bag = [1] * bag_processed.shape[0]
        return bag_processed, intra_labels_for_this_bag