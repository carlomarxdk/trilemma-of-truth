from runners.runner_base import BaseProbeRunner
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

log = logging.getLogger("SawmilProbeRunner")


class SawmilProbeRunner(BaseProbeRunner):
    """
    Sparse Aware Multiple Instance Learning (MIL) SVM (sAwMIL)
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        np.random.seed(getattr(cfg, "random_seed", 42))

        # placeholders to be filled by training
        self.separator = None
        self.scaler = None
        self.transformer = None
        self.eta = None
        self.calibrator = None

    def return_targets(self, y, mask=None):
        """
        Turn {0,1} labels into {−1,+1} and apply mask.
        """
        arr = np.ones_like(y)
        arr[y == 0] = -1
        if mask is not None:
            return arr[mask]
        return arr

    def single_training(self, X, y, mask):
        """
        Wraps the existing single_training(...) function from your script.
        Args:
            - X: list of bags (each is array-like of shape [ #instances × hidden_size ])
            - y: full array of bag-labels (0/1)
            - mask: boolean mask array of length len(X) indicating which bags to train on
        Returns a dict with keys:
            'separator', 'scaler', 'transformer', 'eta'
        """
        mask = np.array(mask, dtype=bool)
        X = deepcopy(X)
        y = deepcopy(y)

        # =========== Step 1: label preprocessing ===========
        y_train = self.return_targets(y, mask)

        # =========== Step 2: normalization ===========
        if self.cfg.probe.get("normalize_data", True):
            # vstack all bags (including those we’ll later sparsify)
            X_all = np.vstack([bag for bag, m in zip(X, mask) if m])
            self.scaler = StandardScaler()
            self.scaler.fit(X_all)
            # transform each bag
            bags = [self.scaler.transform(bag) for bag, m in zip(X, mask) if m]
        else:
            raise NotImplementedError("Only normalization pipeline is implemented")

        # =========== Step 3: sparsification (optional) ===========
        self.transformer = None

        # =========== Step 4: cap bag size and assign intra‐bag labels ===========
        processed_bags = []
        intra_labels = []
        max_bag_size = self.cfg.probe["max_bag_size"]
        for i, bag in enumerate(bags):
            bag_processed, intra_labels_for_this_bag = self.process_single_bag(
                bag, max_bag_size=max_bag_size, rnd_seed_offset=i
            )
            processed_bags.append(bag_processed)
            intra_labels.append(intra_labels_for_this_bag)
            
        # =========== Step 5: compute η (eta) ===========
        pos_lengths = [
            len(bag) for bag in processed_bags
        ]
        eta = sum([sum(lbl) for lbl in intra_labels]) / sum(pos_lengths)
        self.eta = eta

        # =========== Step 6: fit sAwMIL ===========
        separator = sAwMIL(
            C=self.cfg.probe["init_params"]["C"],
            kernel=self.cfg.probe["init_params"]["kernel"],
            scale_C=self.cfg.probe["init_params"]["scale_C"],
            verbose=self.cfg.probe["init_params"]["verbose"],
            eta=eta,
        )
        limit = self.cfg.probe.get("train_bag_limit", len(processed_bags))
        log.warning(
            f"\tFitting sbMIL [C={self.cfg.probe['init_params']['C']}, eta={eta:.2f}] on {len(processed_bags)} bags..."
        )
        separator.fit(
            processed_bags[:limit],
            y_train[:limit],
            intra_labels[:limit],
        )
        self.separator = separator

        return {
            "separator": separator,
            "scaler": self.scaler,
            "transformer": self.transformer,
            "eta": eta,
        }

    def parameter_search(self, X, y, mask):
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
        param_grid = self.cfg.probe.param_grid['C'] # You can redefine this based on your needs
        f_X = deepcopy(X)
        f_y = deepcopy(y)
        f_mask = deepcopy(mask)
        # 0) Get the bags and labels

        assert len(X) == len(y), "X and y must have the same length"
        assert np.unique(y).size == 2, "y must be binary"
        # Convert all‐zero mask to boolean
        mask = np.array(f_mask, dtype=bool)
        y = self.return_targets(f_y, None)
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
                
                # =========== Step 2: normalization ===========
                if self.cfg.probe.get("normalize_data", True):
                    log.warning("\t\tNormalizing the data...")
                    Xt = np.vstack([bag for bag, m in zip(X, mask) if m])
                    scaler = StandardScaler()
                    scaler.fit(Xt)
                    # transform each bag
                    bags = [scaler.transform(bag) for bag in X_train]
                else:
                    raise NotImplementedError("Only normalization pipeline is implemented")

                limit = self.cfg.get('cv_bag_limit', len(bags))
                # =========== Step 3: sparsification (optional) ===========
                self.transformer = None

                # =========== Step 4: cap bag size and assign intra‐bag labels ===========
                processed_bags = []
                intra_labels = []
                max_bag_size = self.cfg.probe["max_bag_size"]
                for i, bag in enumerate(bags):
                    bag_processed, intra_labels_for_this_bag = self.process_single_bag(
                        bag, max_bag_size=max_bag_size, rnd_seed_offset=i
                    )
                    processed_bags.append(bag_processed)
                    intra_labels.append(intra_labels_for_this_bag)
                    
                # =========== Step 5: compute η (eta) ===========
                pos_lengths = [
                    len(bag) for bag in processed_bags
                ]
                eta = sum([sum(lbl) for lbl in intra_labels]) / sum(pos_lengths)
                self.eta = eta          
                # try:
                if True:
                    separator = sAwMIL(
                        C=float(C),
                        kernel=self.cfg.probe.get('kernel', 'linear'),
                        scale_C=self.cfg.probe.get('scale_C', True),
                        verbose=False,
                        eta=self.eta,
                    )
                    separator.fit(
                        bags = processed_bags[:limit],
                        y = y_train[:limit],
                        in_bag_labels = intra_labels[:limit],
                    )
                    direction, bias = separator.linearize(normalize=True)             
                    y_hat = self._decision_function_(X_test, direction = direction, bias = bias, scaler=scaler)
                    _inner_scores.append(mAP(y_test, y_hat))
                    log.warning(
                        f"\t\tmAP for {j}th fold: {_inner_scores[-1]}")
                # except Exception as e:
                #     log.error(f"Error: {e}")
                #     log.warning(
                #         "\t\tMoving to the next one...")
                #     _inner_scores.append(0.1)
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
        """
        Train the InductiveConformalPredictor on the calibration split.
        """
        X = deepcopy(X_cal)
        y = deepcopy(y_cal)
        mask = deepcopy(mask_cal)
        mask_cal = np.array(mask, dtype=bool)
        # keep original {0,1} labels—Conformal predictor takes raw scores + true labels
        config = self.cfg.conformal_params

        if config["nc"] == "binary":
            nc_func = symmetric_nonconformity
        else:
            raise NotImplementedError(f"NC {config['nc']} not implemented")

        # compute “scores” via current separator
        scores_cal = self.decision_function(X)
        self.calibrator = InductiveConformalPredictor(
            nonconformity_func=nc_func,
            alpha=config["alpha"],
            tie_breaking=config["tie_breaking"],
        )
        self.calibrator.fit(
            y=y[mask_cal], 
            scores=scores_cal[mask_cal]
        )

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

    def decision_function(self, X: list):
        """
        Compute raw bag‐scores for a new set of bags, using the trained separator.
        """
        output = []
        if type(X) is np.ndarray:
            X = [X]
            
        for bag in X:
            bag = self.scaler.transform(bag)
            bag, _ = self.process_single_bag(bag)
            scores = np.dot(bag, self.direction) + self.bias
            output.append(np.max(scores))
        return np.array(output)
    
    def _decision_function_(self, X: list, direction, bias, scaler):
        """
        Compute raw bag‐scores for a new set of bags, using the trained separator.
        This is a helper function to avoid code duplication.
        """
        output = []
        if type(X) is np.ndarray:
            X = [X]
            
        for bag in X:
            bag = scaler.transform(bag)
            bag, _ = self.process_single_bag(bag)
            scores = np.dot(bag, direction) + bias
            output.append(np.max(scores))
        return np.array(output)


    def process_input(self, X):
        raise NotImplementedError(
            "SawmilProbeRunner does not support process_input. Use decision_function instead."
        )
        
    def update_metric(self, metric_dict):
        """
        Add probe‐specific hyperparameters to metrics (e.g., C, eta).
        """
        metric_dict["C"] = self.separator.C
        metric_dict["eta"] = self.eta
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
