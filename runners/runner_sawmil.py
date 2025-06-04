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

    def _preprocess_labels(self, y, mask):
        """
        Turn {0,1} labels into {−1,+1} and apply mask.
        """
        arr = y.copy()
        arr[arr == 0] = -1
        return arr[mask]

    def single_training(self, X, y, mask):
        """
        Wraps the existing single_training(...) function from your script.
        Args:
            - X: list of bags (each is array-like of shape [ #instances × hidden_size ])
            - y: full array of bag-labels (0/1)
            - layer_id: which layer we’re processing (used for randomness in drop_rows_with_tail_keep)
            - mask: boolean mask array of length len(X) indicating which bags to train on
        Returns a dict with keys:
            'separator', 'scaler', 'transformer', 'eta'
        """
        mask = np.array(mask, dtype=bool)
        X = deepcopy(X)
        y = deepcopy(y)

        # =========== Step 1: label preprocessing ===========
        y_train = self._preprocess_labels(y, mask)

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
            - layer_id: integer
            - mask: boolean mask array
        Returns:
            same dict as single_training plus 'best_C'
        """
        # Copy inputs so we don’t modify them in‐place
        X_copy = deepcopy(X)
        y_copy = deepcopy(y)
        mask_copy = deepcopy(mask)

        # Convert all‐zero mask to boolean
        mask_bool = np.array(mask_copy, dtype=bool)

        # Pre‐process labels
        y_all = y_copy.copy()
        y_all[y_all == 0] = -1

        # Pull out param grid
        C_candidates = self.cfg.probe["param_grid"]["C"]
        n_splits = self.cfg.get("cv_n_folds", 3)
        seed = self.cfg.get("random_seed", 42)

        scores = []
        stds = []
        n_samples = len(X_copy)

        for idx, Cval in enumerate(C_candidates):
            log.warning(f"\tParam‐search: testing C = {Cval} ...")
            fold_scores = []

            kf = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=seed,
            )

            for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_copy)):
                # build fold-specific mask
                tr_mask = np.zeros(n_samples, dtype=bool)
                te_mask = np.zeros(n_samples, dtype=bool)
                tr_mask[tr_idx] = True
                te_mask[te_idx] = True
                tr_mask = tr_mask & mask_bool
                te_mask = te_mask & mask_bool

                # get fold data
                X_tr = [bag for bag, m in zip(X_copy, tr_mask) if m]
                X_te = [bag for bag, m in zip(X_copy, te_mask) if m]
                y_tr = y_all[tr_mask]
                y_te = y_all[te_mask]

                # step 1: normalize fold data
                if self.cfg.probe["normalize_data"]:
                    flat_tr = np.vstack(X_tr)
                    fold_scaler = StandardScaler()
                    fold_scaler.fit(flat_tr)
                    bags_tr = [fold_scaler.transform(bag) for bag in X_tr]
                    bags_te = [fold_scaler.transform(bag)[-1] for bag in X_te]
                else:
                    raise NotImplementedError("Only normalization is supported in param search")

                # step 2: cap bag size and assign intra labels
                pruned_bags = []
                intra_labels = []
                pos_count = self.cfg.probe["num_known_positives"]
                assume_known = self.cfg.probe.get("assume_known_positives", True)
                max_size = max(self.cfg.probe["max_bag_size"] - 5, 10)

                for i_bag, bag in enumerate(bags_tr):
                    n_inst = bag.shape[0]
                    if n_inst > max_size:
                        rng = seed + layer_id + fold_idx + i_bag
                        bag_cap = drop_rows_with_tail_keep(
                            bag, max_size, pos_count, rng
                        )
                    else:
                        bag_cap = bag
                    pruned_bags.append(bag_cap)

                    if assume_known:
                        lbls = [0] * (bag_cap.shape[0] - pos_count) + [1] * pos_count
                    else:
                        lbls = [1] * bag_cap.shape[0]
                    intra_labels.append(lbls)

                # compute eta for this fold
                lengths_pos = [len(b) for b, lbl in zip(pruned_bags, y_tr) if lbl == 1]
                eta_fold = (y_tr[y_tr == 1] * pos_count).sum() / sum(lengths_pos)
                y_tr[y_tr == 0] = -1
                y_te[y_te == 0] = -1

                # train MIL‐SVM
                try:
                    sep = sAwMIL(
                        C=float(Cval),
                        kernel=self.cfg.probe["init_params"]["kernel"],
                        penalty=self.cfg.probe["init_params"]["penalty"],
                        scale_C=self.cfg.probe["init_params"]["scale_C"],
                        verbose=False,
                        eta=eta_fold,
                    )
                    sep.fit(pruned_bags, y_tr, intra_labels)

                    direction, bias = sep.linearize(normalize=True)
                    bp = BagProcessor(
                        max_bag_size=self.cfg.probe["max_bag_size"],
                        pos_labels_in_bag=pos_count,
                        scaler=fold_scaler,
                    )
                    scores_te = bp.predict_scores(
                        bags=X_te, direction=direction, bias=bias
                    )
                    fold_scores.append(mAP(y_te, scores_te))
                    log.warning(f"\t\tFold {fold_idx} mAP: {fold_scores[-1]:.4f}")
                except Exception as e:
                    log.error(f"Fold error (C={Cval}): {e}")
                    fold_scores.append(0.0)

            scores.append(np.mean(fold_scores))
            stds.append(np.std(fold_scores))
            log.warning(f"\tAvg-mAP for C={Cval}: {scores[-1]:.4f}")

        scores_arr = np.array(scores)
        std_arr = np.array(stds) / np.sqrt(n_splits)
        best_idx = int(np.argmax(scores_arr))
        best_C = C_candidates[best_idx]
        best_se = std_arr[best_idx]
        best_score = scores_arr[best_idx]

        # 1-SE rule
        sel_index = None
        for i, sc in enumerate(scores_arr):
            if sc >= (best_score - best_se):
                sel_index = i
                break

        if sel_index is not None:
            selected_C = C_candidates[sel_index]
            selected_score = scores_arr[sel_index]
        else:
            selected_C = best_C
            selected_score = best_score
            log.warning("\tNo C within 1SE; using best C.")

        log.warning(f"\tBest C: {best_C} (mAP={best_score:.4f})")
        log.warning(f"\tSelected C (1SE): {selected_C} (mAP≈{selected_score:.4f})")

        # update cfg and retrain on full set
        self.cfg.probe["init_params"]["C"] = selected_C
        log.warning(f"\tRetraining on full set with C={selected_C} ...")
        final = self.single_training(X_copy, y_copy, layer_id, mask_copy)

        return {
            "separator": final["separator"],
            "scaler": final["scaler"],
            "transformer": final["transformer"],
            "eta": final["eta"],
            "best_C": selected_C,
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
        bag: np.array, bag
        Returns:
        output_bag: np.array, processed bag
        intra_bag_mask: np.array, mask for the positive labels in the bag
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
