import numpy as np
from copy import deepcopy
import logging

from misc.probe_data import ProbeData 
from probes.multiclass import OvAProjector  # bags -> scores
from probes.multiclass import MulticlassProbe  # sklearn Pipeline

# your existing multiclass conformal bits
from probes.conformal import MulticlassICP, probability_margin_nc

log = logging.getLogger("MulticlassMILRunner")

class MulticlassMILRunner:
    """
    Drop-in sibling to SawmilProbeRunner, but for multiclass using OvA readers.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.separator = None         # sklearn Pipeline (MulticlassProbe)
        self.calibrator = None
        self.layer_id = None

        # hyperparams
        self.pool = "max"
        self.max_iter = 2000

        #  dict like {1: 'T', 0: 'F', 2: 'N'}
        self.tasks = cfg.multiclass_params.tasks

    # ---- path helpers (mirror your MILProbeData layout) ----
    def _probe_dir_for(self, task: str) -> str:
        """
        Build per-class *binary* reader dirs produced by your OvA runs.
        Matches your previous MILProbeData path scheme.
        """
        probe_name = self.cfg.probe.get("name_ova", "sawmil")  # OvA probe name
        model_name = self.cfg.model["name"]
        trial_name = self.cfg.trial_name.split('-')[0]  # keep base trial part before '-task'
        # for OvA we had per-task suffix (e.g., .../{trial}-{task}/)
        return f"outputs/probes/{probe_name}/{model_name}/{trial_name}-{task}/"

    def _build_readers(self) -> dict[int, ProbeData]:
        # class id mapping: 0=F, 1=T, 2=N (as you used before)
        paths = {
            0: self._probe_dir_for(self.tasks['F']),
            1: self._probe_dir_for(self.tasks['T']),
            2: self._probe_dir_for(self.tasks['N']),
        }
        readers = {k: ProbeData(v) for k, v in paths.items()}
        return readers

    # ---- runner API (shape compatible with your run_experiment.py) ----
    def single_training(self, X, y, mask, layer_id=None, **kwargs):
        """
        X: list of bags
        y: int labels in {0,1,2}
        mask: boolean vector
        """
        self.layer_id = self.layer_id if layer_id is None else layer_id
        if self.layer_id is None:
            # if caller didn’t pass it, try cfg (you’ll set cfg.layer_id in the loop)
            self.layer_id = self.cfg.get("layer_id", 0)

        X = deepcopy(X)
        y = np.asarray(y)
        mask = np.asarray(mask, dtype=bool)

        readers = self._build_readers()                       # {0,1,2} -> ProbeData
        projector = OvAProjector(readers, layer_id=self.layer_id)
        self.separator = MulticlassProbe(projector, pool=self.pool, max_iter=self.max_iter)

        bags = [b for b, m in zip(X, mask) if m]
        targets = y[mask]
        log.warning(f"Fitting MulticlassProbe on {len(bags)} bags (layer {self.layer_id}, pool={self.pool})")
        self.separator.fit(bags, targets)

        return {"separator": self.separator, "scaler": None, "transformer": None}

    def parameter_search(self, X, y, mask, **kwargs):
        log.warning("Parameter search not implemented for MulticlassMILRunner; doing single_training.")
        return self.single_training(X, y, mask, **kwargs)

    def predict_proba(self, X):
        return self.separator.predict_proba(X)  # (N, C)

    def predict(self, X):
        return self.separator.predict(X)        # (N,)

    def conformal_training(self, X_cal, y_cal, mask_cal):
        probs_cal = self.predict_proba([b for b, m in zip(X_cal, mask_cal) if m])
        y_c = np.asarray(y_cal)[mask_cal]
        self.calibrator = MulticlassICP(
            nonconformity_func=probability_margin_nc,
            alpha=self.cfg.conformal_params["alpha"],
            n_classes=3,
            tie_breaking=self.cfg.conformal_params["tie_breaking"]
        )
        self.calibrator.fit(y=y_c, scores=probs_cal)
        return self.calibrator

    def conformal_prediction(self, X):
        probs = self.predict_proba(X)
        return self.calibrator.predict(probs)   # set-valued; for metrics you can argmax

    def update_metric(self, metric_dict):
        return metric_dict
