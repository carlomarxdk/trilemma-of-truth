# misc/probe_data.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
import numpy as np
import pickle as pkl
import json
import re
import os



@dataclass(frozen=True)
class LayerParams:
    """
    Immutable container that loads existing data and metadata for the specific layer.
    - direction: coef vector
    - bias: scalar offset
    - scaler: optional sklearn-like transformer
    - calibrator: optional conformal/other calibrator object
    - model: optional pickled end-to-end model (for multiclass runs)
    - metrics_default / metrics_conformal: optional dicts
    - y_hat / y_true: optional arrays for stored predictions
    """
    direction: Optional[np.ndarray]
    bias: Optional[float]
    scaler: Optional[Any]
    calibrator: Optional[Any]
    model: Optional[Any]
    metrics_default: Optional[Dict]
    metrics_conformal: Optional[Dict]
    y_hat: Optional[np.ndarray]
    y_true: Optional[np.ndarray]
    
@dataclass(frozen=True)
class MulticlassLayerParams:
    """
    Immutable container that loads existing data and metadata for the specific layer (of the multiclass probe).
      - cls: pickled sklearn Pipeline (e.g., MulticlassProbe)
      - calibrator: optional MulticlassICP
      - metrics_default / metrics_conformal: optional dicts
      - y_hat: optional stored probabilities (N, C)
      - y_true: optional labels (N,)
    """
    cls: Optional[Any]
    calibrator: Optional[Any]
    metrics_default: Optional[Dict]
    metrics_conformal: Optional[Dict]
    y_hat: Optional[np.ndarray]
    y_true: Optional[np.ndarray]
    
    


class ProbeData:
    """
    Read-only, cached loader for info under a single probe directory.

    Expected files (some are optional):
      - coef_{L}.npy OR direction_{L}.npy
      - bias_{L}.npy
      - scaler_{L}.pkl          (optional)
      - calibrator_{L}.pkl OR cp_{L}.pkl  (optional)
      - model_{L}.pkl           (optional; e.g., saved multiclass sklearn Pipeline)
      - metrics_{L}.json        (optional; { "default": ..., "conformal": ... })
      - y_hat_{L}.npy           (optional)
      - y_true.npy              (optional; shared across layers)

    Notes:
      - Zero repeated disk I/O per layer thanks to @lru_cache.
      - No math here—just I/O. Use a separate projector/estimator for scoring.
    """

    def __init__(self, probe_dir: str):
        self.probe_dir = self._normalize_probe_dir(Path(probe_dir))
        if not self.probe_dir.exists():
            raise FileNotFoundError(f"Probe directory not found: {self.probe_dir}")

    # ---------- public API ----------

    def available_layers(self) -> List[int]:
        """Return sorted list of layer ids discovered in this directory."""
        # Prefer model_*.pkl if present (multiclass). Fall back to metrics_*.json.
        layers = set()
        for f in self.probe_dir.glob("model_*.pkl"):
            m = re.search(r"model_(\d+)", f.name)
            if m: layers.add(int(m.group(1)))
        for f in self.probe_dir.glob("metrics_*.json"):
            m = re.search(r"metrics_(\d+)", f.name)
            if m: layers.add(int(m.group(1)))
        # Also allow coef_/direction_ as a last resort:
        for f in self.probe_dir.glob("coef_*.npy"):
            m = re.search(r"coef_(\d+)", f.name)
            if m: layers.add(int(m.group(1)))
        for f in self.probe_dir.glob("direction_*.npy"):
            m = re.search(r"direction_(\d+)", f.name)
            if m: layers.add(int(m.group(1)))
        return sorted(layers)

    @lru_cache(maxsize=None)
    def load_layer(self, layer_id: int) -> LayerParams:
        """
        Load (and cache) all available artifacts for a layer.
        Missing optional pieces are returned as None.
        """
        # direction / coef
        direction = self._load_direction(layer_id)
        bias = self._load_bias(layer_id)

        # optional bits
        scaler = self._load_pickle_if_exists(f"scaler_{layer_id}.pkl")
        calibrator = (
            self._load_pickle_if_exists(f"calibrator_{layer_id}.pkl")
            or self._load_pickle_if_exists(f"cp_{layer_id}.pkl")
        )
        model = self._load_pickle_if_exists(f"model_{layer_id}.pkl")

        metrics_default, metrics_conformal = self._load_metrics(layer_id)
        y_hat = self._load_npy_if_exists(f"y_hat_{layer_id}.npy")
        y_true = self._load_npy_if_exists("y_true.npy")  # shared across layers

        return LayerParams(
            direction=direction,
            bias=bias,
            scaler=scaler,
            calibrator=calibrator,
            model=model,
            metrics_default=metrics_default,
            metrics_conformal=metrics_conformal,
            y_hat=y_hat,
            y_true=y_true,
        )

    # ---------- helpers ----------

    def _normalize_probe_dir(self, p: Path) -> Path:
        """Handle paths that accidentally include prefixes before 'outputs/'."""
        print(p)
        if p.exists():
            return p
        # try to truncate to 'outputs/...'
        s = str(p)
        m = re.search(r"(^|/)(outputs)(/|$)", s)
        if m:
            tail = s[m.start(2):]  # keep from 'outputs' onward
            q = Path(tail)
            if q.exists():
                return q
        return p  # will error in __init__ if still missing

    def _load_direction(self, layer_id: int) -> Optional[np.ndarray]:
        for name in (f"coef_{layer_id}.npy", f"direction_{layer_id}.npy"):
            path = self.probe_dir / name
            if path.exists():
                return np.load(path)
        # direction is optional for pure multiclass saved-model case
        return None

    def _load_bias(self, layer_id: int) -> Optional[float]:
        path = self.probe_dir / f"bias_{layer_id}.npy"
        if path.exists():
            return float(np.load(path))
        # bias is optional for pure multiclass saved-model case
        return None

    def _load_metrics(self, layer_id: int) -> tuple[Optional[Dict], Optional[Dict]]:
        path = self.probe_dir / f"metrics_{layer_id}.json"
        if not path.exists():
            return None, None
        with open(path, "rb") as f:
            m = json.load(f)
        return m.get("default"), m.get("conformal")

    def _load_pickle_if_exists(self, name: str) -> Optional[Any]:
        path = self.probe_dir / name
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pkl.load(f)

    def _load_npy_if_exists(self, name: str) -> Optional[np.ndarray]:
        path = self.probe_dir / name
        if path.exists():
            return np.load(path)
        return None
    


class MulticlassProbeData:
    """
    Reads multiclass probe data saved as:
      outputs/probes/{probe_name}/{model_name}/{datapack}{_search}_full/
        cls_{L}.pkl
        cp_{L}.pkl                 (optional)
        metrics_{L}.json           (optional; {'default':..., 'conformal':...})
        y_hat_{L}.npy              (optional)
        y_true.npy                 (optional)
    """

    def __init__(self, probe_dir: str):
        self.probe_dir = self._normalize_probe_dir(Path(probe_dir))
        if not self.probe_dir.exists():
            raise FileNotFoundError(f"Probe directory not found: {self.probe_dir}")

    # ------------------------ public API ------------------------

    def available_layers(self) -> List[int]:
        """Discover layers via cls_{L}.pkl (preferred), else metrics_{L}.json."""
        layers = []
        for f in self.probe_dir.glob("cls_*.pkl"):
            m = re.search(r"cls_(\d+)", f.name)
            if m: layers.append(int(m.group(1)))
        if not layers:
            for f in self.probe_dir.glob("metrics_*.json"):
                m = re.search(r"metrics_(\d+)", f.name)
                if m: layers.append(int(m.group(1)))
        return sorted(set(layers))

    def metadata(self, layer_id: int) -> Dict[str, Any]:
        """Backward-compatible accessor (matches your old class)."""
        art = self.load_layer(layer_id)
        return {
            "cls": art.cls,
            "calibrator": art.calibrator,
            "y_hat": art.y_hat,
            "y_true": art.y_true,
            "metrics_default": art.metrics_default,
            "metrics_conformal": art.metrics_conformal,
        }

    @lru_cache(maxsize=None)
    def load_layer(self, layer_id: int) -> MulticlassLayerParams:
        cls = self._load_pickle(f"cls_{layer_id}.pkl", required=False)
        calibrator = (
            self._load_pickle(f"cp_{layer_id}.pkl", required=False)
            or self._load_pickle(f"calibrator_{layer_id}.pkl", required=False)
        )
        md, mc = self._load_metrics(layer_id)
        y_hat = self._load_npy(f"y_hat_{layer_id}.npy", required=False)
        y_true = self._load_npy("y_true.npy", required=False)

        return MulticlassLayerParams(
            cls=cls,
            calibrator=calibrator,
            metrics_default=md,
            metrics_conformal=mc,
            y_hat=y_hat,
            y_true=y_true,
        )

    # ----- convenience: inference using saved model/calibrator -----

    def cls(self, layer_id: int):
        return self.load_layer(layer_id).cls

    def calibrator(self, layer_id: int):
        return self.load_layer(layer_id).calibrator

    def predict_proba(self, layer_id: int, bags: List[np.ndarray]) -> np.ndarray:
        art = self.load_layer(layer_id)
        if art.cls is None:
            raise RuntimeError(f"cls_{layer_id}.pkl not found in {self.probe_dir}")
        return art.cls.predict_proba(bags)

    def predict(self, layer_id: int, bags: List[np.ndarray]) -> np.ndarray:
        art = self.load_layer(layer_id)
        if art.cls is None:
            raise RuntimeError(f"cls_{layer_id}.pkl not found in {self.probe_dir}")
        return art.cls.predict(bags)

    def predict_sets(self, layer_id: int, bags: List[np.ndarray]) -> np.ndarray:
        """
        If a calibrator exists, return set-valued predictions; otherwise singleton sets via argmax.
        """
        art = self.load_layer(layer_id)
        probs = self.predict_proba(layer_id, bags)
        if art.calibrator is not None:
            return art.calibrator.predict(probs)
        argmax = probs.argmax(axis=1)
        out = np.zeros_like(probs, dtype=bool)
        out[np.arange(len(argmax)), argmax] = True
        return out

    # ----- compatibility helpers (names from your old class) -----

    def return_true(self, layer_id: int) -> Optional[np.ndarray]:
        return self.load_layer(layer_id).y_true

    def return_pred(self, layer_id: int) -> Optional[np.ndarray]:
        return self.load_layer(layer_id).y_hat

    def return_preds(self, layer_id: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        art = self.load_layer(layer_id)
        return art.y_hat, art.y_true

    def metrics_conformal(self, layer_id: int) -> Optional[Dict]:
        return self.load_layer(layer_id).metrics_conformal

    def metrics_default(self, layer_id: int) -> Optional[Dict]:
        return self.load_layer(layer_id).metrics_default

    def best_layer(self, metric: str = "map", per_bag: bool = False) -> Optional[int]:
        """
        Select best layer by metric name.
        For legacy compat:
          - if per_bag=True and metrics_conformal has a '_bag' dict, use it.
          - metric values may be scalars or [val, lo, hi]; we take the first element.
        """
        layers = self.available_layers()
        if not layers:
            return None
        best_layer, best_score = None, -float("inf")
        for L in layers:
            m = self.metrics_conformal(L) or {}
            if per_bag and isinstance(m.get("_bag"), dict):
                m = m["_bag"]
            val = m.get(metric)
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            if val is not None and val > best_score:
                best_score = val
                best_layer = L
        return best_layer

    def top_k_layers(self, metric: str = "map", k: int = 5) -> List[int]:
        layers = self.available_layers()
        scored: List[Tuple[int, float]] = []
        for L in layers:
            m = self.metrics_conformal(L) or {}
            val = m.get(metric)
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            if val is not None:
                scored.append((L, float(val)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [L for (L, _) in scored[:k]]

    def return_general_metric(self, layer_id: int, datapack_name: str) -> Dict[str, Dict]:
        """
        Reads generalization metrics from:
          <probe_dir_prefix>-to-{datapack_name}/metrics_{L}.json
        where probe_dir_prefix is self.probe_dir without trailing '..._full/'.
        """
        prefix = str(self.probe_dir)
        if prefix.endswith("/"):
            prefix = prefix[:-1]
        base = prefix.rsplit("_full", 1)[0]
        out_dir = f"{base}-to-{datapack_name}/"
        path = Path(out_dir) / f"metrics_{layer_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing generalization metrics: {path}")
        with open(path, "rb") as f:
            m = json.load(f)
        return {"metrics_default": m.get("default"), "metrics_conformal": m.get("conformal")}

    def best_layer_generalization(
        self,
        datapack_name: str,
        metric: str = "map",
        metric_type: str = "conformal",
    ) -> Optional[int]:
        layers = self.available_layers()
        best_layer, best_score = None, -float("inf")
        for L in layers:
            mm = self.return_general_metric(L, datapack_name)
            m = mm.get(f"metrics_{metric_type}", {})
            val = m.get(metric)
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            if val is not None and val > best_score:
                best_score = val
                best_layer = L
        return best_layer

    # ------------------------ private helpers ------------------------
    
    
    def _normalize_probe_dir(self, p: Path) -> Path:
        """Handle paths that accidentally include prefixes before 'outputs/'."""
        print(p)
        if p.exists():
            return p
        # try to truncate to 'outputs/...'
        s = str(p)
        m = re.search(r"(^|/)(outputs)(/|$)", s)
        if m:
            tail = s[m.start(2):]  # keep from 'outputs' onward
            q = Path(tail)
            if q.exists():
                return q
        return p  # will error in __init__ if still missing

    def _load_pickle(self, name: str, required: bool) -> Any:
        path = self.probe_dir / name
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing required file: {path}")
            return None
        with open(path, "rb") as f:
            return pkl.load(f)

    def _load_npy(self, name: str, required: bool) -> Optional[np.ndarray]:
        path = self.probe_dir / name
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Missing required file: {path}")
            return None
        return np.load(path)

    def _load_metrics(self, layer_id: int) -> Tuple[Optional[Dict], Optional[Dict]]:
        path = self.probe_dir / f"metrics_{layer_id}.json"
        if not path.exists():
            return None, None
        with open(path, "rb") as f:
            m = json.load(f)
        return m.get("default"), m.get("conformal")