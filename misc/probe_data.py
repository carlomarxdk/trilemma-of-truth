# misc/probe_data.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Any, Dict, List
import numpy as np
import pickle as pkl
import json
import re



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


class ProbeData:
    """
    Read-only, cached loader for artifacts under a single probe directory.

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