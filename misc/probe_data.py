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
import importlib
from pathlib import Path
import json
from typing import Any, Iterable, List, Optional, Union
import sys
from packaging import version





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
    
    
class ExperimentData:
    '''
    Class to manage and access experimental data for different models, probes, datasets, and tasks.
    '''
    def __init__(self, 
                 model_name:str, 
                 probe_name:str, 
                 dataset_name:str, 
                 task: int, 
                 with_search:bool = True):
        '''
        Initialize the ExperimentData object with the specified parameters.
        Args:
            model_name (str): The name of the model.
            probe_name (str): The name of the probe.
            dataset_name (str): The name of the dataset.
            task (int): The task number.
            with_search (bool): Whether to include search in the path construction.
        '''
        self.model_name = model_name
        self.probe_name = probe_name
        self.dataset_name = dataset_name
        self.task = task
        self.with_search = with_search
        
    @property
    def base_path(self) -> Path:
        '''
        Construct the base path for the experiment data based on the provided parameters.
        Returns:
            Path: The constructed base path.
        '''
        if self.with_search:
            return Path('outputs') / 'probes' / self.probe_name / self.model_name / f'{self.dataset_name}_search_task-{str(self.task)}'
        else:
            return Path('outputs') / 'probes' / self.probe_name / self.model_name / f'{self.dataset_name}_task-{str(self.task)}'
        
    @property
    def absolute_path(self) -> Path:
        return self.base_path.absolute()
    
    def layer_exists(self, layer:int) -> bool:
        '''Check if the manifest file for a given layer exists.
        Args:
            layer (int): The layer number to check.
        Returns:
            bool: True if the manifest file exists, False otherwise.
        '''
        path = self.base_path / 'manifests' / f'manifest_{str(layer)}.json'
        return path.exists()
    
    def load_manifest(self, layer:int) -> dict:
        '''Load the manifest file for a given layer.
        Args:
            layer (int): The layer number to load the manifest for.
        Returns:
            dict: The content of the manifest file as a dictionary.
        '''
        assert self.layer_exists(layer), f"The experimental data for layer {layer} does not exist. Available layers: {self.available_layers}"
        with open(self.base_path / 'manifests' / f'manifest_{str(layer)}.json', 'r') as f:
            manifest = json.load(f)
        return manifest
    
    @property
    def available_layers(self) -> List[int]:
        '''Get a list of available layers for which manifest files exist.
        Returns:
            list[int]: A list of layer numbers that have manifest files.
        '''
        manifests_path = self.base_path / 'manifests'
        if not manifests_path.exists():
            return []
        
        layer_files = manifests_path.glob('manifest_*.json')
        layers = [int(f.stem.split('_')[1]) for f in layer_files if f.stem.split('_')[1].isdigit()]
        return sorted(layers)
    
    def available_subexperiments(self) -> List[str]:
        '''Get a list of available statistics for which stats files exist.
        Returns:
            list[str]: A list of statistic names that have stats files.
        '''
        folders = [x for x in self.base_path.iterdir() if x.is_dir()]
        folders = [x.name for x in folders if x.name.startswith('g_')]
        return sorted(folders)

    def best_layer(self, keys: List[str] = ['conformal', 'wmcc'], path: Optional[Path] = None) -> int:
        '''Determine the best layer based on a specified metric.
        Args:
            keys (list of str): The key or list of keys to navigate through the JSON structure to find the metric.
            path (Path, optional): The base path to the experiment data. If None, uses the default base path.
        Returns:
            int: The layer number that has the highest value for the specified metric.
        '''
        results = {}
        for layer in self.available_layers:
            metric = self.read_metrics(layer, keys, path=path)
            results[layer] = metric[0]
        return max(results, key=results.get)
    
    def read_metrics(self, layer_id: int,         
                keys: Optional[Union[str, Iterable[str]]] = None,
                default: Any = ...,
                path: Optional[Path] = None) -> Any:
        '''Read the metrics from the JSON file for a specific layer.
        Args:
            layer_id (int): The layer number to read metrics for.
            keys (str or list of str, optional): The key or list of keys to navigate through the JSON structure. If None, returns the entire JSON content.
            default (any, optional): The default value to return if the specified key path does not exist. If not provided and the key path is not found, a KeyError is raised.
            path (Path, optional): The base path to the experiment data. If None, uses the default base path.
        Returns:
            any: The value corresponding to the specified key path, or the entire JSON content if keys is None.
        Raises:
            KeyError: If the specified key path does not exist and no default value is provided.
            AssertionError: If the manifest file for the specified layer does not exist.
        '''
        assert self.layer_exists(layer_id), f"The experimental data for layer {layer_id} does not exist. Available layers: {self.available_layers()}"
        
        
        if path is not None:
            assert path.exists(), f"The provided path {str(path)} does not exist."
            path = Path(path) 
        else:
            path = self.base_path  
        with open(path / 'manifests' / f'manifest_{str(layer_id)}.json', 'r') as f:
            metric_path = json.load(f)['paths']['metrics']
            
        with open(metric_path, 'r') as f:
            data = json.load(f)
        if keys is None:
            return data

        # Normalize keys to a list
        if isinstance(keys, str):
            keys = [keys]
        else:
            keys = list(keys)

        cur = data
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                if default is ...:
                    path_so_far = "/".join(keys)
                    raise KeyError(
                        f"Key path '{path_so_far}' not found (stopped at '{k}')."
                    )
                return default
        return cur
    
    def validate_manifest(self, layer_id: int, path: Optional[Path] = None) -> bool:
        '''Validate the manifest file for a specific layer.
        Args:
            layer_id (int): The layer number to validate the manifest for.
            path (Path, optional): The base path to the experiment data. If None, uses the default base path.
        Returns:
            bool: True if the manifest file is valid, False otherwise.  
        '''
        assert self.layer_exists(layer_id), f"The experimental data for layer {layer_id} does not exist. Available layers: {self.available_layers()}"
        manifest = self.load_manifest(layer_id)['env']
        not_installed = []
        not_matching = []
        for pkg, required_version in manifest.items():
            if pkg == 'python':
                v = ".".join(map(str, sys.version_info[:3]))
                if version.parse(v) == version.parse(required_version):
                    print(f'python: {v} (matches manifest)')
                elif version.parse(v) != version.parse(required_version):
                    print(f"python: installed {v}, but manifest requires {required_version}")
                    not_matching.append(pkg)
                continue
                
            try:
                # dynamically import the module
                mod = importlib.import_module(pkg)
                installed_version = getattr(mod, "__version__", None)
            except ModuleNotFoundError:
                print(f"{pkg}: not installed (expected {required_version})")
                not_installed.append(pkg)
                continue
            if version.parse(installed_version) == version.parse(required_version):
                print(f'{pkg}: installed {installed_version} matches manifest')
            elif version.parse(installed_version) != version.parse(required_version):
                print(f"{pkg}: installed {installed_version}, but manifest requires {required_version}")
                not_matching.append(pkg)
        if not_installed or not_matching:
            return False
        return True

    def read_predictions(self, layer_id: int, path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
        '''Read the predictions and true labels from the files for a specific layer.
        Args:
            layer_id (int): The layer number to read predictions for.
            path (Path, optional): The base path to the experiment data. If None, uses the default base path.
        Returns:
            tuple: A tuple containing two numpy arrays: (y_hat, y_true).
        '''
        y_hat = self._load_npy_if_exists(f"y_hat_{layer_id}.npy", path)
        y_true = self._load_npy_if_exists("y_true.npy", path)
        if y_true is None:
            y_true = self._load_npy_if_exists(f"y_true_{layer_id}.npy", path)
        return y_hat, y_true
    
    def _load_npy_if_exists(self, name: str, path: Optional[Path] = None) -> Optional[np.ndarray]:
        if path is not None:
            assert path.exists(), f"The provided path {str(path)} does not exist."
            path = Path(path) 
        else:
            path = self.base_path  
        file_path = path / name
        if file_path.exists():
            return np.load(file_path)
        return None


#### Old code below for backward compatibility ####

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