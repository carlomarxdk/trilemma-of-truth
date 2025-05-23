import logging
from typing import List, Dict
from abc import ABC, abstractmethod
import pickle as pkl
import json
import os
from glob import glob
import numpy as np
from hydra.utils import to_absolute_path
import re
import torch
import pandas as pd
log = logging.getLogger(__name__)


class ProbeData(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()


class MILProbeData(ProbeData):
    def __init__(self, output_dir, task, model_name, datapack, trial_name, probe_name: str = 'sbMIL', intervention_type: str = 'default'):
        super().__init__()
        self.output_dir = output_dir
        self.task = task
        self.model_name = model_name

        self.probe_name = probe_name
        self.datapack = datapack

        if task == 'full':
            self.trial_name = trial_name.split('task')[0]
            self.probe_dir = to_absolute_path(
                f'outputs/probes/{probe_name}/{model_name}/{self.trial_name}{task}/')

            self.intervention_dir = to_absolute_path(
                f'outputs/interv/{probe_name}/{model_name}/{self.trial_name}{task}/')
            self.mc_intervention_dir = to_absolute_path(
                f'outputs/interv/{probe_name}/{model_name}/_mc_/{self.trial_name}{task}/')
        else:
            self.trial_name = trial_name[:-2]
            self.probe_dir = to_absolute_path(
                f'outputs/probes/{probe_name}/{model_name}/{self.trial_name}-{task}/')
            self.intervention_dir = to_absolute_path(
                f'outputs/interv/{probe_name}/{model_name}/{self.trial_name}-{task}/')
            self.mc_intervention_dir = to_absolute_path(
                f'outputs/interv/{probe_name}/{model_name}/_mc_/{self.trial_name}-{task}/')
        assert len(self.available_layers()
                   ) > 0, f"No layers found in {self.probe_dir}"

    def metadata(self, layer_id: int):
        return self._load_layer_metadata(layer_id)

    def _load_layer_metadata(self, layer_id: int):
        if 'radlab' in self.probe_dir:
            a = re.search(r'\b(outputs)\b', self.probe_dir)
            probe_dir = self.probe_dir[a.start():]
            self.probe_dir = to_absolute_path(probe_dir)
        try:
            coef = np.load(f"/{self.probe_dir}/coef_{layer_id}.npy")
        except:
            coef = np.load(f"/{self.probe_dir}/direction_{layer_id}.npy")
        bias = np.load(f"/{self.probe_dir}/bias_{layer_id}.npy")
        yhat = np.load(self.probe_dir + f"/y_hat_{layer_id}.npy")
        ytrue = np.load(self.probe_dir + f"/y_true.npy")

        try:
            with open(self.probe_dir + f"/scaler_{layer_id}.pkl", "rb") as f:
                scaler = pkl.load(f)
        except:
            scaler = None

        try:
            with open(self.probe_dir + f"/calibrator_{layer_id}.pkl", "rb") as f:
                calibrator = pkl.load(f)
        except:
            with open(self.probe_dir + f"/cp_{layer_id}.pkl", "rb") as f:
                calibrator = pkl.load(f)
        with open(self.probe_dir + f"/metrics_{layer_id}.json", "rb") as f:
            metrics = json.load(f)

        return {
            "direction": coef,
            "bias": bias,
            "scaler": scaler,
            "calibrator": calibrator,
            "y_hat": yhat,
            "y_true": ytrue,
            "metrics_default": metrics['default'],
            "metrics_conformal": metrics['conformal']
        }

    def return_preds(self, layer_id: int):
        '''
        Returns the predictions for the given layer id:
        yhat: the predictions
        ytrue: the true labels
        '''
        yhat = np.load(self.probe_dir + f"/y_hat_{layer_id}.npy")
        ytrue = np.load(self.probe_dir + f"/y_true.npy")
        return yhat, ytrue

    def return_scores(self, layer_id: int):
        '''
        Returns the scores for (15 items of the bag) the given layer id:
        yhat: the scores
        ytrue: the true labels
        '''
        yhat = np.load(self.probe_dir + f"/y_15hat_{layer_id}.npy")
        ytrue = np.load(self.probe_dir + f"/y_true.npy")
        return yhat, ytrue

    def calibrator(self, layer_id: int):
        return self.metadata(layer_id)["calibrator"]

    def scaler(self, layer_id: int):
        return self.metadata(layer_id)["scaler"]

    def direction(self, layer_id: int, as_tensor=False):
        if as_tensor:
            return torch.tensor(self.metadata(layer_id)["direction"])
        return self.metadata(layer_id)["direction"]

    def bias(self, layer_id: int, as_tensor=False):
        if as_tensor:
            return torch.tensor(self.metadata(layer_id)["bias"])
        return self.metadata(layer_id)["bias"]

    def predict(self, layer_id: int, X):
        return np.dot(X, self.direction(layer_id)) + self.bias(layer_id)

    def predict_with_bag(self, layer_id: int, bags: list, with_scaling=False):
        preds = []
        for bag in bags:
            if with_scaling:
                bag = self.scaler(layer_id).transform(bag)
            pred = self.predict(layer_id=layer_id, X=bag)
            pred = np.max(pred)
            preds.append(pred)
        return np.array(preds)

    def metrics_conformal(self, layer_id: int):
        return self.metadata(layer_id)["metrics_conformal"]

    def metrics_default(self, layer_id: int):
        return self.metadata(layer_id)["metrics_default"]

    def available_layers(self):
        """
        Load all the layers that have metadata"""
        recorded_coefs = glob(f"{self.probe_dir}/metrics_*")
        layers = []
        for file in recorded_coefs:
            match = re.search(r'metrics_(\d+)', file)
            if match:
                layers.append(int(match.group(1)))
        return sorted(layers)

    def available_intervention_layers(self):
        """
        Load all the layers that have metadata"""
        recorded_coefs = glob(f"{self.intervention_dir}/layer_*.json")
        layers = []
        for file in recorded_coefs:
            match = re.search(r'layer_(\d+)', file)
            if match:
                layers.append(int(match.group(1)))
        return sorted(layers)

    def available_mc_intervention_layers(self):
        """
        Load all the layers that have metadata"""
        recorded_coefs = glob(f"{self.mc_intervention_dir}/layer_*.json")
        layers = []
        for file in recorded_coefs:
            match = re.search(r'layer_(\d+)', file)
            if match:
                layers.append(int(match.group(1)))
        return sorted(layers)

    def best_layer(self, metric='map'):
        layers = self.available_layers()
        best_layer = None
        best_score = -1
        for layer in layers:
            metrics = self.metrics_conformal(layer)
            val = metrics[metric]
            if type(val) == list:
                val = val[0]
            if val > best_score:
                best_score = val
                best_layer = layer
        return best_layer

    def intervention_metadata(self, layer_id: int):
        with open(self.intervention_dir + f"/layer_{layer_id}.json", "rb") as f:
            return json.load(f)

    def mc_intervention_metadata(self, layer_id: int):
        with open(self.mc_intervention_dir + f"/layer_{layer_id}.json", "rb") as f:
            return json.load(f)

    def mc_intervention_scores(self, layer_id: int):
        return {
            'neg': pd.read_csv(self.mc_intervention_dir + f"/layer_{layer_id}_neg.csv.gz", compression='gzip'),
            'orig': pd.read_csv(self.mc_intervention_dir + f"/layer_{layer_id}_orig.csv.gz", compression='gzip'),
            'pos': pd.read_csv(self.mc_intervention_dir + f"/layer_{layer_id}_pos.csv.gz", compression='gzip'),
        }

    def intervention_scores(self, layer_id: int):
        return {
            'rneg': np.load(self.intervention_dir + f"/layer_{layer_id}_rneg.npy"),
            'rorig': np.load(self.intervention_dir + f"/layer_{layer_id}_rorig.npy"),
            'rpos': np.load(self.intervention_dir + f"/layer_{layer_id}_rpos.npy"),
            'sneg': np.load(self.intervention_dir + f"/layer_{layer_id}_sneg.npy"),
            'sorig': np.load(self.intervention_dir + f"/layer_{layer_id}_sorig.npy"),
            'spos': np.load(self.intervention_dir + f"/layer_{layer_id}_spos.npy"),
        }


class MCProbeData(ProbeData):
    '''
    Reads the multiclss MIL probe data.
    '''

    def __init__(self,
                 model_name,
                 datapack_name,
                 trained_with_search: bool = True,
                 probe_name: str = 'sbMIL2'):
        super().__init__()
        if trained_with_search:
            self.probe_dir = f"outputs/probes/{probe_name}/{model_name}/{datapack_name}_search_full/"
        else:
            self.probe_dir = f"outputs/probes/{probe_name}/{model_name}/{datapack_name}_full/"
        self.model_name = model_name
        self.probe_name = probe_name
        self.datapack_name = datapack_name
        assert len(self.available_layers()
                   ) > 0, f"No layers found in {self.probe_dir}"

    def metadata(self, layer_id: int):
        return self._load_layer_metadata(layer_id)

    def _load_layer_metadata(self, layer_id: int):
        yhat = np.load(self.probe_dir + f"/y_hat_{layer_id}.npy")
        ytrue = np.load(self.probe_dir + f"/y_true.npy")

        with open(self.probe_dir + f"/cp_{layer_id}.pkl", "rb") as f:
            calibrator = pkl.load(f)

        with open(self.probe_dir + f"/metrics_{layer_id}.json", "rb") as f:
            metrics = json.load(f)

        try:
            with open(self.probe_dir + f"/model_{layer_id}.pkl", "rb") as f:
                model = pkl.load(f)
        except:
            model = None

        return {
            "model": model,
            "calibrator": calibrator,
            "y_hat": yhat,
            "y_true": ytrue,
            "metrics_default": metrics['default'],
            "metrics_conformal": metrics['conformal']
        }

    def calibrator(self, layer_id: int):
        return self.metadata(layer_id)["calibrator"]

    def model(self, layer_id: int):
        return self.metadata(layer_id)["model"]

    def predict(self, layer_id: int, X):
        return np.dot(X, self.direction(layer_id)) + self.bias(layer_id)

    def predict_with_bag(self, layer_id: int, bags):
        preds = []
        for bag in bags:
            pred = self.predict(layer_id=layer_id, X=bag)
            pred = np.max(pred)
            preds.append(pred)
        return np.array(preds)

    def return_true(self, layer_id: int):
        return self.metadata(layer_id)["y_true"]

    def return_pred(self, layer_id: int):
        return self.metadata(layer_id)["y_hat"]

    def return_preds(self, layer_id: int):
        '''
        Returns the predictions for the given layer id:
        yhat: the predictions
        ytrue: the true labels
        '''
        yhat = np.load(self.probe_dir + f"/y_hat_{layer_id}.npy")
        ytrue = np.load(self.probe_dir + f"/y_true.npy")
        return yhat, ytrue

    def metrics_conformal(self, layer_id: int):
        return self.metadata(layer_id)["metrics_conformal"]

    def metrics_default(self, layer_id: int):
        return self.metadata(layer_id)["metrics_default"]

    def available_layers(self):
        """
        Load all the layers that have metadata"""
        files = glob(f"{self.probe_dir}/model_*")
        layers = []
        for file in files:
            match = re.search(r'model_(\d+)', file)
            if match:
                layers.append(int(match.group(1)))
        return sorted(layers)

    def best_layer(self, metric='map', per_bag=False):
        layers = self.available_layers()
        best_layer = None
        best_score = -1
        for layer in layers:
            if per_bag:
                metrics = self.metrics_conformal(layer)['_bag']
            else:
                metrics = self.metrics_conformal(layer)
            val = metrics[metric]
            if type(val) == list:
                val = val[0]
            if val > best_score:
                best_score = val
                best_layer = layer
        return best_layer

    def top_k_layers(self, metric='map', k=5):
        layers = self.available_layers()
        best_layers = []
        best_scores = []
        for layer in layers:
            metrics = self.metrics_conformal(layer)
            val = metrics[metric]
            if type(val) == list:
                val = val[0]
            best_layers.append(layer)
            best_scores.append(val)
        best_layers = np.array(best_layers)[np.argsort(best_scores)[::-1]]
        return best_layers[:k]

    def return_general_metric(self, layer_id: int, datapack_name: str):
        '''
        Returns the generalization metric for the given layer id:'''
        output_dir = self.probe_dir.split('full')[0] + f'-to-{datapack_name}/'

        with open(output_dir + f"/metrics_{layer_id}.json", "rb") as f:
            metrics = json.load(f)

        return {
            "metrics_default": metrics['default'],
            "metrics_conformal": metrics['conformal']
        }

    def best_layer_generalization(self, datapack_name: str, metric='map', metric_type='conformal'):
        layers = self.available_layers()
        best_layer = None
        best_score = -1
        for layer in layers:
            metrics = self.return_general_metric(
                layer, datapack_name)[f"metrics_{metric_type}"]
            val = metrics[metric]
            if type(val) == list:
                val = val[0]
            if val > best_score:
                best_score = val
                best_layer = layer
        return best_layer
