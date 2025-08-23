# Description: Script to compress activations from a model

import logging
import hydra
from omegaconf import DictConfig
from data_handler import shape_as_tuple
import os
import numpy as np
import gc
log = logging.getLogger(__name__)


def validate_config(cfg: DictConfig):
    assert cfg.agg in [
        "last", "mean", "max", 'full'], "Aggregation tupe must be either 'last', 'mean' or 'max'."
    assert len(cfg.layers) > 0, "At least one layer must be selected."
    assert type(
        cfg.datasets) == list or type(cfg.datasets).__name__ == "ListConfig", f"Datasets must be a list. Not {type(cfg.datasets)}"
    assert len(cfg.datasets) > 0, "At least one dataset must be selected."


def log_stats(cfg):
    log.warning(
        f"Collecting activations for: {cfg.model.name} (device: {cfg.device})")


@hydra.main(version_base=None, config_path="configs", config_name="activations")
def main(cfg: DictConfig):
    validate_config(cfg)
    log_stats(cfg)

    for dataset in cfg.datasets:
        log.warning(
            f'Running the compression for {cfg.model["name"]} ->> {dataset}...')

        save_dir = f"{cfg.output_dir}/{dataset}/{cfg.agg}/"
        save_path = {}
        compress_path = {}
        for layer in cfg.layers:
            save_path[layer] = save_dir + f"layer_{layer}_e_temp.npy"
            compress_path[layer] = save_dir + f"layer_{layer}_e.npz"

        gc.collect()
        for layer in cfg.layers:
            try:
                _shape = shape_as_tuple(np.load(save_dir + "shape.npy"))

                acts = np.memmap(save_path[layer],
                                 dtype='float16', mode='r', shape=_shape)

                acts = np.array(acts)  # Convert to standard NumPy array

                # Save the compressed array
                np.savez_compressed(compress_path[layer], acts)
                del acts
                # Delete the temporary memmap file
                os.remove(save_path[layer])
                log.warning(
                    f"\tCompressed activations for {cfg.model.name} ->> {dataset} ->> {layer}")
            except Exception as e:
                log.warning(
                    f"\tError compressing {cfg.model.name} activations for {dataset} ->> {layer}: {e}")

        log.warning(f"{cfg.model.name} activations compressed for {dataset}")
    exit()


if __name__ == "__main__":
    main()
