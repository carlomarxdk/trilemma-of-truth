from __future__ import annotations

from hydra import compose, initialize
from omegaconf import OmegaConf


def load_hydra_config(path):
    with initialize(version_base="1.1", config_path="configs"):
        cfg = compose(config_name=path)
    return OmegaConf.to_container(cfg, resolve=True)


def load_hydra_config_with_params(model, datapack, probe, task, config_name):
    with initialize(version_base="1.1", config_path="configs"):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"model={model}",
                f"datapack={datapack}",
                f"probe={probe}",
                f"task={task}",
            ],
        )
    return cfg
