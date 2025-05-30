from hydra import compose, initialize
import os
from omegaconf import OmegaConf


def load_hydra_config(path):
    with initialize(version_base="1.1", config_path="configs"):
        cfg = compose(config_name=path)
    return OmegaConf.to_container(cfg, resolve=True)


def load_hydra_config_with_params(model, datapack, probe, config_name):
    with initialize(version_base="1.1", config_path="configs"):
        cfg = compose(config_name=config_name, overrides=[
                      f"model={model}", f"datapack={datapack}", f"probe={probe}"])
    OmegaConf.set_struct(cfg, False)  # Allow overriding
    trial_name = cfg.trial_name
    if cfg.probe['name'] == 'mean-diff':
        cfg.search = False
    if cfg.probe["sparsify_data"] > 0:
        trial_name += f"_sparse-{cfg.probe['sparsify_data']}"
    if cfg.search:
        trial_name += "_search"
    trial_name += f'_task-{cfg.task}'
    cfg["trial_name"] = trial_name
    # if cfg["task"] == 2:
    #     cfg["probe"]["assume_known_positives"] = False
    cfg["output_dir"] = os.path.join(cfg.output_dir, trial_name)
    OmegaConf.set_struct(cfg, True)
    return OmegaConf.to_container(cfg, resolve=True)
