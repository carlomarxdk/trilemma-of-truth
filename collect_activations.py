# Description: Experiment script to GENERATE and SAVE activations from a model

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from utils_hydra import get_device, prepare_hf_model, load_statements, return_layers
from tqdm import tqdm
import torch
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
    if cfg.device == None:
        OmegaConf.set_struct(cfg, False)  # Allow overriding
        cfg["device"] = str(get_device())
        OmegaConf.set_struct(cfg, True)


def log_stats(cfg):
    log.warning(
        f"Collecting activations for: {cfg.model.name} (device: {cfg.device})")
    log.warning(f'Max length of the input sequences: {cfg.max_length}')


def tokenize(batch, tokenizer, cfg):
    if cfg.model["instruct"]:
        return instruct_tokenize(batch, tokenizer, cfg)
    else:
        return default_tokenize(batch, tokenizer, cfg)


def default_tokenize(batch, tokenizer, cfg):
    if cfg.agg == 'last':
        input_seqs = tokenizer(
            batch.tolist(), return_tensors="pt", padding=True)
    elif cfg.agg == 'full':
        input_seqs = tokenizer(
            batch.tolist(), return_tensors="pt", padding="max_length",  truncation=True, max_length=cfg.max_length)
    return input_seqs


def instruct_tokenize(batch, tokenizer, cfg):
    message_batch = [[{"role": "user", "content": x}] for x in batch]
    text_batch = tokenizer.apply_chat_template(
        message_batch,
        tokenize=False,
        add_generation_prompt=False,
    )
    if cfg.agg == 'last':
        input_seqs = tokenizer(
            text_batch, return_tensors="pt", padding=True)
    elif cfg.agg == 'full':
        input_seqs = tokenizer(
            text_batch, return_tensors="pt", padding="max_length",  truncation=True, max_length=cfg.max_length)
    return input_seqs


class Hook:
    """
    Class to extract the outputs from the model
    """

    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        try:
            self.out, _ = module_outputs
        except:
            self.out = module_outputs[0]


@hydra.main(version_base=None, config_path="configs", config_name="activations")
def main(cfg: DictConfig):
    validate_config(cfg)
    log_stats(cfg)
    model, tokenizer = prepare_hf_model(cfg)
    log.warning(model)
    _check_layers = return_layers(model, cfg)
    if len(cfg.layers) != len(_check_layers):
        log.warning(
            f"Selected number of layers ({len(cfg.layers)}) DOES NOT match the model's number of layers ({len(_check_layers)})")
    else:
        log.warning(
            f"Selected number of layers ({len(cfg.layers)}) match the model's number of layers ({len(_check_layers)})")
    torch.set_grad_enabled(False)
    # Setup hooks
    hooks, handles = [], []
    for layers in cfg.layers:
        hook = Hook()
        handle = model.get_submodule(cfg.model["module"]).get_submodule(
            cfg.model["encoders"])[layers].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)

    for dataset in cfg.datasets:
        statements = load_statements(dataset)
        n_batches = len(statements) // int(cfg.batch_size)
        batches = np.array_split(statements, n_batches)

        log.warning(
            f"Generating activations for {dataset} with {len(statements)} statements in {len(batches)} batches.")
        log.info(f"\tExample of a statement: {statements[0]}")

        # Precompute the length of the input sequences
        input_seq = tokenizer(statements[0], return_tensors="pt")
        _ = model(input_seq["input_ids"].to(cfg.device))
        hidden_size = hooks[0].out.shape[-1]

        save_dir = f"{cfg.output_dir}/{dataset}/{cfg.agg}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        acts_memmap = {}
        save_path = {}
        compress_path = {}
        MAX_LEN = cfg.max_length

        for layer in cfg.layers:
            save_path[layer] = save_dir + f"layer_{layer}_e_temp.npy"
            compress_path[layer] = save_dir + f"layer_{layer}_e.npz"
            if cfg.agg == 'last':
                acts_memmap[layer] = np.memmap(save_path[layer], dtype='float16', mode='w+',
                                               shape=(len(statements), hidden_size))
            elif cfg.agg == 'full':
                acts_memmap[layer] = np.memmap(save_path[layer], dtype='float16', mode='w+',
                                               shape=(len(statements), MAX_LEN, hidden_size))

                _shape = (len(statements), MAX_LEN, hidden_size)
                np.save(save_dir + "shape.npy", _shape)

        _last_row = 0
        masks = []
        for _, batch in tqdm(enumerate(batches), total=len(batches)):
            input_seqs = tokenize(batch, tokenizer, cfg)
            input_ids = input_seqs["input_ids"].to(cfg.device)
            input_att = input_seqs["attention_mask"].to(cfg.device)
            masks.append(input_att[-MAX_LEN:].detach())

            _ = model(input_ids, attention_mask=input_att,
                      use_cache=False)
            for layer, hook in zip(cfg.layers, hooks):
                output = hook.out

                if output.dtype != torch.float32:
                    output = output.float()
                if cfg.agg == 'last':
                    embeddings = output[:, -1].detach().cpu().numpy().astype(
                        np.float16)

                    if torch.isnan(embeddings).any():
                        log.warning(
                            f"NaN values found in the embeddings for {dataset} | {layer} | {cfg.model.name}")

                elif cfg.agg == 'full':
                    embeddings = output.detach().cpu().numpy().astype(np.float16)
                else:
                    raise NotImplementedError(
                        "Other aggregation types are not implemented")

                for i in range(batch.shape[0]):
                    acts_memmap[layer][_last_row + i, :, :] = embeddings[i]
            _last_row += batch.shape[0]

        masks = torch.vstack(masks).to('cpu').numpy()
        np.save(save_dir + "mask.npy", masks)

        log.info(f'\tCompression of activations for {dataset} started...')
        for layer in cfg.layers:
            acts_memmap[layer].flush()

        log.warning(f"{cfg.model.name} activations saved for {dataset}")
    exit()


if __name__ == "__main__":
    main()
