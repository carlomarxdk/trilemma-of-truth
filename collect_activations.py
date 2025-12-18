"""Script to generate and save model activations from language models.

This script processes datasets through a language model and saves the hidden
state activations for downstream analysis tasks.
"""

from __future__ import annotations

import gc
import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils_hydra import get_device, load_statements, prepare_hf_model

log = logging.getLogger(__name__)


def validate_config(cfg: DictConfig):
    """Validate configuration parameters.

    Args:
        cfg: Configuration object to validate.

    Raises:
        AssertionError: If configuration is invalid.
    """
    assert cfg.agg in [
        "last",
        "mean",
        "max",
        "full",
    ], "Aggregation tupe must be either 'last', 'mean' or 'max'."
    assert len(cfg.layers) > 0, "At least one layer must be selected."
    assert (
        isinstance(cfg.datasets, list) or type(cfg.datasets).__name__ == "ListConfig"
    ), f"Datasets must be a list. Not {type(cfg.datasets)}"
    assert len(cfg.datasets) > 0, "At least one dataset must be selected."
    if cfg.device is None:
        OmegaConf.set_struct(cfg, False)  # Allow overriding
        cfg["device"] = str(get_device())
        OmegaConf.set_struct(cfg, True)


def log_stats(cfg):
    """Log configuration statistics.

    Args:
        cfg: Configuration object.
    """
    log.warning(f"Collecting activations for: {cfg.model.name} (device: {cfg.device})")
    log.warning(f"Max length of the input sequences: {cfg.max_length}")


def tokenize(batch, tokenizer, cfg):
    """Tokenize a batch based on model configuration.

    Args:
        batch: Batch of text strings to tokenize.
        tokenizer: Tokenizer instance.
        cfg: Configuration object.

    Returns:
        Tokenized input sequences.
    """
    if cfg.model["instruct"]:
        return instruct_tokenize(batch, tokenizer, cfg)
    else:
        return default_tokenize(batch, tokenizer, cfg)


def default_tokenize(batch, tokenizer, cfg):
    """Tokenize batch using default settings.

    Args:
        batch: Batch of text strings.
        tokenizer: Tokenizer instance.
        cfg: Configuration object.

    Returns:
        Tokenized input sequences.
    """
    if cfg.agg == "last":
        input_seqs = tokenizer(batch.tolist(), return_tensors="pt", padding=True)
    elif cfg.agg == "full":
        input_seqs = tokenizer(
            batch.tolist(),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )
    return input_seqs


def instruct_tokenize(batch, tokenizer, cfg):
    """Tokenize batch using instruction template.

    Args:
        batch: Batch of text strings.
        tokenizer: Tokenizer instance with chat template.
        cfg: Configuration object.

    Returns:
        Tokenized input sequences formatted as instructions.
    """
    message_batch = [[{"role": "user", "content": x}] for x in batch]
    text_batch = tokenizer.apply_chat_template(
        message_batch,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenizer.truncation_side = "left"  # optional but recommended
    tokenizer.padding_side = "left"
    if cfg.agg == "last":
        input_seqs = tokenizer(text_batch, return_tensors="pt", padding=True)
    elif cfg.agg == "full":
        input_seqs = tokenizer(
            text_batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )
    return input_seqs


class Hook:
    """Hook to extract and store model layer outputs."""

    def __init__(self):
        """Initialize hook with empty output storage."""
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        """Extract output from module.

        Args:
            module: The module being hooked.
            module_inputs: Input tensors to the module.
            module_outputs: Output tensors from the module.
        """
        try:
            output, _ = module_outputs
        except (ValueError, TypeError):
            output = module_outputs[0]

        self.out = output.detach().clone()


@hydra.main(version_base=None, config_path="configs", config_name="activations")
def main(cfg: DictConfig):
    validate_config(cfg)
    log_stats(cfg)
    model, tokenizer = prepare_hf_model(cfg)
    log.warning(model)

    # Check if we need to adjust layer indices
    # Some models include embedding layer in hidden_states[0]
    with torch.no_grad():
        test_input = tokenizer("test", return_tensors="pt")
        test_output = model(
            test_input["input_ids"].to(cfg.device),
            output_hidden_states=True,
            use_cache=False,
        )
        num_hidden_states = len(test_output.hidden_states)
        log.warning(
            f"Model outputs {num_hidden_states} hidden states (including embeddings)"
        )
        log.warning(f"Hidden state 0 shape: {test_output.hidden_states[0].shape}")
        log.warning(f"Hidden state 1 shape: {test_output.hidden_states[1].shape}")

    torch.set_grad_enabled(False)
    model.eval()

    for dataset in cfg.datasets:
        statements = load_statements(dataset)
        n_batches = len(statements) // int(cfg.batch_size)
        batches = np.array_split(statements, n_batches)

        log.warning(
            f"Generating activations for {dataset} with {len(statements)} statements in {len(batches)} batches."
        )
        log.info(f"\tExample of a statement: {statements[0]}")

        # Get dimensions from test run
        input_seq = tokenizer(statements[0], return_tensors="pt")
        with torch.no_grad():
            test_output = model(
                input_seq["input_ids"].to(cfg.device),
                output_hidden_states=True,
                use_cache=False,
            )

        # Check the shape of hidden states
        # hidden_states[0] is usually the embedding layer
        # hidden_states[1] through hidden_states[n] are the transformer layers
        test_hidden = test_output.hidden_states[1]  # First transformer layer
        log.warning(f"Test hidden state shape: {test_hidden.shape}")

        if test_hidden.dim() == 3:
            hidden_size = test_hidden.shape[-1]
            log.warning(f"3D hidden states detected: hidden_size={hidden_size}")
        else:
            log.error(f"Unexpected hidden state dimensions: {test_hidden.shape}")
            raise ValueError("Hidden states should be 3D")

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
            if cfg.agg == "last":
                acts_memmap[layer] = np.memmap(
                    save_path[layer],
                    dtype="float16",
                    mode="w+",
                    shape=(len(statements), hidden_size),
                )
            elif cfg.agg == "full":
                acts_memmap[layer] = np.memmap(
                    save_path[layer],
                    dtype="float16",
                    mode="w+",
                    shape=(len(statements), MAX_LEN, hidden_size),
                )
                _shape = (len(statements), MAX_LEN, hidden_size)
                np.save(save_dir + "shape.npy", _shape)

        _last_row = 0
        masks = []

        # Checkpoint setup
        total_samples = sum(len(b) for b in batches)
        batch1_cp = len(batches[0])
        batch2_cp = batch1_cp + len(batches[1])
        percent_cps = {int(total_samples * p / 10) for p in range(1, 10)}
        checkpoints = sorted({batch1_cp, batch2_cp} | percent_cps)
        next_checkpoint = min(checkpoints) if checkpoints else None

        for batch_idx, batch in tqdm(enumerate(batches), total=len(batches)):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            input_seqs = tokenize(batch, tokenizer, cfg)
            input_ids = input_seqs["input_ids"].to(cfg.device)
            input_att = input_seqs["attention_mask"].to(cfg.device)
            masks.append(input_att[:, -MAX_LEN:].detach().cpu())

            # Get hidden states directly from model
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=input_att,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hidden_states = outputs.hidden_states

            # Process each requested layer
            for layer in cfg.layers:
                # Note: layer 0 in cfg.layers corresponds to hidden_states[1] (first transformer layer)
                # Adjust index as needed based on your model
                layer_output = hidden_states[layer + 1]  # +1 to skip embedding layer

                # Debug first batch
                if batch_idx == 0 and layer == cfg.layers[0]:
                    log.info(f"Layer {layer} hidden state shape: {layer_output.shape}")

                # hidden_states should always be 3D: (batch_size, seq_len, hidden_size)
                if layer_output.dim() != 3:
                    log.error(
                        f"Unexpected hidden state dimensions for layer {layer}: {layer_output.shape}"
                    )
                    continue

                batch_size, seq_len, hidden_dim = layer_output.shape

                if cfg.agg == "full":
                    # Adjust to MAX_LEN
                    if seq_len > MAX_LEN:
                        embeddings = (
                            layer_output[:, -MAX_LEN:, :]
                            .cpu()
                            .numpy()
                            .astype(np.float16)
                        )
                    elif seq_len < MAX_LEN:
                        embeddings = layer_output.cpu().numpy().astype(np.float16)
                        pad_width = MAX_LEN - seq_len
                        embeddings = np.pad(
                            embeddings,
                            ((0, 0), (pad_width, 0), (0, 0)),
                            mode="constant",
                            constant_values=0,
                        )
                    else:
                        embeddings = layer_output.cpu().numpy().astype(np.float16)

                    # Verify shape
                    assert embeddings.shape == (
                        batch.shape[0],
                        MAX_LEN,
                        hidden_dim,
                    ), f"Shape mismatch: {embeddings.shape}"

                    # Store
                    for i in range(batch.shape[0]):
                        acts_memmap[layer][_last_row + i, :, :] = embeddings[i]

                    if batch_idx < 2 and layer % 5 == 0:
                        log.info(
                            f"Full Emb L{layer} | Example {embeddings[0, -1, -3:]}"
                        )

                elif cfg.agg == "last":
                    # Extract last token
                    embeddings = layer_output[:, -1, :].cpu().numpy().astype(np.float16)

                    for i in range(batch.shape[0]):
                        acts_memmap[layer][_last_row + i, :] = embeddings[i]

                # Force flush
                acts_memmap[layer].flush()

            _last_row += batch.shape[0]

            # Checkpoint logging
            if next_checkpoint is not None and _last_row >= next_checkpoint:
                log.info(
                    f"[{int((next_checkpoint/total_samples)*100)}%] Processing activations | Statement: {batch[-1]}"
                )

                for layer in cfg.layers:
                    if layer % 5 == 0:
                        if cfg.agg == "last":
                            last_emb = acts_memmap[layer][_last_row - 1]
                            log.info(f"Layer: {layer:>3} | Example {last_emb[-3:]}")
                        elif cfg.agg == "full":
                            last_emb = acts_memmap[layer][_last_row - 1]
                            log.info(f"Layer: {layer:>3} | Example {last_emb[-1][-3:]}")

                checkpoints.remove(next_checkpoint)
                next_checkpoint = min(checkpoints) if checkpoints else None

            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Later, at the end of processing (no change needed if you fixed above):
        masks = torch.vstack(
            masks
        ).numpy()  # Now this works because masks are already on CPU
        np.save(save_dir + "mask.npy", masks)

        log.info(f"\tCompression of activations for {dataset} started...")
        for layer in cfg.layers:
            acts_memmap[layer].flush()
            # Optional: compress to npz
            # data = np.array(acts_memmap[layer][:])
            # np.savez_compressed(compress_path[layer], activations=data)

        log.warning(f"{cfg.model.name} activations saved for {dataset}")

    exit()


if __name__ == "__main__":
    main()
