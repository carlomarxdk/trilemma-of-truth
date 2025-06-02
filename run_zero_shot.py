# Description: Experiment script to COLLECT responses of the model (based on the prompt)
from response.collector import MultichoiceLogitCollector
from response.prompt_templates import MultichoicePrompt, MultichoicePromptTF, MultichoicePromptABC
from tqdm import tqdm
import logging
from utils_hydra import get_device, prepare_hf_model, load_statements_with_targets
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import re

log = logging.getLogger(__name__)


def validate_config(cfg: DictConfig):
    assert type(
        cfg.datasets) == list or type(cfg.datasets).__name__ == "ListConfig", f"Datasets paramaeter must be a list. Not {type(cfg.datasets)}"
    assert len(cfg.datasets) > 0, "At least one dataset must be selected."
    if cfg.device == None:
        OmegaConf.set_struct(cfg, False)  # Allow overriding
        cfg["device"] = str(get_device()) # Change the device (if it is not set)
        OmegaConf.set_struct(cfg, True)


def log_stats(cfg):
    log.warning(
        f"Collecting prompt-based scores for: {cfg.model.name} (device: {cfg.device})")


@hydra.main(version_base=None, config_path="configs", config_name="probe_zeroshot")
def main(cfg: DictConfig):
    validate_config(cfg)
    log_stats(cfg)
    model, tokenizer = prepare_hf_model(cfg)
    torch.set_grad_enabled(False)
    print(cfg.output_dir)
    # print(model)
    if cfg.model["instruct"]:
        mode = "instruct"
    else:
        mode = "default"
    if cfg.question_type == "binary":
        raise NotImplementedError()
        # prompt_f = BinaryPrompt(enumeration=cfg.enum_list)
    elif cfg.question_type == "multichoice":
        if cfg['variation'] == "default":
            prompt_f = MultichoicePrompt(
                enumeration=cfg.enum_list, prompt_type=mode, system_role=cfg.model["system_role"],
                user_role=cfg.model["user_role"],
                assist_role=cfg.model["assist_role"])
        elif cfg['variation'] == "shuffled":
            prompt_f = MultichoicePrompt(
                enumeration=["2", "6", "1", "5", "3", "4"], prompt_type=mode, system_role=cfg.model["system_role"],
                user_role=cfg.model["user_role"],
                assist_role=cfg.model["assist_role"])
        elif cfg['variation'] == "tf":
            prompt_f = MultichoicePromptTF(
                enumeration=cfg.enum_list, prompt_type=mode, system_role=cfg.model["system_role"],
                user_role=cfg.model["user_role"],
                assist_role=cfg.model["assist_role"])
        elif cfg['variation'] == "abc":
            prompt_f = MultichoicePromptABC(
                prompt_type=mode, system_role=cfg.model["system_role"],
                user_role=cfg.model["user_role"],
                assist_role=cfg.model["assist_role"])
        else:
            raise ValueError(
                "Invalid variation. Choose from 'default', 'tf', 'abc' or 'shuffled'")
        collector = MultichoiceLogitCollector(tokenizer, prompt_f)

    # 4. RUN THE EXPERIMENT
    for dataset in cfg.datasets:
        statetements, targets = load_statements_with_targets(dataset)
        # 1. Assembe batches and set bookikeeping
        n_batches = len(statetements) // cfg.batch_size
        if cfg.limit_batches > 0:
            batches = np.array_split(statetements, n_batches)[
                :cfg.limit_batches]
        else:
            batches = np.array_split(statetements, n_batches)
        __log_intervals = np.linspace(0, n_batches, 5, dtype=int)
        log.warning(
            f"Generating scores for {dataset} with {len(statetements)} statements ({n_batches} batches)")
        if cfg.model["instruct"]:
            if cfg.model["end_token"] == "auto":
                end_token = tokenizer.eos_token
            else:
                end_token = cfg.model["end_token"]
            _prompt = re.sub(rf"{re.escape(end_token)}\s*$", " ",
                             tokenizer.apply_chat_template(prompt_f(batches[0][0]),
                                                           tokenize=False,
                                                           add_generation_prompt=False,
                                                           continue_final_message=True))  # Remove end token if present
    # Ensure a single space at the end  # we need to make sure to remove the token <eot_id> which says that the responses is finished
        else:
            _prompt = prompt_f(batches[0][0])

        log.warning(
            f"Examples of a prompt:\n---\n {_prompt}\n---")

        scores = []
        # 3. Collect scores
        for i, batch in enumerate(tqdm(batches, total=len(batches))):
            prompt = [prompt_f(s) for s in batch]
            tokenized = tokenize(prompt, tokenizer, cfg)
            input_ids, input_att = tokenized["input_ids"].to(
                cfg.device), tokenized["attention_mask"].to(cfg.device)
            if i in __log_intervals:
                status = (i/n_batches)
                log.warning(
                    f"(BATCH) Processed {status:.2%} of the statements")
            if cfg.question_type == "binary":
                raise NotImplementedError()
            elif cfg.question_type == "multichoice":
                out = model(input_ids,
                            attention_mask=input_att,
                            use_cache=False).logits
                s = collector.collect_proba(out[:, -1])
                scores.append(s)
            else:
                raise ValueError(
                    "Invalid question type. Choose from 'binary' or 'multichoice'")

        scores = torch.cat(scores, dim=0)
        if cfg.limit_batches < 0:
            assert scores.shape[0] == len(
                statetements), "Scores and statements must have the same length"
        save_dir = f"{cfg.output_dir}/{dataset}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(f"{save_dir}/scores.npy", scores.cpu().float().numpy())
        np.save(f"{save_dir}/targets.npy", np.array(targets))
        log.warning(
            f"(BATCH) Processed 100.00% of the statements\n\n")


def tokenize(batch, tokenizer, cfg):
    if cfg.model["instruct"]:
        return instruct_tokenize(batch, tokenizer, cfg)
    else:
        return default_tokenize(batch, tokenizer, cfg)


def default_tokenize(batch, tokenizer, cfg):
    input_seqs = tokenizer(
        batch, return_tensors="pt", padding="longest")
    return input_seqs


def instruct_tokenize(batch, tokenizer, cfg):
    batch = tokenizer.apply_chat_template(
        batch,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True
    )
    if cfg.model["end_token"] == "auto":
        end_token = tokenizer.eos_token
    else:
        end_token = cfg.model["end_token"]
    text_batch = [
        b.strip(' ').rstrip(end_token).strip(' ') + " "
        for b in batch
    ]
    input_seqs = tokenizer(
        text_batch, return_tensors="pt", padding="longest")
    return input_seqs


if __name__ == "__main__":
    """
    Compute the true scores for the statements (as a probability of tokens, i.e., output of the model).
    """
    main()
