import torch
import re
import numpy as np
import yaml
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from data_handler import DataHandler
import polars as pl
import os
import logging


log = logging.getLogger(__name__)

# Misc Functions


def should_process_layer(layer_id, cfg):
    """Determine if a given layer should be processed."""
    layer_range = np.quantile(
        cfg.model['layers'], cfg.layer_range, method="closest_observation")
    return (layer_range[0] <= layer_id <= layer_range[1])


# Other misc function


def return_layers(model, config):
    """
    Return the layers to save activations from (as a list)
    """
    return list(range(len(model.get_submodule(config["module"]).get_submodule(config["encoders"]))))


def load_data(args, datasets):
    """
    Load the data from the config file
    """
    dh = DataHandler(args.experiment, datasets,
                     args.agg, with_calibration=True, load_scores=True,
                     )
    dh.assemble(test_size=args.test_size, calibration_size=args.cal_size,
                seed=args.random_seed, exclusive_split=True)
    return dh


def save_checkpoint(scores, dataset, experiment, batch_index, suffix=""):
    """Save checkpoint with scores and the current batch index."""
    save_dir = f"outputs/checkpoints/{experiment}/{dataset}/{suffix}"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({'scores': scores, 'batch_index': batch_index},
               os.path.join(save_dir, "checkpoint.pt"))


def load_checkpoint(dataset, experiment):
    """Load checkpoint if it exists."""
    checkpoint_path = f"outputs/checkpoints/{experiment}/{dataset}/checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        return checkpoint['scores'], checkpoint['batch_index']
    return None, 0


def load_statements_from_multiple_data(dataset: List) -> List[str]:
    """
    Load the dataset from the mulitple files
    """
    data = list()
    for dataset in dataset:
        data.append(pl.read_csv(f"datasets/{dataset}.csv"))
    return flatten(pl.concat(data).select("statement").to_numpy())


def load_statements(dataset: List) -> List[str]:
    """
    Load the dataset from the file
    """
    return flatten(pl.read_csv(f"datasets/{dataset}.csv").select("statement").to_numpy()), flatten(pl.read_csv(f"datasets/{dataset}.csv").select("correct").to_numpy())


def prepare_hf_model(model_config, device):
    """
    Prepare the HuggingFace model and tokenizerz for the experiment
    """
    if model_config["dtype"] == "float16":
        _dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config["model"], token=model_config["token"], torch_dtype=_dtype,  attn_implementation="eager", device_map={"": device})
    elif model_config["dtype"] == "float32":
        _dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config["model"], token=model_config["token"], torch_dtype="auto", device_map={"": device})
    else:
        raise ValueError("dtype must be either 'bfloat16' or 'float32'.")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_config["model"], token=model_config["token"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # Load model

    return model, tokenizer


def load_config(key: str) -> List[int]:
    """
    Load the configuration from the file and return the values for the key
    """
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)[key]


def flatten(xss) -> List:
    """
    Flatten a list of lists
    """
    return [x for xs in xss for x in xs]


# class LogitCollector:
#     """
#     Collector creates and stores the information about the logits for the true and false tokens
#     """

#     def __init__(self,
#                  tokenizer,
#                  true_tokens=["true", "correct", "1", "yes", "right"],
#                  false_tokens=["false", "incorrect", "0", "no", "wrong"],
#                  agg: str = "sum"):
#         assert agg in [
#             "sum", "mean", "max"], "Only sum and mean aggregations are supported"
#         log.warning("This class is an old implementation of LogitCollector")
#         self.aggregation = agg
#         self.tokenizer = tokenizer
#         self.true_tokens = true_tokens
#         self.false_tokens = false_tokens
#         self.model_name = self.tokenizer.name_or_path

#         self.augmented_true_tokens = self.augment_token_list(self.true_tokens)
#         self.augmented_false_tokens = self.augment_token_list(
#             self.false_tokens)

#         self.true_ids, self.false_ids = self.assemble_ids(
#             true_tokens=self.augmented_true_tokens, false_tokens=self.augmented_false_tokens)
#         print(
#             f"True tokens: {len(self.true_ids)}, False tokens: {len(self.false_ids)}")

#     def augment_token_list(self, tokens):
#         """
#         Augment the token list by adding upper case, capitalized, and space separated versions of the tokens
#         """
#         output = tokens.copy()
#         output.extend([t.upper() for t in tokens])
#         output.extend([t.capitalize() for t in tokens])
#         output.extend([" %s" % t for t in output])

#         if self.model_name in ["microsoft/BioGPT-Large-PubMedQA"]:
#             output.extend(["%s</w>" % t for t in output])
#         return list(set(output))

#     def __return_ids__(self, tokens):
#         output = []
#         for t in tokens:
#             res = self.tokenizer.encode(t)
#             for r in res:
#                 if r not in self.tokenizer(" ").input_ids:
#                     output.append(r)
#         return list(set(output))

#     def assemble_ids(self, true_tokens, false_tokens):
#         """
#         Return the ids for the true and false tokens (augmented by upper case, capitalized, etc.)
#         """
#         true_ids = self.__return_ids__(true_tokens)
#         false_ids = self.__return_ids__(false_tokens)

#         intersections = set(true_ids).intersection(set(false_ids))
#         true_ids = list(set(true_ids).difference(intersections))
#         false_ids = list(set(false_ids).difference(intersections))
#         return true_ids, false_ids

#     def collect_logits(self, logits):
#         """
#         Get the aggregated logits for the true and false tokens
#         """
#         # if logits.dtype == torch.float16:
#         #    logits = logits.float()
#         # if logits.dim() == 1:
#         #    logits = logits.unsqueeze(0)
#         if self.aggregation == "sum":
#             false_logits = logits[:, self.false_ids].sum(-1)
#             true_logits = logits[:, self.true_ids].sum(-1)
#         elif self.aggregation == "mean":
#             false_logits = logits[:, self.false_ids].mean(-1)
#             true_logits = logits[:, self.true_ids].mean(-1)
#         elif self.aggregation == "max":
#             false_logits = torch.maximum(logits[:, self.false_ids], dim=-1)
#             true_logits = torch.maximum(logits[:, self.true_ids], dim=-1)
#         return torch.stack([false_logits, true_logits]).T

#     def collect_proba(self, logits):
#         """
#         Get the probabilities for the true and false tokens
#         """
#         return torch.softmax(self.collect_logits(logits), axis=1)


# def return_message(statement: str):
#     """
#     Create a prompt/message for the Instruct-type LLMs
#     """
#     return [
#         {"role": 'system',
#          "content": "You are an expert in fact-checking. Your task is to evaluate whether the user's "
#             "statement is correct based on your comprehensive knowledge. Please respond with "
#             "'Yes' if the statement is correct, or 'No' if it is not."},
#         {'role': 'user',
#          'content': f'Is the following statement correct? {statement}'}
#     ]


# def collect_predictions(scores, lengths, detach: bool = False):
#     """
#     Collect the prediction at the end of the sequence
#     """
#     if scores.dtype == torch.float16:
#         scores = scores.float()
#     indices = lengths.unsqueeze(1).unsqueeze(
#         2).expand(scores.shape[0], 1, scores.shape[2])
#     if detach:
#         return torch.gather(scores, 1, indices).squeeze(1).detach()
#     return torch.gather(scores, 1, indices).squeeze(1)


# def collect_last_embeddings(x, lengths, detach: bool = True):
#     if x.dtype == torch.float16:
#         x = x.float()
#     return collect_predictions(scores=x, lengths=lengths, detach=detach)


def translate_embedding(X, direction, target_coord: float, absolute: bool = True):
    """
    Translate the 'X' embedding in the direction  by 'new_coord' units (if abslute is True)
    Args:
    - X: The embedding to translate
    - direction: The direction to translate the embedding
    - new_coord: The new coordinate to move to
    Returns:
    - The translated embedding
    """
    # move in respect to the current coordinate
    curr_coord = torch.einsum("bsh, h -> bs", X, direction)
    delta = target_coord - curr_coord
    proj = torch.einsum("h, bs -> bsh", direction, delta)
    return X + proj


def amplify_embedding(X, direction, factor: float = 1.0):
    """
    Amplify the 'X' embedding in the direction by 'factor' units
    Args:
    - X: The embedding to amplify
    - direction: The direction to amplify the embedding
    - factor: The factor to amplify the embedding
    Returns:
    - The amplified embedding
    """
    proj = torch.einsum("h, bsh -> bsh", direction, X)
    proj = torch.sign(proj) * direction
    return X + factor * proj


def bootstrap_ci(metric_func, y_true, y_pred, n_bootstraps=1000, alpha=0.05, random_state=None):
    """
    Calculate bootstrapped confidence intervals for a given scikit-learn metric.

    Parameters:
    - metric_func: A scikit-learn metric function (e.g., accuracy_score, precision_score).
    - y_true: Array-like of shape (n_samples,), Ground truth (correct) labels.
    - y_pred: Array-like of shape (n_samples,), Predicted labels, as returned by a classifier.
    - n_bootstraps: Number of bootstrap samples to generate (default: 1000).
    - alpha: Significance level for the confidence intervals (default: 0.05).
    - random_state: Random seed for reproducibility (default: None).

    Returns:
    - ci_lower: Lower bound of the confidence interval.
    - ci_upper: Upper bound of the confidence interval.
    """

    if random_state is not None:
        np.random.seed(random_state)

    bootstrapped_scores = []

    for _ in range(n_bootstraps):
        # Generate a bootstrap sample
        indices = np.random.randint(0, len(y_true), len(y_true))
        y_true_bootstrap = y_true[indices]
        y_pred_bootstrap = y_pred[indices]

        # Calculate the metric for the bootstrap sample
        score = metric_func(y_true_bootstrap, y_pred_bootstrap)
        bootstrapped_scores.append(score)

    # Calculate the confidence interval
    ci_lower = np.percentile(bootstrapped_scores, 100 * (alpha / 2))
    ci_upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha / 2))

    return metric_func(y_true, y_pred), ci_lower, ci_upper


def return_label(data):
    """
    Return labels from the dataframe
    """
    correct, real, fake, fictional, negated = data["correct"].values, data["real_object"].values, data[
        "fake_object"].values, data["fictional_object"].values, data["negation"].values
    combined = np.select(
        [
            (correct == 0) & (real == 1) & (fake == 0) & (fictional == 0),
            (correct == 1) & (real == 1) & (fake == 0) & (fictional == 0),
            (fake == 1) & (fictional == 0) & (real == 0),
            (correct == 0) & (fake == 1) | (correct == 0) & (fictional == 1),
            (correct == 1) & (fake == 1) | (correct == 1) & (fictional == 1),
        ],
        [0, 1, 4, 2, 3], default=4
    )
    return correct, real, fake, combined, negated, fictional


def normalize(X):
    if X.ndim == 1:
        return X / np.linalg.norm(X)
    return X / np.linalg.norm(X, axis=1)[:, np.newaxis]


def create_label_map(unique_labels, colors):
    """
    Create a label map for plots with unique labels and colors
    """
    return {label: color for label, color in zip(unique_labels, colors)}


def get_device():
    """
    Get the device to use for computation
    """
    if torch.cuda.is_available():
        # If CUDA is available, select the first CUDA device
        device = torch.device("cuda:0")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    # Check for MPS availability on supported macOS devices (requires PyTorch 1.12 or newer)
    elif torch.backends.mps.is_available():
        # If MPS is available, use MPS device
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) device")
    else:
        # Fallback to CPU if neither CUDA nor MPS is available
        device = torch.device("cpu")
        print("Using CPU")
    return device


def extract_number(filename):
    """
    Extract the number from a filename
    """
    match = re.search(r'layer_(\d+)', filename)
    return int(match.group(1)) if match else 0


# def distance_point_to_direction(point, direction):
#    # Ensure direction is not a zero vector
#    direction = np.array(direction)
#    point = np.array(point)

#    # Projection of point onto direction
#    proj = (np.dot(point, direction) / np.dot(direction, direction)) * direction
#
#    # Perpendicular vector
#    perpendicular_vector = point - proj
#
#    # Distance is the magnitude of the perpendicular vector
#    distance = np.linalg.norm(perpendicular_vector, ord=-1)
#    return dist.chebyshev(point, v=normalize(direction))
# def augment_token_list(tokens):
#     """
#     Augment the token list by adding upper case, capitalized, and space separated versions of the tokens
#     """
#     output = tokens.copy()
#     output.extend([t.upper() for t in tokens])
#     output.extend([t.capitalize() for t in tokens])
#     output.extend([" %s" % t for t in output])
#     return list(set(output))


# def return_ids(tokenizer, tokens):
#     """
#     Convert tokens to their corresponding ids (in the vocabulary)
#     """
#     output = []
#     for t in tokens:
#         res = tokenizer.encode(t)
#         for r in res:
#             if r not in tokenizer(" ").input_ids:
#                 output.append(r)
#     return list(set(output))
