"""Shared plotting constants and label/order mappings."""

from __future__ import annotations

from pathlib import Path

import matplotlib
from cmcrameri import cm

# Models, datapacks, probes (matching notebook)
MODELS: list[str] = [
    "qwen-2.5-14b",
    "qwen-2.5-7b",
    "mistral-7B-v0.3",
    "gemma-2-9b",
    "gemma-7b",
    "llama-3-8b",
    "llama-3.2-3b",
    "_qwen-2.5-14b",
    "_qwen-2.5-7b",
    "_mistral-7B-v0.3",
    "_gemma-2-9b",
    "_gemma-7b",
    "_llama-3.1-8b",
    "_llama-3-8b-med",
    "_llama-3.1-8b-bio",
    "_llama-3.2-3b",
]

DATAPACKS: list[str] = ["city_locations", "med_indications", "word_definitions"]
PROBES: list[str] = ["zero_shot", "mean_diff", "ttpd", "spca", "svm", "sawmil"]

PROBE_ORDER: list[str] = ["zero_shot", "mean_diff", "ttpd", "spca", "svm", "sawmil"]
DATAPACK_ORDER: list[str] = ["city_locations", "med_indications", "word_definitions"]


TASKS: dict[str, int] = {
    "mean_diff": 3,
    "ttpd": 3,
    "spca": 3,
    "sawmil": -1,
    "svm": -1,
}

MODEL_NAMES: dict[str, str] = {
    "llama-3-8b": "Llama-3-8B",
    "llama-3.2-3b": "Llama-3.2-3B",
    "mistral-7B-v0.3": "Mistral-7B-v0.3",
    "qwen-2.5-7b": "Qwen-2.5-7B",
    "qwen-2.5-14b": "Qwen-2.5-14B",
    "gemma-7b": "Gemma-7B",
    "gemma-2-9b": "Gemma-2-9B",
    "_llama-3.1-8b": "Llama-3.1-8B-Instruct",
    "_llama-3.2-3b": "Llama-3.2-3B-Instruct",
    "_mistral-7B-v0.3": "Mistral-7B-Instruct-v0.3",
    "_qwen-2.5-7b": "Qwen-2.5-7B-Instruct",
    "_qwen-2.5-14b": "Qwen-2.5-14B-Instruct",
    "_gemma-7b": "Gemma-7B-it",
    "_gemma-2-9b": "Gemma-2-9B-it",
    "_llama-3.1-8b-bio": "Bio-Medical-Llama-3-8B",
    "_llama-3-8b-med": "Llama3-Med42-8B",
}

MODEL_SHORTNAMES: dict[str, str] = {
    "llama-3-8b": "Llama-3-8B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "llama-3.2-3b": "Llama-3.2-3B",
    "mistral-7B-v0.3": "Mistral-7B-v0.3",
    "qwen-2.5-7b": "Qwen-2.5-7B",
    "qwen-2.5-14b": "Qwen-2.5-14B",
    "gemma-7b": "Gemma-7B",
    "gemma-2-9b": "Gemma-2-9B",
    "_llama-3.1-8b": "Llama-3.1-8B",
    "_llama-3.2-3b": "Llama-3.2-3B",
    "_mistral-7B-v0.3": "Mistral-7B-v0.3",
    "_qwen-2.5-7b": "Qwen-2.5-7B",
    "_qwen-2.5-14b": "Qwen-2.5-14B",
    "_gemma-7b": "Gemma-7B",
    "_gemma-2-9b": "Gemma-2-9B",
    "_llama-3.1-8b-bio": "Bio-Medical-Llama",
    "_llama-3-8b-med": "Llama3-Med42-8B",
}

MODEL_TYPES = {
    "llama-3-8b": "default",
    "llama-3.1-8b": "default",
    "llama-3.2-3b": "default",
    "mistral-7B-v0.3": "default",
    "qwen-2.5-7b": "default",
    "qwen-2.5-14b": "default",
    "gemma-7b": "default",
    "gemma-2-9b": "default",
    "_llama-3.1-8b": "chat",
    "_llama-3.2-3b": "chat",
    "_mistral-7B-v0.3": "chat",
    "_qwen-2.5-7b": "chat",
    "_qwen-2.5-14b": "chat",
    "_gemma-7b": "chat",
    "_gemma-2-9b": "chat",
    "_llama-3.1-8b-bio": "chat",
    "_llama-3-8b-med": "chat",
}

DATASET_NAMES: dict[str, str] = {
    "city_locations": "City Locations",
    "med_indications": "Medical Indications",
    "word_definitions": "Word Definitions",
}

PROBE_NAMES: dict[str, str] = {
    "zero_shot": "Zero-Shot",
    "mean_diff": "MD+CP",
    "ttpd": "TTPD+CP",
    "spca": "sPCA+CP",
    "svm": "SVM",
    "sawmil": "sAwMIL",
}

CONDITION_NAMES = {
    "bag": "Bag-Level",
    "instance": "Instance-Level",
    "instance_tf": "Instance-Level (TF Only)",
}


SAVE_DIR = Path("outputs/figures/summaries")


DATASET_COLOR: dict[str, tuple] = {
    "city_locations": cm.lipariS(2),
    "med_indications": cm.lipariS(4),
    "word_definitions": cm.lipariS(3),
}

CONDITION_COLOR: dict[str, tuple] = {
    "instance": cm.glasgowS(6),
    "bag": cm.glasgowS(10),
    "instance_tf": cm.glasgowS(2),
}

PROBE_COLOR: dict[str, tuple] = {
    "zero_shot": cm.lipariS(0),
    "mean_diff": cm.lipariS(2),
    "ttpd": cm.lipariS(4),
    "spca": cm.lipariS(3),
}

### OPTS
SAVEFIG_OPTS = {
    "format": "pdf",
    "bbox_inches": "tight",
    "transparent": False,
    "dpi": 300,
    "pad_inches": 0.02,
}


def _setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,  # Base font size
            "axes.labelsize": 8,  # Axis label: 7-9 pt
            "axes.titlesize": 9,  # Title: 9-10 pt
            "xtick.labelsize": 7,  # Tick labels: 6-7 pt
            "ytick.labelsize": 7,
            "legend.fontsize": 7,  # Legend: 6-8 pt
            "figure.titlesize": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
