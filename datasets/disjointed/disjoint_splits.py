"""Helpers for the entity-disjoint train / cal / test splits of the Trilemma of Truth datasets (`make_splits.py`).

Reads the local CSVs in datasets/ (the same statements and labels as the HF release). Contents: loading a datapack,
entity keys, negation-flip candidates, and the entity graph that METIS cuts. See DOCS.md for the full pipeline.

    import disjoint_splits as ds                     # from datasets/disjointed/
    df, n_dupes = ds.load("city_locations")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pymetis

DATA_DIR = Path(__file__).resolve().parents[1]  # datasets/
DATAPACKS = {  # as in configs/datapack/
    "city_locations": ["city_locations", "city_locations_synthetic"],
    "med_indications": ["med_indications", "med_indications_synthetic"],
    "word_definitions": [
        "word_instances",
        "word_types",
        "word_synonyms",
        "word_types_synthetic",
        "word_instances_synthetic",
        "word_synonyms_synthetic",
    ],
}
DATASETS = list(DATAPACKS)
COLUMNS = [
    "statement",
    "object_1",
    "object_2",
    "correct_object_2",
    "correct",
    "negation",
    "real_object",
    "is_true",
    "is_false",
    "is_neither",
    "multiclass_label",
    "category",
    "source_file",
]
SPLITS = ["train", "cal", "test"]
TARGET = np.array([0.55, 0.20, 0.25])
SIZE_TOL = 0.02  # absolute, on the share of kept statements
INCORRECT_WEIGHT = 2  # METIS edge weight of incorrect-pair statements (correct pairs: 1)
LABELS = {0: "false", 1: "true", 2: "neither"}
TEXT_COLS = [
    "statement",
    "object_1",
    "object_2",
    "correct_object_2",
    "category",
    "source_file",
]  # never parsed as missing
FOLD_STATES = {
    "city_locations"
}  # datapacks whose objects carry a state suffix ('the United States, AZ')


# --------------------------------------------------------------------------- data
def read_csv(path: Path, **kwargs) -> pl.DataFrame:
    """CSV reader that never parses text as missing: the word "nan" (object_2 of 'Grandmother is not a synonym of a
    nan.') stays text, and an empty text field is "" (polars 2.0 has no missing_utf8_is_empty_string). Extra kwargs
    go to pl.read_csv, e.g. infer_schema=False."""
    head = pl.read_csv(path, n_rows=0, infer_schema=False)
    text = [c for c in TEXT_COLS if c in head.columns]
    df = pl.read_csv(
        path, null_values=[], schema_overrides={c: pl.String for c in text}, **kwargs
    )
    return df.with_columns(pl.col(text).fill_null(""))


def entity_key(col: str, fold_states: bool) -> pl.Expr:
    """Entity key of `col`, shared by subjects and objects: ignores case, whitespace, hyphens, apostrophes and periods;
    with fold_states, a trailing US state code is dropped ('the United States, AZ' -> 'theunitedstates')."""
    key = (
        pl.col(col).cast(pl.String).fill_null("nan").str.strip_chars().str.to_lowercase()
    )
    if fold_states:
        key = key.str.replace(
            r",\s*[a-z]{2}$", ""
        )  # 'the united states, az' -> 'the united states'
    return key.str.replace_all(
        r"[\s\-'.]", ""
    )  # 'dock worker' / 'dock-worker' / 'dockworker'


def entity_keys(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Adds the entity keys of object_1 / object_2 as `_e1` / `_e2`."""
    fold = name in FOLD_STATES
    return df.with_columns(
        entity_key("object_1", fold).alias("_e1"),
        entity_key("object_2", fold).alias("_e2"),
    )


def entities(df: pl.DataFrame) -> set:
    """All entity keys of df (subjects and objects); needs `_e1` / `_e2`."""
    return set(df["_e1"]) | set(df["_e2"])


def load(name: str) -> tuple[pl.DataFrame, int]:
    """All statements of a datapack from the local CSVs, with the HF label columns (0 false, 1 true, 2 neither),
    entity keys `_e1` / `_e2` and `augmented` = 0. Duplicate statements are dropped (first kept); returns
    (statements, number of duplicates dropped)."""
    flag = lambda c: (pl.col(c).str.to_lowercase() == "true").cast(pl.Int64)
    frames = [
        read_csv(DATA_DIR / f"{f}.csv", infer_schema=False)  # all strings; the word "nan" stays text
        .select(
            "statement",
            "object_1",
            "object_2",
            "correct_object_2",
            "correct",
            "negation",
            "real_object",
            "category",
        )
        .with_columns(
            flag("correct"),
            flag("negation"),
            flag("real_object"),
            pl.lit(f).alias("source_file"),
        )
        for f in DATAPACKS[name]
    ]
    df = (
        pl.concat(frames)
        .with_columns(
            (pl.col("correct") * pl.col("real_object")).alias("is_true"),
            ((1 - pl.col("correct")) * pl.col("real_object")).alias("is_false"),
            (1 - pl.col("real_object")).alias("is_neither"),
        )
        .with_columns(
            (pl.col("is_true") + 2 * pl.col("is_neither")).alias("multiclass_label")
        )
        .select(COLUMNS)
    )
    n = df.height
    df = df.unique("statement", keep="first", maintain_order=True)
    df = entity_keys(df, name).with_columns(pl.lit(0, dtype=pl.Int64).alias("augmented"))
    return df, n - df.height


def negation_candidates(df: pl.DataFrame) -> pl.DataFrame:
    """Missing negation counterparts (true <-> false; neither stays neither). Only kept if the rewrite is invertible.
    They are candidates: the pipeline decides which ones to add."""
    neg = pl.col("negation") == 0
    flipped = (
        pl.when(neg)
        .then(pl.col("statement").str.replace(" is ", " is not ", literal=True))
        .otherwise(pl.col("statement").str.replace(" is not ", " is ", literal=True))
    )
    back = (
        pl.when(neg)
        .then(pl.col("_new").str.replace(" is not ", " is ", literal=True))
        .otherwise(pl.col("_new").str.replace(" is ", " is not ", literal=True))
    )
    cand = (
        df.with_columns(flipped.alias("_new"))
        .filter((pl.col("_new") != pl.col("statement")) & (back == pl.col("statement")))
        .filter(~pl.col("_new").is_in(df["statement"].implode()))
        .unique("_new", keep="first", maintain_order=True)
    )
    return cand.with_columns(
        pl.col("_new").alias("statement"),
        (1 - pl.col("negation")).alias("negation"),
        pl.col("is_false").alias(
            "correct"
        ),  # true <-> false flip; neither stays correct=0 like the originals
        pl.col("is_false").alias("is_true"),  # both 0 for neither
        pl.col("is_true").alias("is_false"),
        pl.when(pl.col("multiclass_label") == 2)
        .then(2)
        .otherwise(1 - pl.col("multiclass_label"))
        .cast(pl.Int64)
        .alias("multiclass_label"),
        pl.lit(1, dtype=pl.Int64).alias("augmented"),
    ).drop("_new")


# --------------------------------------------------------------------------- graph
@dataclass
class Part:
    """One entity graph: entities are nodes, <subject, object> pairs are weighted edges."""

    name: str
    df: pl.DataFrame  # the graph's statements, with _gid = index of their pair in gpair
    entities: np.ndarray  # entity keys
    gpair: np.ndarray  # (G, 2) entity ids of each <subject, object> pair
    metis_graph: dict  # CSR input for pymetis


def metis_graph(n: int, gpair: np.ndarray, gweight: np.ndarray) -> dict:
    """CSR graph for pymetis (vertex weight = weighted degree, edge weight = pair weight). gpair rows are unique, a <= b.
    Self-loops and zero-weight pairs are left out of the adjacency."""
    mask = (gpair[:, 0] != gpair[:, 1]) & (gweight > 0)
    a, b, w = gpair[mask, 0], gpair[mask, 1], gweight[mask]
    src, dst, ww = (
        np.concatenate([a, b]),
        np.concatenate([b, a]),
        np.concatenate([w, w]).astype(int),
    )
    order = np.lexsort((dst, src))
    deg = np.bincount(gpair.ravel(), weights=np.repeat(gweight, 2), minlength=n)
    return {
        "adjacency": pymetis.CSRAdjacency(
            np.searchsorted(src[order], np.arange(n + 1)), dst[order]
        ),
        "eweights": ww[order],
        "vweights": np.maximum(deg, 1).astype(int),
    }


def build_part(name: str, df: pl.DataFrame, composition: bool) -> Part:
    """Entity graph of df. With composition=True (true/false statements), incorrect-pair statements weigh
    INCORRECT_WEIGHT so that the cut does not keep mostly correct pairs; augmented rows weigh 0."""
    ents, inv = np.unique(
        np.concatenate([df["_e1"].to_numpy(), df["_e2"].to_numpy()]), return_inverse=True
    )
    e1, e2 = inv[: df.height], inv[df.height :]
    pair = np.stack([np.minimum(e1, e2), np.maximum(e1, e2)], 1)
    upair, gid = np.unique(pair, axis=0, return_inverse=True)
    df = df.with_columns(pl.Series("_gid", gid))
    aug = df["augmented"].to_numpy() == 1
    correct_pair = df["correct"].to_numpy() != df["negation"].to_numpy()
    w = np.where(correct_pair | (not composition), 1, INCORRECT_WEIGHT) * ~aug
    gweight = np.bincount(gid, weights=w, minlength=len(upair))
    return Part(name, df, ents, upair, metis_graph(len(ents), upair, gweight))
