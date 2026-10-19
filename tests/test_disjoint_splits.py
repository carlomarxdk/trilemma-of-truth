from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_equal

HERE = Path(__file__).resolve().parents[1] / "datasets" / "disjointed"
sys.path.insert(0, str(HERE))

import disjoint_splits as ds  # noqa: E402

CSV_HEADER = "statement,object_1,object_2,correct_object_2,correct,negation,real_object,category\n"


def write_csv(path: Path, rows: list[str]) -> None:
    path.write_text(CSV_HEADER + "\n".join(rows) + "\n")


def frame(rows: list[dict], name: str = "toy") -> pl.DataFrame:
    """Statements with the columns load() produces, built from minimal dicts."""
    df = pl.DataFrame(rows).with_columns(
        pl.col(c).cast(pl.Int64) for c in ("correct", "negation", "real_object")
    )
    df = df.with_columns(
        (pl.col("correct") * pl.col("real_object")).alias("is_true"),
        ((1 - pl.col("correct")) * pl.col("real_object")).alias("is_false"),
        (1 - pl.col("real_object")).alias("is_neither"),
    ).with_columns((pl.col("is_true") + 2 * pl.col("is_neither")).alias("multiclass_label"))
    if "augmented" not in df.columns:
        df = df.with_columns(pl.lit(0, dtype=pl.Int64).alias("augmented"))
    return ds.entity_keys(df, name)


def keys(values: list, fold: bool) -> list[str]:
    return pl.DataFrame({"x": values}).select(ds.entity_key("x", fold))["x"].to_list()


def neighbours(g: dict, v: int) -> tuple[np.ndarray, np.ndarray]:
    xadj, adj = np.asarray(g["adjacency"].adj_starts), np.asarray(g["adjacency"].adjacent)
    return adj[xadj[v]: xadj[v + 1]], np.asarray(g["eweights"])[xadj[v]: xadj[v + 1]]


# ---------- entity keys ----------


def test_entity_key_ignores_case_whitespace_hyphens_apostrophes_periods():
    out = keys(["Dock worker", "dock-worker", " DOCKWORKER ", "St. John's"], fold=False)
    assert out == ["dockworker", "dockworker", "dockworker", "stjohns"]


def test_entity_key_null_becomes_nan_text():
    assert keys([None, "nan"], fold=False) == ["nan", "nan"]


def test_entity_key_fold_states_drops_trailing_state_code():
    vals = ["the United States, AZ", "the United States,TX", "the United States"]
    assert keys(vals, fold=True) == ["theunitedstates"] * 3
    assert keys(vals[:1], fold=False) == ["theunitedstates,az"]


def test_entity_key_fold_states_only_two_letter_suffix():
    assert keys(["Korea, South"], fold=True) == ["korea,south"]


def test_entity_keys_folds_only_for_fold_state_datapacks():
    df = pl.DataFrame({"object_1": ["Phoenix"], "object_2": ["the United States, AZ"]})
    assert ds.entity_keys(df, "city_locations")["_e2"].item() == "theunitedstates"
    assert ds.entity_keys(df, "med_indications")["_e2"].item() == "theunitedstates,az"


def test_entities_unions_subjects_and_objects():
    df = pl.DataFrame({"_e1": ["a", "b"], "_e2": ["b", "c"]})
    assert ds.entities(df) == {"a", "b", "c"}


# ---------- load ----------


@pytest.fixture
def toy_pack(tmp_path, monkeypatch):
    write_csv(tmp_path / "toy_a.csv", [
        "A is B.,A,B,B,TRUE,FALSE,TRUE,c",          # true
        "A is not C.,A,C,B,TRUE,TRUE,TRUE,c",       # true (negated)
        "A is C.,A,C,B,FALSE,FALSE,TRUE,c",         # false
    ])
    write_csv(tmp_path / "toy_b.csv", [
        "Zorp is nan.,Zorp,nan,nan,FALSE,FALSE,FALSE,c",  # neither, literal "nan"
        "A is B.,A,B,B,TRUE,FALSE,TRUE,c",                # duplicate of toy_a row 1
    ])
    monkeypatch.setattr(ds, "DATA_DIR", tmp_path)
    monkeypatch.setitem(ds.DATAPACKS, "toy", ["toy_a", "toy_b"])
    return ds.load("toy")


def test_load_columns_and_dupes(toy_pack):
    df, n_dupes = toy_pack
    assert n_dupes == 1
    assert df.columns == ds.COLUMNS + ["_e1", "_e2", "augmented"]
    assert df["statement"].to_list() == ["A is B.", "A is not C.", "A is C.", "Zorp is nan."]
    assert df["source_file"].to_list() == ["toy_a", "toy_a", "toy_a", "toy_b"]  # first kept
    assert (df["augmented"] == 0).all()


def test_load_labels(toy_pack):
    df, _ = toy_pack
    assert df["multiclass_label"].to_list() == [1, 1, 0, 2]
    assert df["is_true"].to_list() == [1, 1, 0, 0]
    assert df["is_false"].to_list() == [0, 0, 1, 0]
    assert df["is_neither"].to_list() == [0, 0, 0, 1]
    assert_equal((df["is_true"] + df["is_false"] + df["is_neither"]).to_numpy(), 1)


def test_load_keeps_nan_as_text(toy_pack):
    df, _ = toy_pack
    row = df.row(3, named=True)
    assert row["object_2"] == "nan" and row["_e2"] == "nan"


@pytest.mark.parametrize("name", ds.DATASETS)
def test_load_real_datapack(name):
    files = ds.DATAPACKS[name]
    if not all((ds.DATA_DIR / f"{f}.csv").exists() for f in files):
        pytest.skip("datapack CSVs not available")
    df, n_dupes = ds.load(name)
    assert n_dupes >= 0
    assert df["statement"].is_unique().all()
    assert set(df["source_file"].unique()) <= set(files)
    assert set(df["multiclass_label"].unique()) <= set(ds.LABELS)
    assert_equal((df["is_true"] + df["is_false"] + df["is_neither"]).to_numpy(), 1)
    assert df.select(pl.col("_e1", "_e2").null_count()).row(0) == (0, 0)
    # where a CSV ships its own labels, load() must reproduce them
    for f in files:
        raw = pl.read_csv(ds.DATA_DIR / f"{f}.csv", infer_schema=False)
        if "multiclass_label" not in raw.columns:
            continue
        joined = df.filter(pl.col("source_file") == f).join(
            raw.select("statement", pl.col("multiclass_label").cast(pl.Int64).alias("csv_label"))
            .unique("statement", keep="first"),
            on="statement",
        )
        assert_equal(joined["multiclass_label"].to_numpy(), joined["csv_label"].to_numpy())


# ---------- negation candidates ----------


def test_negation_candidates_flip_labels():
    df = frame([
        {"statement": "A is B.", "object_1": "A", "object_2": "B", "correct": 1, "negation": 0, "real_object": 1},
        {"statement": "A is not C.", "object_1": "A", "object_2": "C", "correct": 1, "negation": 1, "real_object": 1},
        {"statement": "Zorp is Q.", "object_1": "Zorp", "object_2": "Q", "correct": 0, "negation": 0, "real_object": 0},
    ])
    cand = ds.negation_candidates(df)
    got = {r["statement"]: r for r in cand.iter_rows(named=True)}
    assert set(got) == {"A is not B.", "A is C.", "Zorp is not Q."}

    assert got["A is not B."]["negation"] == 1
    assert got["A is not B."]["multiclass_label"] == 0
    assert (got["A is not B."]["is_true"], got["A is not B."]["is_false"], got["A is not B."]["correct"]) == (0, 1, 0)

    assert got["A is C."]["negation"] == 0
    assert got["A is C."]["multiclass_label"] == 0  # true 'A is not C.' -> false 'A is C.'
    assert (got["A is C."]["is_true"], got["A is C."]["is_false"], got["A is C."]["correct"]) == (0, 1, 0)

    nei = got["Zorp is not Q."]
    assert (nei["multiclass_label"], nei["is_true"], nei["is_false"], nei["is_neither"], nei["correct"]) == (2, 0, 0, 1, 0)

    assert (cand["augmented"] == 1).all()
    assert cand["multiclass_label"].dtype == pl.Int64
    assert cand.columns == df.columns


def test_negation_candidates_keep_entity_keys():
    df = frame([{"statement": "A is B.", "object_1": "A", "object_2": "B", "correct": 1, "negation": 0, "real_object": 1}])
    cand = ds.negation_candidates(df)
    assert cand.select("_e1", "_e2").row(0) == ("a", "b")


def test_negation_candidates_skip_existing_counterpart():
    df = frame([
        {"statement": "A is B.", "object_1": "A", "object_2": "B", "correct": 1, "negation": 0, "real_object": 1},
        {"statement": "A is not B.", "object_1": "A", "object_2": "B", "correct": 0, "negation": 1, "real_object": 1},
    ])
    assert ds.negation_candidates(df).height == 0


def test_negation_candidates_skip_non_invertible_and_unchanged():
    df = frame([
        # no " is ": the rewrite is a no-op
        {"statement": "A means B.", "object_1": "A", "object_2": "B", "correct": 1, "negation": 0, "real_object": 1},
        # 'What is A is B.' flips back to 'What is not A is B.' (earlier ' is ' hit first): not invertible
        {"statement": "What is A is not B.", "object_1": "A", "object_2": "B", "correct": 1, "negation": 1, "real_object": 1},
        # two ' is ', but the round trip still restores the original: kept
        {"statement": "This is what A is.", "object_1": "A", "object_2": "B", "correct": 1, "negation": 0, "real_object": 1},
    ])
    assert ds.negation_candidates(df)["statement"].to_list() == ["This is not what A is."]


def test_negation_candidates_dedupe_new_statements():
    row = {"statement": "A is B.", "object_1": "A", "object_2": "B", "correct": 1, "negation": 0, "real_object": 1}
    df = frame([row, {**row, "statement": "A is B."}])
    assert ds.negation_candidates(df)["statement"].to_list() == ["A is not B."]


# ---------- metis graph ----------


def test_metis_graph_symmetric_csr():
    gpair = np.array([[0, 1], [0, 2], [1, 2]])
    g = ds.metis_graph(3, gpair, np.array([1.0, 2.0, 3.0]))
    xadj = np.asarray(g["adjacency"].adj_starts)
    assert_equal(xadj, [0, 2, 4, 6])
    for v, (nbrs, ws) in enumerate(neighbours(g, u) for u in range(3)):
        assert list(nbrs) == sorted(nbrs)
        for u, w in zip(nbrs, ws, strict=True):
            back_n, back_w = neighbours(g, u)
            assert w == back_w[list(back_n).index(v)]
    assert_equal(neighbours(g, 0)[1], [1, 2])
    assert_equal(g["vweights"], [3, 4, 5])


def test_metis_graph_drops_self_loops_and_zero_weights():
    gpair = np.array([[0, 0], [0, 1], [1, 2]])
    g = ds.metis_graph(4, gpair, np.array([5.0, 0.0, 1.0]))
    assert_equal(np.asarray(g["adjacency"].adj_starts), [0, 0, 1, 2, 2])
    assert_equal(np.asarray(g["adjacency"].adjacent), [2, 1])
    # self-loop still counts twice in the degree; isolated / zero-degree vertices get weight 1
    assert_equal(g["vweights"], [10, 1, 1, 1])
    assert g["eweights"].dtype.kind == "i"


# ---------- build part ----------


@pytest.fixture
def part_df():
    return frame([
        {"statement": "A is B.", "object_1": "A", "object_2": "B", "correct": 1, "negation": 0, "real_object": 1},
        {"statement": "B is not A.", "object_1": "B", "object_2": "A", "correct": 0, "negation": 1, "real_object": 1},
        {"statement": "A is C.", "object_1": "A", "object_2": "C", "correct": 0, "negation": 0, "real_object": 1},
        {"statement": "C is not D.", "object_1": "C", "object_2": "D", "correct": 0, "negation": 1, "real_object": 1},
        {"statement": "D is A.", "object_1": "D", "object_2": "A", "correct": 1, "negation": 0, "real_object": 1, "augmented": 1},
    ])


def edge_weight(part: ds.Part, a: str, b: str) -> int:
    ia, ib = (int(np.flatnonzero(part.entities == x)[0]) for x in (a, b))
    nbrs, ws = neighbours(part.metis_graph, ia)
    hit = np.flatnonzero(nbrs == ib)
    return int(ws[hit[0]]) if hit.size else 0


def test_build_part_structure(part_df):
    part = ds.build_part("toy", part_df, composition=True)
    assert part.name == "toy"
    assert list(part.entities) == ["a", "b", "c", "d"]
    assert part.gpair.shape == (4, 2)  # (a,b) shared by two statements
    assert (part.gpair[:, 0] <= part.gpair[:, 1]).all()
    gid = part.df["_gid"].to_numpy()
    assert gid[0] == gid[1]  # 'A is B' and 'B is not A' are the same unordered pair
    ents = part.entities[part.gpair[gid]]
    e = part.df.select("_e1", "_e2").to_numpy()
    assert_equal(np.sort(ents, 1), np.sort(e, 1))


def test_build_part_composition_weights(part_df):
    part = ds.build_part("toy", part_df, composition=True)
    assert edge_weight(part, "a", "b") == 2  # two correct-pair statements ('B is not A.' is false)
    assert edge_weight(part, "a", "c") == ds.INCORRECT_WEIGHT  # 'A is C' false, not negated -> incorrect pair
    assert edge_weight(part, "c", "d") == 1  # 'C is not D' false + negated -> correct pair
    assert edge_weight(part, "a", "d") == 0  # augmented only: dropped


def test_build_part_without_composition_weights_all_one(part_df):
    part = ds.build_part("toy", part_df, composition=False)
    assert edge_weight(part, "a", "b") == 2
    assert edge_weight(part, "a", "c") == 1
    assert edge_weight(part, "c", "d") == 1
    assert edge_weight(part, "a", "d") == 0


def test_build_part_runs_through_pymetis(part_df):
    import pymetis

    part = ds.build_part("toy", part_df, composition=True)
    g = part.metis_graph
    _, membership = pymetis.part_graph(2, adjacency=g["adjacency"], eweights=g["eweights"], vweights=g["vweights"])
    assert len(membership) == len(part.entities)
