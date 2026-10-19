"""Builds the entity-disjoint train / cal / test splits of the Trilemma of Truth datapacks, end to end.

Steps (see DOCS.md): 1. load, 2. METIS split, 3. candidate statements, 4. balance, 5. checks, 6. near-duplicate
dedup + rebalance (dedup_splits.py), 7. write. Nothing is written unless every check passes.

    uv run python datasets/disjointed/make_splits.py                       # all datapacks
    uv run python datasets/disjointed/make_splits.py city_locations --out-dir /tmp/splits

Writes per datapack: outputs/<name>-disjoint/{train,cal,test}.csv (one file per split), entity_to_split.csv,
excluded.parquet, and reports/<name>.md.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import Counter
from pathlib import Path

import dedup_splits as dd
import disjoint_splits as ds
import numpy as np
import polars as pl
import pymetis

HERE = Path(__file__).resolve().parent
TARGET, SIZE_TOL = ds.TARGET, ds.SIZE_TOL  # 55/20/25, +-2 points
SEEDS, UFACTORS, N_ITER = range(10), [30, 100, 300], 4  # METIS seeds, imbalance tolerances (x/1000), re-weightings
AUG_CAP = 0.30  # true/false statements added: at most 30% of a split's true/false statements
SEED = 0  # with a fixed row order everywhere, the same SEED gives the same splits
REAL_CELLS = [(1, 0), (1, 1), (0, 0), (0, 1)]  # (label, negation): true 'is', true 'is not', false 'is', false 'is not'
is_true_false = pl.col("multiclass_label") < 2
is_neither = pl.col("multiclass_label") == 2
split_name = pl.col("_split").replace_strict(dict(enumerate(ds.SPLITS)), default=None)


def graph_groups(name: str) -> dict[str, pl.Expr]:
    """Entity graphs that are split separately and then merged. word_definitions: 'instances' share almost no words
    with 'types'/'synonyms', so a joint split puts nearly all of them into one split; splitting them on their own
    keeps every statement type in every split."""
    if name == "word_definitions":
        instances = pl.col("category") == "instances"
        return {
            "true_false/instances": is_true_false & instances,
            "true_false/types+synonyms": is_true_false & ~instances,
            "neither": ~is_true_false,
        }
    return {"true_false": is_true_false, "neither": ~is_true_false}


# --------------------------------------------------------------------------- 2. METIS split
def metis_split(part, tpwgts, seed, ufactor):
    _, labels = pymetis.part_graph(
        3, **part.metis_graph, tpwgts=list(tpwgts),
        options=pymetis.Options(seed=seed, ufactor=ufactor, ncuts=4, niter=20),
    )
    return np.asarray(labels)


def kept_rows(part, entity_split):
    g = entity_split[part.gpair]  # split of both entities of every pair
    pair_split = np.where(g[:, 0] == g[:, 1], g[:, 0], -1)
    return pair_split[part.df["_gid"].to_numpy()]  # every statement inherits its pair's split


def best_split(part):
    """Target fractions are re-weighted a few times so the *kept* statements follow 55 / 20 / 25; the run keeping the
    most statements within +-2 points wins."""
    best = None
    for seed in SEEDS:
        for ufactor in UFACTORS:
            t = TARGET.copy()
            for _ in range(N_ITER):
                ent = metis_split(part, t, seed, ufactor)
                rows = kept_rows(part, ent)
                kept = rows >= 0
                frac = np.bincount(rows[kept], minlength=3) / kept.sum()
                ok = bool(np.all(np.abs(frac - TARGET) <= SIZE_TOL))
                if best is None or (ok, kept.mean()) > (best["ok"], best["kept"]):
                    best = {"ok": ok, "kept": kept.mean(), "entity_split": ent, "seed": seed, "ufactor": ufactor}
                if ok:
                    break
                t = t * TARGET / np.maximum(frac, 1e-3)  # ask METIS for more where too little was kept
                t /= t.sum()
    return best


def composition(frame):
    """Rows per split and (label x negation) cell."""
    k = frame.filter(pl.col("_split") >= 0)
    return (
        k.group_by("_split")
        .agg(
            pl.len().alias("n"),
            *[((pl.col("multiclass_label") == l) & (pl.col("negation") == n)).sum()
              .alias(f"{ds.LABELS[l]}_{'not' if n else 'is'}") for l in (1, 0, 2) for n in (0, 1)],
        )
        .with_columns((pl.col("n") / pl.col("n").sum()).round(3).alias("share"))
        .with_columns(split_name)
        .sort("_split")
    )


# --------------------------------------------------------------------------- 1-5. split and balance
def build(name: str, log: list[str]) -> dict:
    """Steps 1-5 (formerly metis_splits.ipynb). Returns the balanced statements (`final`, with `_split`), the
    original statements (`orig`), the METIS entity -> split map and the neither statements dropped for having both
    forms in a split."""
    # 1. load and build the entity graphs
    df, n_dupes = ds.load(name)
    parts = {
        g: ds.build_part(f"{name}/{g}", df.filter(f), composition=g.startswith("true_false"))
        for g, f in graph_groups(name).items()
    }
    assert not ds.entities(df.filter(~is_neither)) & ds.entities(df.filter(is_neither))
    log.append(f"{df.height} statements ({n_dupes} duplicates dropped)")
    for g, part in parts.items():
        log.append(f"{g}: entities={len(part.entities)}, pairs={len(part.gpair)}, statements={part.df.height}")

    # 2. METIS split; an entity that two graphs put into different splits goes to the split where it has the most
    # statements (statements that end up across splits are dropped)
    votes = {}  # entity key -> Counter(split -> number of its statements in that graph)
    for g, part in parts.items():
        best = best_split(part)
        load = Counter(part.df["_e1"].to_list() + part.df["_e2"].to_list())
        for e, s in zip(part.entities.tolist(), best["entity_split"].tolist(), strict=True):
            votes.setdefault(e, Counter())[s] += load[e]
        log.append(f"{g}: kept {best['kept']:.1%} (seed {best['seed']}, ufactor {best['ufactor']}, "
                   f"sizes ok: {best['ok']})")
    split_of = {e: c.most_common(1)[0][0] for e, c in votes.items()}  # entity key -> split
    log.append(f"entities put into different splits by different graphs: {sum(len(c) > 1 for c in votes.values())}")

    def entity_split(col):
        return pl.col(col).replace_strict(split_of, default=None)  # split of the entity in `col`; None = in no split

    def usable_split(frame):
        """Split a statement belongs to: both entities in the same split; -1 otherwise (or if an entity is unknown)."""
        s1, s2 = entity_split("_e1"), entity_split("_e2")
        return frame.with_columns(pl.when(s1.is_not_null() & (s1 == s2)).then(s1).otherwise(-1).alias("_split"))

    orig = usable_split(df).with_columns(pl.lit(None, dtype=pl.String).alias("aug_type"))
    log.append(f"original statements kept: {(orig['_split'] >= 0).sum()}/{orig.height} "
               f"({(orig['_split'] >= 0).mean():.1%})")

    # 3. candidate statements: negation flips and correct pairs, rejected if their label contradicts correct_object_2
    def correct_objects(value):
        """Correct objects listed in correct_object_2. City keeps a state code with its country:
        'the United States, KY, the United States, TX, Guyana' -> ['the United States, KY', 'the United States, TX',
        'Guyana']."""
        out = []
        for item in (value or "").split(","):
            item = item.strip()
            if name in ds.FOLD_STATES and out and len(item) == 2 and item.isupper():
                out[-1] = f"{out[-1]}, {item}"
            elif item:
                out.append(item)
        return out

    # every <subject, correct object> key pair in the data, to reject generated statements whose label contradicts it
    _key = ds.entity_key("o", name in ds.FOLD_STATES)
    correct_keys = (
        df.filter(~is_neither)
        .select("_e1", pl.col("correct_object_2").map_elements(correct_objects, return_dtype=pl.List(pl.String))
                .alias("o"))
        .explode("o")
        .drop_nulls()
        .select("_e1", _key.alias("_e2"))
        .unique()
        .with_columns(pl.lit(True).alias("_is_correct"))
    )

    def consistent(frame):
        """Keep true/false statements whose label agrees with correct_object_2 (neither rows pass through)."""
        claims_correct = (pl.col("multiclass_label") == 1) == (pl.col("negation") == 0)  # "object is correct"
        return (
            frame.join(correct_keys, on=["_e1", "_e2"], how="left", maintain_order="left")
            .filter(is_neither | (claims_correct == pl.col("_is_correct").fill_null(False)))
            .drop("_is_correct")
        )

    # no neither flips: a neither pair keeps a single form
    flips = consistent(
        usable_split(ds.negation_candidates(df.filter(~is_neither))).with_columns(
            pl.lit("negation_flip").alias("aug_type"))
    )

    def correct_pair_candidates():
        fold = name in ds.FOLD_STATES  # city: US states fold into their country
        rows = []
        for r in df.filter(~is_neither).iter_rows(named=True):
            text, obj = r["statement"], r["object_2"]
            if text.count(obj) != 1 or " is " not in text:
                continue
            for c in correct_objects(r["correct_object_2"]):
                if not c or c == obj:
                    continue
                aff = text.replace(obj, c)
                if r["negation"] == 1:
                    aff = aff.replace(" is not ", " is ", 1)
                neg = aff.replace(" is ", " is not ", 1)
                base = {k: r[k] for k in ds.COLUMNS} | {"object_2": c, "_e1": r["_e1"]}
                rows.append(base | {"statement": aff, "negation": 0, "correct": 1, "is_true": 1, "is_false": 0,
                                    "multiclass_label": 1})
                rows.append(base | {"statement": neg, "negation": 1, "correct": 0, "is_true": 0, "is_false": 1,
                                    "multiclass_label": 0})
        cand = (
            pl.DataFrame(rows)
            .with_columns(ds.entity_key("object_2", fold).alias("_e2"))
            .join(df.select("statement"), on="statement", how="anti", maintain_order="left")
            .unique(["_e1", "_e2", "statement"], keep="first", maintain_order=True)
            .join(df.select("_e1", "_e2").unique(), on=["_e1", "_e2"], how="anti", maintain_order="left")
        )  # new pairs only
        cand = cand.with_columns(entity_split("_e1").alias("_split"), entity_split("_e2").alias("_c_split")).filter(
            pl.col("_split").is_not_null() & (pl.col("_c_split").is_null() | (pl.col("_c_split") == pl.col("_split")))
        )
        # a new entity (in no split) is given to one split only: the one proposing it most often
        home = (
            cand.filter(pl.col("_c_split").is_null())
            .group_by("_e2", "_split")
            .len()
            .sort("len", "_split", descending=[True, False])
            .unique("_e2", keep="first")
            .select("_e2", pl.col("_split").alias("_home"))
        )
        cand = cand.join(home, on="_e2", how="left").filter(
            pl.col("_home").is_null() | (pl.col("_home") == pl.col("_split")))
        cand = cand.filter(pl.len().over("_e1", "_e2") == 2)  # both statements of the pair
        return cand.select(ds.COLUMNS + ["_e1", "_e2", "_split"]).with_columns(
            pl.lit(1, dtype=pl.Int64).alias("augmented"), pl.lit("correct_pair").alias("aug_type"))

    correct = consistent(correct_pair_candidates())

    def avail(f):
        return dict(f.filter(pl.col("_split") >= 0).group_by("_split").len().sort("_split").iter_rows())

    log.append(f"negation flips available per split: {avail(flips)}")
    log.append(f"correct-pair statements available per split: {avail(correct)}")

    # 4. balance every split
    rng = np.random.default_rng(SEED)
    existing_statements = set(df["statement"]) | set(flips["statement"]) | set(correct["statement"])
    existing_pairs = set(zip(df["_e1"], df["_e2"], strict=True))

    def sample(frame, n):
        n = max(0, min(n, frame.height))
        return frame.sample(n, seed=int(rng.integers(1 << 31))) if n else frame.clear()

    # statement types (column `category`) inside each label in the full data; additions and drops follow this mix
    cat_share = {
        l: dict(g.group_by("category").len().sort("category").with_columns(pl.col("len") / g.height).iter_rows())
        for (l,), g in df.group_by("multiclass_label", maintain_order=True)
    }

    def by_category(frame, n, label):
        # n rows of frame spread over categories like the full data's mix for `label`; shortfalls filled from the rest
        n = max(0, min(n, frame.height))
        if n == 0:
            return frame.clear()
        got = pl.concat([sample(frame.filter(pl.col("category") == c), round(n * sh))
                         for c, sh in cat_share[label].items()])
        if got.height >= n:
            return got.head(n)
        return pl.concat([got, sample(frame.join(got, on=frame.columns, how="anti"), n - got.height)])

    def heaviest_first(frame, n):
        """n statements of the subjects with the most statements (random among equals)."""
        return (
            frame.with_columns(pl.len().over("_e1").alias("_load"), pl.Series("_r", rng.random(frame.height)))
            .sort("_load", "_r", descending=[True, False])
            .head(max(0, n))
            .drop("_load", "_r")
        )

    def neither_shuffles(pool, negation, n, s):
        """n new neither statements: a neither subject of split s with another neither object of split s."""
        objects = pool.select("category", "object_2", "_e2").unique(maintain_order=True)
        by_cat = {c: g.rows() for (c,), g in objects.group_by("category", maintain_order=True)}
        templates = pool.rows(named=True)
        if not templates:
            return pool.clear()
        out, tries = [], 0
        while len(out) < n and tries < 200 * max(n, 1):
            tries += 1
            r = templates[rng.integers(len(templates))]
            _, obj, key = by_cat[r["category"]][rng.integers(len(by_cat[r["category"]]))]
            if key == r["_e2"] or (r["_e1"], key) in existing_pairs or r["statement"].count(r["object_2"]) != 1:
                continue
            text = r["statement"].replace(r["object_2"], obj)
            text = text.replace(" is not ", " is ", 1)
            if negation:
                text = text.replace(" is ", " is not ", 1)
            if text in existing_statements:
                continue
            existing_statements.add(text)
            existing_pairs.add((r["_e1"], key))
            out.append(r | {"statement": text, "object_2": obj, "_e2": key, "negation": negation, "augmented": 1,
                            "aug_type": "neither_shuffle", "_split": s})
        return pl.DataFrame(out, schema=pool.schema) if out else pool.clear()

    def cell(f, l, n):
        return f.filter((pl.col("multiclass_label") == l) & (pl.col("negation") == n))

    added, dropped, dropped_conflict = [], [], []
    for s in range(3):
        kept = orig.filter(pl.col("_split") == s)
        real = kept.filter(~is_neither)
        count = {c: cell(real, *c).height for c in REAL_CELLS}
        budget = int(np.floor(AUG_CAP / (1 - AUG_CAP) * real.height))
        m = max(max(count.values()), int(np.ceil(TARGET[s] * df.height / 6)))  # largest cell, at least original size

        # true/false: correct pairs (+1 true 'is', +1 false 'is not')
        pairs = correct.filter(pl.col("_split") == s)
        pair_keys = pairs.select("_e1", "_e2", "category").unique(["_e1", "_e2"], maintain_order=True)
        k = max(0, min(m - count[1, 0], m - count[0, 1], pair_keys.height, budget // 2))
        chosen = pairs.join(by_category(pair_keys, k, 1).drop("category"), on=["_e1", "_e2"])
        added.append(chosen)
        count[1, 0] += k
        count[0, 1] += k
        budget -= 2 * k

        # true/false: flips for what is still short
        for c in sorted(REAL_CELLS, key=lambda c: count[c]):
            pool = cell(flips.filter(pl.col("_split") == s), *c)
            take = by_category(pool, min(m - count[c], budget), c[0])
            added.append(take)
            count[c] += take.height
            budget -= take.height

        # neither: target (true + false) / 2, half 'is', half 'is not', each statement type at its full-data share
        per_polarity = round(sum(count.values()) / 4)
        nei = kept.filter(is_neither)
        # a neither pair with both forms in this split: keep one form (alternate which, so 'is' / 'is not' stay even)
        both = (
            nei.group_by("_e1", "_e2", maintain_order=True)
            .agg(pl.col("negation").n_unique().alias("_forms"))
            .filter(pl.col("_forms") == 2)
            .select("_e1", "_e2")
        )
        both = sample(both, both.height).with_columns((pl.int_range(pl.len()) % 2).alias("negation"))  # form to drop
        conflict = nei.join(both, on=["_e1", "_e2", "negation"]).select("statement")
        dropped_conflict.append(conflict)
        nei = nei.join(conflict, on="statement", how="anti")
        for n in (0, 1):
            for cat, share in cat_share[2].items():
                want = round(per_polarity * share)
                have = cell(nei.filter(pl.col("category") == cat), 2, n)
                if have.height > want:
                    dropped.append(heaviest_first(have, have.height - want).select("statement"))
                else:
                    added.append(neither_shuffles(nei.filter(pl.col("category") == cat), n, want - have.height, s))

    generated = pl.concat([a.select(orig.columns) for a in added if a.height], how="vertical_relaxed")
    drop_conflict = pl.concat(dropped_conflict)["statement"]
    drop = pl.concat(dropped + dropped_conflict)["statement"]
    final = pl.concat(
        [orig.filter((pl.col("_split") >= 0) & ~pl.col("statement").is_in(drop.implode())), generated],
        how="vertical_relaxed",
    )
    log.append(f"dropped neither statements: {drop.len()} (of which {drop_conflict.len()} because the pair had "
               "both forms)")
    log.append(f"added: {dict(generated.group_by('aug_type').len().sort('aug_type').iter_rows())}")

    # 5. checks: no entity in two splits, no duplicate statements, a neither pair in one form per split
    ents = {sn: ds.entities(final.filter(pl.col("_split") == i)) for i, sn in enumerate(ds.SPLITS)}
    overlap = {f"{a} & {b}": len(ents[a] & ents[b]) for i, a in enumerate(ds.SPLITS) for b in ds.SPLITS[i + 1:]}
    assert not any(overlap.values()), overlap
    assert final["statement"].is_unique().all()
    both_forms = (
        final.filter(is_neither)
        .group_by("_split", "_e1", "_e2")
        .agg(pl.col("negation").n_unique().alias("forms"))
        .filter(pl.col("forms") == 2)
    )
    assert both_forms.height == 0, f"{both_forms.height} neither pairs with both forms in a split"
    return {"df": df, "orig": orig, "final": final, "split_of": split_of, "drop_conflict": drop_conflict}


# --------------------------------------------------------------------------- 6-7. dedup and write
def as_text(frame: pl.DataFrame) -> pl.DataFrame:
    """The frame as `ds.read_csv(path, infer_schema=False)` would read it back from CSV (all strings)."""
    return pl.read_csv(io.BytesIO(frame.write_csv().encode()), infer_schema=False, null_values=[]).with_columns(
        pl.col(ds.TEXT_COLS).fill_null(""))


def excluded_statements(b: dict, splits: dict[str, pl.DataFrame], dedup_kept: dict[str, pl.DataFrame] | None):
    """Every original statement that is in no split file, with the reason."""
    orig, final, split_of = b["orig"], b["final"], b["split_of"]
    written = pl.concat([splits[s].select("statement") for s in ds.SPLITS])["statement"]
    balanced = final["statement"]
    after_drops = pl.concat([dedup_kept[s].select("statement") for s in ds.SPLITS])["statement"] \
        if dedup_kept is not None else balanced
    named = lambda c: pl.col(c).replace_strict(split_of, default=None).replace_strict(  # noqa: E731
        dict(enumerate(ds.SPLITS)), default=None)
    return (
        orig.filter(~pl.col("statement").is_in(written.implode()))
        .with_columns(
            pl.when(pl.col("_split") < 0).then(pl.lit("cut"))
            .when(pl.col("statement").is_in(b["drop_conflict"].implode())).then(pl.lit("dropped (both forms)"))
            .when(~pl.col("statement").is_in(balanced.implode())).then(pl.lit("dropped (balancing)"))
            .when(~pl.col("statement").is_in(after_drops.implode())).then(pl.lit("dropped (near-duplicate)"))
            .otherwise(pl.lit("dropped (rebalancing after dedup)"))
            .alias("reason"),
            named("_e1").alias("subject_split"),
            named("_e2").alias("object_split"),
        )
        .select(ds.COLUMNS + ["reason", "subject_split", "object_split"])
    )


def run(name: str, out_dir: Path, report_dir: Path, dedup: bool = True) -> bool:
    """All steps for one datapack. Writes the outputs only if every check passes."""
    log = []
    b = build(name, log)
    final = b["final"]
    # per-split shuffle as in the notebook; the dedup works on the same all-string frames that a CSV read gives
    splits = {
        sn: as_text(final.filter(pl.col("_split") == s).select(ds.COLUMNS).sample(fraction=1.0, shuffle=True, seed=s))
        for s, sn in enumerate(ds.SPLITS)
    }
    balanced = {sn: splits[sn].height for sn in ds.SPLITS}
    md = [f"# Splits: {name}", "", "## Build", "", *(f"- {line}" for line in log), "",
          "Composition after balancing:", "", "```", str(composition(final)), "```", ""]
    fails, kept = [], None
    if dedup:
        splits, kept, dedup_md, fails = dd.dedup(name, splits)
        md += ["## Near-duplicate dedup", "", *dedup_md, ""]

    excluded = excluded_statements(b, splits, kept)
    # every original statement is in exactly one split file or in excluded
    written = pl.concat([splits[s].select("statement") for s in ds.SPLITS])["statement"]
    orig_statements = set(b["orig"]["statement"])
    in_files = orig_statements & set(written)
    if in_files | set(excluded["statement"]) != orig_statements or in_files & set(excluded["statement"]):
        fails.append("original statements missing from both the split files and excluded")
    if not written.is_unique().all():
        fails.append("duplicate statements across split files")
    md += ["## Excluded original statements", "", "```",
           str(excluded.group_by("reason").len().sort("reason")), "```", ""]
    md += ["## Result", "", *dd.table(["split", "after balancing", "written"],
                                      [[s, balanced[s], splits[s].height] for s in ds.SPLITS]), ""]
    md += ["**PASS**"] if not fails else ["**FAIL: nothing written**", "", *(f"- {f}" for f in fails)]

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{name}.md").write_text("\n".join(md) + "\n")
    print(f"{name}: " + ", ".join(f"{s} {splits[s].height}" for s in ds.SPLITS) + (" PASS" if not fails else
                                                                                       f" FAIL {fails}"))
    if fails:
        return False

    dst = out_dir / f"{name}-disjoint"
    dst.mkdir(parents=True, exist_ok=True)
    for s in ds.SPLITS:
        splits[s].write_csv(dst / f"{s}.csv")
    # entity -> split of the entities in the written files
    keyed = {s: ds.entities(ds.entity_keys(splits[s], name)) for s in ds.SPLITS}
    (pl.DataFrame([(e, s) for s in ds.SPLITS for e in keyed[s]], schema=["entity_key", "split"], orient="row")
     .sort("split", "entity_key").write_csv(dst / "entity_to_split.csv"))
    excluded.write_parquet(dst / "excluded.parquet")
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("datapacks", nargs="*", default=ds.DATASETS, help=f"any of {ds.DATASETS} (default: all)")
    p.add_argument("--out-dir", type=Path, default=HERE / "outputs", help="writes <out-dir>/<name>-disjoint/")
    p.add_argument("--report-dir", type=Path, default=HERE / "reports")
    p.add_argument("--no-dedup", action="store_true", help="skip step 6 (near-duplicate dedup + rebalance)")
    a = p.parse_args(argv)
    if unknown := set(a.datapacks) - set(ds.DATASETS):
        p.error(f"unknown datapacks: {sorted(unknown)}")
    ok = [run(n, a.out_dir, a.report_dir, dedup=not a.no_dedup) for n in a.datapacks]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
