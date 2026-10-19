"""Removes near-duplicate entities across the entity-disjoint splits, then rebalances (step 6 of make_splits.py).

`disjoint_splits.entity_key` keeps exact entities apart; this catches what it misses: plural and British spelling
variants of conditions and words, salt / ester / isotope forms of drugs, and accents in city names.

Policy: the splits are walked from the highest priority down (default test > cal > train). A row is dropped if one of
its entities collides with an entity kept in a higher-priority split; rows are never moved between splits. Each split
is then downsampled to equal true / false / neither counts with "is" / "is not" at 50 / 50 inside each class
(within POLARITY_TOL for city_locations), keeping its source_file mix.

    import dedup_splits as dd
    out, kept, report, fails = dd.dedup("med_indications", splits)   # splits: {split: all-string DataFrame}
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, permutations

import disjoint_splits as ds
import numpy as np
import polars as pl

SPLITS = ds.SPLITS
PRIORITY = ["test", "cal", "train"]  # first keeps all its rows; train loses rows so cal and test keep theirs
SEED = 0
CELL = ["multiclass_label", "negation"]
SOURCE_TOL = 0.5  # pp, allowed change of a split's source_file shares
POLARITY_TOL = {"city_locations": Fraction(1, 10)}  # affirmative share per class: 0.5 +- tol (default 0: exact)
LABELS = {"0": "F", "1": "T", "2": "N"}

# --------------------------------------------------------------------------- normalization
BRIT = [("anaem", "anem"), ("haem", "hem"), ("oedem", "edem"), ("oesoph", "esoph"), ("paediat", "pediat"),
        ("tumour", "tumor"), ("diarrhoea", "diarrhea"), ("leukaem", "leukem"), ("ischaem", "ischem"),
        ("oestr", "estr"), ("aetiol", "etiol"), ("faec", "fec"), ("colour", "color"), ("behaviour", "behavior")]
NO_SINGULARIZE = re.compile(r"(is|us|ss|ics)$")  # arthritis, status, abscess, ...
INVARIANT = {"diabetes", "herpes", "rabies", "scabies", "measles", "mumps", "rickets", "aids", "shingles",
             "species", "series", "caries", "feces"}
SALT = {"phosphate", "sulfate", "sulphate", "sodium", "potassium", "calcium", "hydrochloride", "hcl", "hydrobromide",
        "mononitrate", "dinitrate", "furoate", "propionate", "dipropionate", "acetate", "glucuronide", "sulfide",
        "sulfadiazine", "maleate", "mesylate", "tartrate", "citrate", "succinate", "besylate", "bromide", "chloride",
        "fumarate", "valerate", "oleate", "monohydrate", "dihydrate", "trihydrate", "anhydrous", "ester"}
ISOTOPE = re.compile(r"^[a-z]{1,2}-?\d{1,3}m?$")  # c-13, ga-67, tc-99m
REVIEW = {"aluminum", "aluminium", "magnesium", "strontium", "choline", "silver", "sodium", "potassium", "calcium",
          "zinc", "lithium", "iron"}  # shared drug tokens that may be false positives: listed, still dropped


def ascii_lower(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def singular(w: str) -> str:
    if w in INVARIANT or len(w) <= 3:
        return w
    if w.endswith("ies"):
        return w[:-3] + "y"
    if re.search(r"(ch|sh|x|ss|us|is)es$", w):  # boxes, abscesses, viruses
        return w[:-2]
    if w.endswith("s") and not NO_SINGULARIZE.search(w):
        return w[:-1]
    return w


def key_condition(s: str) -> str:
    """Conditions and words: ASCII, lower case, no punctuation, singular tokens, American spelling.
    "Parkinson's" / "Parkinsons" / "Parkinson" -> "parkinson"."""
    s = re.sub(r"['`]s\b", "s", ascii_lower(s))  # the curly apostrophe is already gone (not ASCII)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    for a, b in BRIT:
        s = s.replace(a, b)
    return " ".join(singular(t) for t in s.split())


def key_drug(s: str) -> str:
    """Drugs: salt / ester tokens, standalone numbers and isotope tags after the first token dropped. The first token is
    kept, so a leading cation stays ('Calcium carbonate' -> 'calcium', 'Naproxen sodium' -> 'naproxen'). Hyphens are
    kept, so research codes stay distinct (ABT-501 vs ABT-510)."""
    head, *rest = re.sub(r"[^a-z0-9 \-]", " ", ascii_lower(s)).split() or [""]
    return " ".join([head, *(t for t in rest if t not in SALT and not t.isdigit() and not ISOTOPE.match(t))])


def key_location(s: str) -> str:
    """Cities and countries: ASCII, lower case, no leading "the " / "city of ", no US state code, no punctuation."""
    s = re.sub(r",\s*[a-z]{2}$", "", ascii_lower(s))  # 'the united states, az' -> 'the united states'
    s = re.sub(r"^(the |city of )+", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def token_prefix(a: str, b: str) -> bool:
    """Different keys where one is a whole-token prefix of the other ('insulin' / 'insulin glargine')."""
    ta, tb = a.split(), b.split()
    n = min(len(ta), len(tb))
    return a != b and n > 0 and ta[:n] == tb[:n]


@dataclass(frozen=True)
class Rule:
    """A normalizer; two entities collide if their keys are equal ('<name>-key') or, with prefix, one key is a
    whole-token prefix of the other ('<name>-prefix')."""
    name: str
    key: Callable[[str], str]
    prefix: bool = False

    def matches(self, value: str, protected: dict[str, set]) -> list[tuple[str, str, str]]:
        """(rule fired, split, value) of every protected entity that `value` collides with."""
        k = self.key(value)
        out = [(f"{self.name}-key", s, v) for s, v in sorted(protected.get(k, ()))]
        if self.prefix:
            out += [(f"{self.name}-prefix", s, v) for pk in sorted(protected) if token_prefix(k, pk)
                    for s, v in sorted(protected[pk])]
        return out


COND, DRUG, LOC = Rule("cond", key_condition), Rule("drug", key_drug, prefix=True), Rule("loc", key_location)
GROUPS = {  # datapack -> [(columns sharing one entity space, rule)]
    "city_locations": [(("object_1", "object_2"), LOC)],
    "med_indications": [(("object_1",), DRUG), (("object_2",), COND)],
    "word_definitions": [(("object_1", "object_2"), COND)],  # words are subjects in one split, objects in another
}


# --------------------------------------------------------------------------- helpers
def ints(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.select(pl.col(cols).cast(pl.Int64))


def values(df: pl.DataFrame, cols: tuple[str, ...]) -> list[str]:
    return sorted({v for c in cols for v in df[c].drop_nulls().to_list()})


# --------------------------------------------------------------------------- checks
def label_errors(df: pl.DataFrame) -> dict[str, int]:
    d = ints(df, ["correct", "real_object", "is_true", "is_false", "is_neither", "multiclass_label"])
    real = d["real_object"] == 1
    return {
        "one-hot labels": int((d["is_true"] + d["is_false"] + d["is_neither"] != 1).sum()),
        "multiclass == T + 2N": int((d["multiclass_label"] != d["is_true"] + 2 * d["is_neither"]).sum()),
        "neither <=> real_object == 0": int((d["is_neither"] != 1 - d["real_object"]).sum()),
        "is_true <=> correct (real rows)": int(((d["is_true"] != d["correct"]) & real).sum()),
    }


def overlap_rows(S: dict[str, pl.DataFrame], name: str) -> list[tuple[str, str, int]]:
    """(check, split pair, number of shared values) for exact, lower-case, cross-role, entity-key and normalized keys."""
    lower = lambda xs: {x.lower() for x in xs}  # noqa: E731
    col = lambda s, c: set(S[s][c].drop_nulls().to_list())  # noqa: E731
    pair = lambda s: set(S[s].select("object_1", "object_2").rows())  # noqa: E731
    rows = []
    for a, b in combinations(SPLITS, 2):
        ab = f"{a}-{b}"
        for c in ["object_1", "object_2", "statement"]:
            rows += [(f"{c} exact", ab, len(col(a, c) & col(b, c))),
                     (f"{c} lower", ab, len(lower(col(a, c)) & lower(col(b, c))))]
        rows += [("pair exact", ab, len(pair(a) & pair(b))),
                 ("pair lower", ab, len({tuple(map(str.lower, p)) for p in pair(a)}
                                        & {tuple(map(str.lower, p)) for p in pair(b)}))]
        ka, kb = (ds.entities(ds.entity_keys(S[s], name)) for s in (a, b))
        rows.append(("entity_key (both roles)", ab, len(ka & kb)))
        for cols, rule in GROUPS[name]:
            na, nb = ({rule.key(v) for v in values(S[s], cols)} for s in (a, b))
            shared = na & nb
            if rule.prefix:
                shared |= {x for x in na for y in nb if token_prefix(x, y)}
            rows.append((f"{rule.name} key ({'+'.join(cols)})", ab, len(shared)))
    for a, b in permutations(SPLITS, 2):
        rows += [("cross-role object_1 -> object_2 exact", f"{a}-{b}", len(col(a, "object_1") & col(b, "object_2"))),
                 ("cross-role object_1 -> object_2 lower", f"{a}-{b}",
                  len(lower(col(a, "object_1")) & lower(col(b, "object_2"))))]
    return rows


def hygiene(S: dict[str, pl.DataFrame]) -> list[tuple[str, str, int]]:
    """Duplicate statements and nulls per split (nulls only for columns that have any)."""
    rows = []
    for s, df in S.items():
        rows.append(("duplicate statements", s, df.height - df["statement"].n_unique()))
        rows += [(f"nulls in {c}", s, n) for c, n in df.null_count().row(0, named=True).items() if n]
    return rows


def cells(df: pl.DataFrame) -> dict[str, int]:
    """Rows per (label x negation) cell, as 'T/A' (true, affirmative), 'F/N' (false, negated), ..."""
    counts = {k: v for k, v in df.group_by(CELL).len().select(pl.concat_str(CELL, separator="|"), "len").iter_rows()}
    return {f"{LABELS[lab]}/{'N' if neg == '1' else 'A'}": counts.get(f"{lab}|{neg}", 0)
            for lab in LABELS for neg in "01"}


def source_shares(df: pl.DataFrame) -> dict[str, float]:
    return {s: 100 * n / df.height for s, n in sorted(df["source_file"].value_counts().iter_rows())}


# --------------------------------------------------------------------------- dedup
def find_drops(S: dict[str, pl.DataFrame], name: str, priority: list[str]):
    """Walks the splits from the highest priority down and drops every row with an entity that collides with one kept
    in a higher-priority split. Returns (kept splits, collisions, {split: {rule: rows dropped}})."""
    groups = GROUPS[name]
    protected = [defaultdict(set) for _ in groups]  # per group: key -> {(split, value)}
    kept, collisions, dropped = {}, [], {}
    for split in priority:
        df, hit = S[split], {}
        for g, (cols, rule) in enumerate(groups):
            bad = defaultdict(set)
            for v in values(df, cols):
                for fired, ks, kv in rule.matches(v, protected[g]):
                    collisions.append((ks, kv, split, v, fired))
                    bad[fired].add(v)
            for fired, vs in sorted(bad.items()):
                m = df.select(pl.any_horizontal(pl.col(c).is_in(sorted(vs)) for c in cols)).to_series()
                hit[fired] = hit[fired] | m if fired in hit else m
        mask = pl.Series([False] * df.height)
        for m in hit.values():
            mask = mask | m
        kept[split] = df.filter(~mask)
        dropped[split] = {**{f: int(m.sum()) for f, m in sorted(hit.items())}, "total": int(mask.sum())}
        for g, (cols, rule) in enumerate(groups):
            for v in values(kept[split], cols):
                protected[g][rule.key(v)].add((split, v))
    return {s: kept[s] for s in SPLITS}, collisions, dropped


def largest_remainder(counts: dict[str, int], n: int) -> dict[str, int]:
    """Splits n over the keys of counts in proportion to counts (ties broken by key)."""
    total = sum(counts.values())
    exact = {k: n * c / total for k, c in counts.items()}
    q = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (q[k] - exact[k], k))[: n - sum(q.values())]:
        q[k] += 1
    return q


def quotas(counts: dict[str, int], tol: Fraction = Fraction(0)) -> dict[str, int]:
    """Rows to keep per (label x negation) cell (keys as in `cells`): the same number n for every class, the largest
    n for which each class can have an affirmative share within 0.5 +- tol; that share as close to 0.5 as possible."""
    largest = min(counts[f"{lab}/A"] + counts[f"{lab}/N"] for lab in LABELS.values())
    for n in range(largest, 0, -1):
        q = {}
        for lab in LABELS.values():
            a, neg = counts[f"{lab}/A"], counts[f"{lab}/N"]
            lo = max(n - neg, -(-(Fraction(1, 2) - tol) * n // 1))  # ceil
            hi = min(a, (Fraction(1, 2) + tol) * n // 1)  # floor
            if lo > hi:
                break
            q[f"{lab}/A"] = int(min(max(n // 2, lo), hi))
            q[f"{lab}/N"] = n - q[f"{lab}/A"]
        else:
            return q
    return dict.fromkeys(counts, 0)


def rebalance(df: pl.DataFrame, seed: int, tol: Fraction = Fraction(0)) -> pl.DataFrame:
    """Downsamples to the `quotas` of the split; inside each cell, rows are allocated over source_file in proportion
    to the cell's own mix. Deterministic for a given seed and input."""
    rng = np.random.default_rng(seed)
    df = df.sort("statement")
    quota = quotas(cells(df), tol)
    parts = []
    for (lab, neg), q_cell in zip(((lab, neg) for lab in LABELS for neg in "01"), quota.values(), strict=True):
        cell = df.filter((pl.col("multiclass_label") == lab) & (pl.col("negation") == neg))
        if not q_cell:
            continue
        for src, q in sorted(largest_remainder(dict(cell["source_file"].value_counts().iter_rows()), q_cell).items()):
            rows = cell.filter(pl.col("source_file") == src)
            parts.append(rows[np.sort(rng.choice(rows.height, q, replace=False))])
    if not parts:
        return df.clear()
    out = pl.concat(parts)
    return out[rng.permutation(out.height)]


def balance_errors(df: pl.DataFrame, tol: Fraction) -> list[str]:
    """Unequal class sizes, or an affirmative share outside 0.5 +- tol."""
    c = cells(df)
    sizes = {lab: c[f"{lab}/A"] + c[f"{lab}/N"] for lab in LABELS.values()}
    errors = [f"class sizes {sizes}"] if len(set(sizes.values())) > 1 else []
    return errors + [f"{lab} affirmative {c[f'{lab}/A']}/{n}" for lab, n in sizes.items()
                     if n and abs(Fraction(c[f"{lab}/A"], n) - Fraction(1, 2)) > tol]


def related_drugs(S: dict[str, pl.DataFrame], name: str) -> list[tuple[str, str, str, str]]:
    """Not handled: distinct drugs with a shared stem in different splits (one key a substring of the other, at least
    6 characters: omeprazole / esomeprazole). Listed only."""
    out = []
    for cols, rule in GROUPS[name]:
        if not rule.prefix:
            continue
        keys = {s: {rule.key(v): v for v in values(S[s], cols)} for s in SPLITS}
        for a, b in permutations(SPLITS, 2):
            out += [(a, keys[a][x], b, keys[b][y]) for x in sorted(keys[a]) for y in sorted(keys[b])
                    if len(x) >= 6 and x != y and x in y]
    return out


def needs_review(collisions: list) -> list:
    """Drug collisions whose shared tokens are all inorganic cations or generic words."""
    out = []
    for ks, kv, ls, lv, fired in collisions:
        if fired.startswith(DRUG.name):
            a, b = key_drug(kv).split(), key_drug(lv).split()
            if set(min(a, b, key=len)) <= REVIEW:
                out.append((ks, kv, ls, lv, fired))
    return out


# --------------------------------------------------------------------------- report
def table(header: list[str], rows: list) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(str(x) for x in r) + " |" for r in rows)]


def check_table(rows: list[tuple[str, str, int]]) -> list[str]:
    """Checks as rows, split pairs as columns."""
    pairs = list(dict.fromkeys(p for _, p, _ in rows))
    by = defaultdict(dict)
    for c, p, n in rows:
        by[c][p] = n
    return table(["check", *pairs], [[c, *(by[c].get(p, "") for p in pairs)] for c in by])


def dedup(name: str, S: dict[str, pl.DataFrame], priority: list[str] = PRIORITY, seed: int = SEED):
    """Dedups the splits of one datapack (all columns as strings, as read with infer_schema=False). Returns (rebalanced
    splits, splits after the drops, report as markdown lines, failed post-checks); writes nothing."""
    md = [f"Priority (first keeps its rows): {' > '.join(priority)}. Seed: {seed}.", "",
          "Normalization: " + "; ".join(f"{'+'.join(c)}: `{r.name}` key{' + token prefix' if r.prefix else ''}"
                                        for c, r in GROUPS[name]) + ".", ""]

    md += ["### Inventory", ""]
    md += table(["split", "rows", *cells(S["train"])], [[s, S[s].height, *cells(S[s]).values()] for s in SPLITS])

    errors = {s: label_errors(df) for s, df in S.items()}
    md += ["", "### Pre-checks", "", "Label consistency (violating rows):", ""]
    md += table(["check", *SPLITS], [[c, *(errors[s][c] for s in SPLITS)] for c in errors["train"]])
    md += ["", "Overlap across splits (shared values):", "", *check_table(overlap_rows(S, name))]
    md += ["", "Duplicates and nulls:", "", *table(["check", "split", "n"], hygiene(S))]
    if any(n for e in errors.values() for n in e.values()):
        md += ["", "**Stopped: label columns are inconsistent.**"]
        return S, S, md, ["inconsistent labels"]

    kept, collisions, dropped = find_drops(S, name, priority)
    tol = POLARITY_TOL.get(name, Fraction(0))
    out = {s: rebalance(kept[s], seed, tol) for s in SPLITS}

    rules = sorted({f for d in dropped.values() for f in d if f != "total"})
    md += ["", "### Dropped rows", "", "A row hit by two rules counts once per rule, and once in `total`.", ""]
    md += table(["split", "before", *rules, "total", "after drops", "after rebalance"],
                [[s, S[s].height, *(dropped[s].get(r, 0) for r in rules), dropped[s]["total"], kept[s].height,
                  out[s].height] for s in SPLITS])
    md += ["", "### Balance (label x negation)", "",
           f"Target: equal true / false / neither; affirmative share 50 +- {float(tol) * 100:g}% inside each class.", ""]
    md += table(["split", "stage", *cells(S["train"])],
                [[s, stage, *cells(d[s]).values()] for s in SPLITS
                 for stage, d in [("before", S), ("after drops", kept), ("after rebalance", out)]])
    md += ["", "### source_file shares (%)", ""]
    shares = []
    for s in SPLITS:
        before, after = source_shares(S[s]), source_shares(out[s])
        shares += [[s, f, f"{before[f]:.2f}", f"{after.get(f, 0):.2f}", f"{after.get(f, 0) - before[f]:+.2f}"]
                   for f in before]
    md += table(["split", "source_file", "before", "after", "change (pp)"], shares)

    md += ["", f"### Collisions ({len(collisions)})", "", "`keep_split:value ~ lose_split:value` (rule).", ""]
    md += [f"- `{ks}:{kv}` ~ `{ls}:{lv}` ({fired})" for ks, kv, ls, lv, fired in collisions] or ["None."]
    review = needs_review(collisions)
    md += ["", f"### Needs human review ({len(review)})", "",
           "Drug matches whose shared tokens are only cations or generic words "
           f"({', '.join(sorted(REVIEW))}). They may be false positives; the rows were dropped anyway.", ""]
    md += [f"- `{ks}:{kv}` ~ `{ls}:{lv}` ({fired})" for ks, kv, ls, lv, fired in review] or ["None."]
    related = related_drugs(out, name)
    md += ["", "### Not handled", "",
           "Related but distinct drugs (e.g. omeprazole / esomeprazole, chloroquine / hydroxychloroquine) are kept "
           "apart only if their keys collide. No local RxNorm / ATC / MeSH resource exists in the repo, so "
           "concept-level disjointness is not possible here. Candidates left in the output, by a shared-stem "
           "heuristic (one key is a substring of the other, at least 6 characters):", ""]
    md += [f"- `{a}:{x}` / `{b}:{y}`" for a, x, b, y in related] or ["None."]

    post = overlap_rows(out, name)
    fails = [f"{c} {p}: {n}" for c, p, n in post if n and "lower" not in c]
    fails += [f"labels {s}: {c}" for s in SPLITS for c, n in label_errors(out[s]).items() if n]
    fails += [f"balance {s}: {e}" for s in SPLITS for e in balance_errors(out[s], tol)]
    fails += [f"source_file {s} {f}: {d}" for s, f, _, _, d in shares if abs(float(d)) > SOURCE_TOL]
    md += ["", "### Post-checks", "", *check_table(post), "",
           "Overlap counts must be zero for every exact, entity_key and normalized check (lower-case counts are "
           f"informational). True / false / neither must be equal, with an affirmative share of 50 +- {float(tol) * 100:g}% "
           f"inside each class; source_file shares must change by at most {SOURCE_TOL} pp.", ""]
    md += ["**PASS**"] if not fails else ["**FAIL**", "", *(f"- {f}" for f in fails)]

    return out, kept, md, fails
