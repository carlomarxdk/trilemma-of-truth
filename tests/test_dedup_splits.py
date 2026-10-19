from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import polars as pl
import pytest

HERE = Path(__file__).resolve().parents[1] / "datasets" / "disjointed"
sys.path.insert(0, str(HERE))

import dedup_splits as dd  # noqa: E402

COLUMNS = ["statement", "object_1", "object_2", "correct_object_2", "correct", "negation", "real_object",
           "is_true", "is_false", "is_neither", "multiclass_label", "category", "source_file"]


# ---------- normalizers ----------


@pytest.mark.parametrize("variants", [
    ["Parkinson's disease", "Parkinsons disease", "Parkinson disease", "Parkinson’s disease"],
    ["anaemia", "anemia", "anemias", "Anaemias"],
    ["oedema", "edema"],
    ["haemorrhage", "hemorrhage"],
    ["hairy cell leukaemia", "hairy cell leukemia"],
    ["bacterial infection", "bacterial infections"],
    ["abscess", "abscesses"],
    ["virus", "viruses"],
    ["allergy", "allergies"],
])
def test_key_condition_merges_variants(variants):
    assert len({dd.key_condition(v) for v in variants}) == 1


@pytest.mark.parametrize("word", ["diabetes", "herpes", "arthritis", "measles", "status", "aids"])
def test_key_condition_keeps_invariants(word):
    assert dd.key_condition(word) == word


def test_key_condition_keeps_distinct_conditions_apart():
    assert dd.key_condition("diabetes") != dd.key_condition("diabete insipidus")
    assert dd.key_condition("anxiety disorder") != dd.key_condition("bipolar disorder")


@pytest.mark.parametrize("a, b", [
    ("Prednisolone phosphate", "Prednisolone"),
    ("Isosorbide mononitrate", "Isosorbide dinitrate"),
    ("Mometasone furoate", "Mometasone"),
    ("Polyethylene glycol 400", "Polyethylene glycol"),
    ("Gallium citrate Ga-67", "Gallium chloride Ga-67"),
    ("Urea C-14", "Urea C-13"),
    ("Pentobarbital sodium", "Pentobarbital"),
])
def test_key_drug_strips_salts_numbers_isotopes(a, b):
    assert dd.key_drug(a) == dd.key_drug(b)


@pytest.mark.parametrize("a, b", [("ABT-501", "ABT-510"), ("Omeprazole", "Esomeprazole"), ("LY-2", "LY-3")])
def test_key_drug_keeps_research_codes_and_distinct_drugs_apart(a, b):
    ka, kb = dd.key_drug(a), dd.key_drug(b)
    assert ka != kb
    assert not dd.token_prefix(ka, kb)


def test_key_drug_keeps_leading_cation():
    assert dd.key_drug("Sodium") == "sodium"
    assert dd.key_drug("Naproxen sodium") == "naproxen"
    assert dd.key_drug("Calcium carbonate") == "calcium carbonate"
    assert dd.token_prefix(dd.key_drug("Calcium carbonate"), dd.key_drug("Calcium"))  # collide, listed for review
    assert dd.key_drug("Calcium chloride") == dd.key_drug("Calcium")
    a, b = dd.key_drug("Sodium bicarbonate"), dd.key_drug("Potassium bicarbonate")
    assert a != b and not dd.token_prefix(a, b)


def test_token_prefix():
    assert dd.token_prefix("insulin", "insulin glargine")
    assert dd.token_prefix("insulin glargine", "insulin")
    assert not dd.token_prefix("insulin", "insulin")  # equal keys are the '-key' rule
    assert not dd.token_prefix("insulin", "insulinoma")  # whole tokens only


@pytest.mark.parametrize("a, b", [
    ("Los Ángeles", "Los Angeles"),
    ("the United States, AZ", "United States"),
    ("the Philippines", "Philippines"),
    ("São Paulo", "Sao Paulo"),
])
def test_key_location_merges_variants(a, b):
    assert dd.key_location(a) == dd.key_location(b)


def test_drug_rule_matches_key_and_prefix():
    protected = {"insulin glargine": {("test", "Insulin glargine")}, "prednisolone": {("cal", "Prednisolone")}}
    assert dd.DRUG.matches("Prednisolone phosphate", protected) == [("drug-key", "cal", "Prednisolone")]
    assert dd.DRUG.matches("Insulin", protected) == [("drug-prefix", "test", "Insulin glargine")]
    assert dd.COND.matches("insulin", {"insulin glargine": {("test", "x")}}) == []  # no prefix rule for conditions


def test_largest_remainder():
    assert dd.largest_remainder({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    assert dd.largest_remainder({"a": 6, "b": 3}, 3) == {"a": 2, "b": 1}
    assert sum(dd.largest_remainder({"a": 7, "b": 5, "c": 2}, 9).values()) == 9


def counts(fa, fn, ta, tn, na, nn):
    return {"F/A": fa, "F/N": fn, "T/A": ta, "T/N": tn, "N/A": na, "N/N": nn}


def test_quotas_exact_by_default():
    assert dd.quotas(counts(242, 162, 161, 242, 202, 202)) == counts(*[161] * 6)
    assert dd.quotas(counts(5, 5, 5, 0, 5, 5)) == counts(*[0] * 6)  # a missing cell empties the split


def test_quotas_with_polarity_tolerance():
    q = dd.quotas(counts(242, 162, 161, 242, 202, 202), Fraction(1, 10))
    sizes = {lab: q[f"{lab}/A"] + q[f"{lab}/N"] for lab in "FTN"}
    assert set(sizes.values()) == {402}  # 161 / 403 is just under 40%, so 402 per class
    assert q["T/A"] == 161 and q["N/A"] == q["N/N"] == 201  # as close to 50 / 50 as the data allows
    assert all(abs(Fraction(q[f"{lab}/A"], 402) - Fraction(1, 2)) <= Fraction(1, 10) for lab in "FTN")


def test_balance_errors():
    df = pl.DataFrame({"multiclass_label": ["0", "0", "1", "1", "2", "2"], "negation": ["0", "1"] * 3})
    assert dd.balance_errors(df, Fraction(0)) == []
    assert dd.balance_errors(df.head(5), Fraction(0)) != []


# ---------- end to end ----------


def row(statement, o1, o2, label, neg, source):
    real = label != 2
    correct = int(label == 1)
    return {"statement": statement, "object_1": o1, "object_2": o2, "correct_object_2": "x",
            "correct": str(correct), "negation": str(neg), "real_object": str(int(real)),
            "is_true": str(int(label == 1)), "is_false": str(int(label == 0)), "is_neither": str(int(label == 2)),
            "multiclass_label": str(label), "category": "indications", "source_file": source}


def split_rows(split, drug, conds, extra=()):
    """One row per (label x negation) cell per condition, plus `extra` rows."""
    rows = []
    for i, cond in enumerate(conds):
        for label in (0, 1, 2):
            for neg in (0, 1):
                src = "med_indications_synthetic" if label == 2 else "med_indications"
                rows.append(row(f"{split} {i} {label} {neg}", f"{drug}{label}" if label == 2 else drug,
                                f"{cond}{label}" if label == 2 else cond, label, neg, src))
    return rows + list(extra)


@pytest.fixture
def toy():
    splits = {
        "train": split_rows("train", "Prednisolone", ["anaemia", "gout"]),  # anaemia ~ test 'anemias'
        "cal": split_rows("cal", "Budesonide", ["asthma"])
        + split_rows("cal2", "Mometasone furoate", ["rhinitis"]),  # ~ test 'Mometasone'
        "test": split_rows("test", "Mometasone", ["anemias", "eczema"]),
    }
    return {s: pl.DataFrame(rows).select(COLUMNS) for s, rows in splits.items()}


def test_dedup_drops_and_balances(toy):
    out, kept, md, fails = dd.dedup("med_indications", toy)
    assert fails == []
    assert out["test"].height == 12  # highest priority keeps everything
    assert "Mometasone furoate" not in out["cal"]["object_1"].to_list()
    assert "anaemia" not in out["train"]["object_2"].to_list()
    assert kept["train"].height == 8  # minus the 4 true/false 'anaemia' rows
    assert kept["cal"].height == 6  # minus the whole 'Mometasone furoate' block ('...furoate2' is a token prefix)
    for s in ("train", "cal"):  # then the neither cells are cut to match
        assert out[s].height == 6
        assert set(dd.cells(out[s]).values()) == {1}
    report = "\n".join(md)
    assert "`test:Mometasone` ~ `cal:Mometasone furoate` (drug-key)" in report
    assert "`test:anemias` ~ `train:anaemia` (cond-key)" in report


def test_dedup_is_deterministic(toy):
    a, b = dd.dedup("med_indications", toy, seed=3), dd.dedup("med_indications", toy, seed=3)
    assert all(a[0][s].equals(b[0][s]) for s in dd.SPLITS) and a[2] == b[2]


def test_dedup_stops_on_inconsistent_labels(toy):
    toy["test"] = toy["test"].with_columns(pl.lit("1").alias("is_true"))
    out, _, md, fails = dd.dedup("med_indications", toy)
    assert fails == ["inconsistent labels"]
    assert out is toy  # nothing changed
