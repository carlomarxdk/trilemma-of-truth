from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

HERE = Path(__file__).resolve().parents[1] / "datasets" / "disjointed"
sys.path.insert(0, str(HERE))

import disjoint_splits as ds  # noqa: E402
import make_splits as ms  # noqa: E402

pytestmark = pytest.mark.skipif(
    not all((ds.DATA_DIR / f"{f}.csv").exists() for f in ds.DATAPACKS["city_locations"]),
    reason="datapack CSVs not available",
)


# ---------- NA-safe reader ----------


def test_read_csv_keeps_na_tokens_as_text(tmp_path):
    p = tmp_path / "risky.csv"
    p.write_text("statement,object_2,correct,multiclass_label\na,nan,TRUE,1\nb,NA,FALSE,0\nc,null,TRUE,1\n"
                 "d,None,TRUE,1\ne,,TRUE,1\n")
    df = ds.read_csv(p)
    assert df["object_2"].to_list() == ["nan", "NA", "null", "None", ""]
    assert df["multiclass_label"].dtype == pl.Int64  # non-text columns keep their dtypes


def test_grandmother_row_keeps_nan():
    df, _ = ds.load("word_definitions")
    row = df.filter(pl.col("statement") == "Grandmother is not a synonym of a nan.")
    assert row["object_2"].to_list() == ["nan"]


# ---------- end to end ----------


@pytest.fixture(scope="module")
def city(tmp_path_factory):
    out = tmp_path_factory.mktemp("splits")
    assert ms.run("city_locations", out, out / "reports")
    return out / "city_locations-disjoint", out / "reports" / "city_locations.md"


def test_one_file_per_split(city):
    dst, _ = city
    assert sorted(p.name for p in dst.iterdir()) == ["cal.csv", "entity_to_split.csv", "excluded.parquet",
                                                     "test.csv", "train.csv"]


def test_splits_are_disjoint_and_complete(city):
    dst, report = city
    splits = {s: ds.read_csv(dst / f"{s}.csv", infer_schema=False) for s in ds.SPLITS}
    assert all(splits[s].columns == ds.COLUMNS for s in ds.SPLITS)  # no `augmented`
    keys = {s: ds.entities(ds.entity_keys(splits[s], "city_locations")) for s in ds.SPLITS}
    assert not keys["train"] & keys["cal"] and not keys["train"] & keys["test"] and not keys["cal"] & keys["test"]
    written = set(pl.concat(splits.values())["statement"])
    excluded = set(pl.read_parquet(dst / "excluded.parquet")["statement"])
    orig = set(ds.load("city_locations")[0]["statement"])
    assert orig == (orig & written) | excluded and not written & excluded
    assert report.read_text().rstrip().endswith("**PASS**")
