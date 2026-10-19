# Entity-disjoint splits

Builds train / cal / test splits of the three datapacks (`city_locations`, `med_indications`, `word_definitions`)
in which no entity appears in more than one split, not even as a near-duplicate (plural, British spelling, salt form).
They are the `*_disjoint` configs on Hugging Face.

The source CSVs in `datasets/` are only read, never changed. Run everything with one command:

    uv run python datasets/disjointed/make_splits.py              # all three datapacks
    uv run python datasets/disjointed/make_splits.py city_locations --out-dir /tmp/splits

Nothing is written unless every check passes; the run exits non-zero otherwise. Same inputs give the same files.
Tests: `uv run --with pytest python -m pytest tests/test_disjoint_splits.py tests/test_dedup_splits.py tests/test_make_splits.py`.

## Files

| File | What it does |
|---|---|
| `make_splits.py` | The pipeline, steps 1–7 below. Writes `outputs/<name>-disjoint/` and `reports/<name>.md`. |
| `disjoint_splits.py` | Helpers: NA-safe CSV reader, loading, entity keys, negation flips, the METIS graph. Settings shared by all datapacks (`TARGET`, `SIZE_TOL`, `INCORRECT_WEIGHT`). |
| `dedup_splits.py` | Step 6: near-duplicate normalizers, collision rules, rebalancing, post-checks. |
| `HF_README.md` | The Hugging Face dataset card (copy of `hf_release/README.md`; upload manually). |


## Pipeline

1. **Load.** `ds.load(name)` reads all CSVs of the datapack (see `DATAPACKS`) with `ds.read_csv`, which never parses
   text as missing (the word "nan" in `Grandmother is not a synonym of a nan.` stays text). It adds the HF label columns
   (`is_true`, `is_false`, `is_neither`, `multiclass_label`) and drops duplicate statements. Every `object_1` and
   `object_2` gets an entity key (`ds.entity_key`): lower case, without whitespace, hyphens, apostrophes and periods,
   so `dock worker`, `dock-worker` and `dockworker` are one entity. In `city_locations`, US state codes are dropped
   (`the United States, AZ` → `the United States`). Subjects and objects share one set of keys.

2. **Split with METIS.** Each entity is a node and each (subject, object) pair is an edge. True/false and neither
   statements are separate graphs, because they share no entities. In `word_definitions`, `instances` are cut apart
   from `types` + `synonyms`; cut together, nearly all instances end up in one split. In the true/false graphs,
   incorrect pairs weigh `INCORRECT_WEIGHT` (2), correct pairs 1. Otherwise the cut tends to keep mostly correct
   pairs.

   METIS cuts each graph into 3 parts with target sizes 55 / 20 / 25. A statement is kept only if both of its
   entities are in the same part. The script tries 10 seeds × 3 imbalance settings, and re-weights the targets up to
   4 times so that the *kept* statements match 55 / 20 / 25 within ±2 points (`SIZE_TOL`). Among the runs that meet
   the sizes, it keeps the one that keeps the most statements. When graphs are merged, an entity that two graphs put
   into different splits goes to the split where it has the most statements.

3. **Candidate statements.** Two kinds of new true/false statements, built from the same templates:
   - *negation flips*: the missing "is" / "is not" twin of a statement, with the label flipped (true ↔ false);
   - *correct pairs*: subject X with one of its correct objects C from `correct_object_2`, as "X is C" (true) and
     "X is not C" (false).

   A candidate is used only in the split that holds both of its entities. It is rejected if its label contradicts
   `correct_object_2`; for example, a city name that exists in two countries.

4. **Balance each split.** Target: about 1/3 true, 1/3 false, 1/3 neither, and about half "is" / half "is not"
   within each class.
   - True/false: the four cells (true is, true is not, false is, false is not) are filled up to the largest cell, and
     at least to 1/6 of the split's target size. Correct pairs are used first, then flips. Additions are capped at
     `AUG_CAP` (30%) of the split's true/false statements. Original true/false statements are not dropped here (only
     in step 6). In `city_locations` the candidates run out before the cap (raising it changes nothing), so true and
     false stay at about 40 / 60 "is" / "is not" in cal.
   - Neither: if a pair appears as both "X is Y" and "X is not Y" in a split, one form is dropped. The class is then
     brought to (true + false) / 2: extra statements are dropped, subjects with the most statements first; missing
     ones are added as *shuffles*, which pair a synthetic subject with another synthetic object of the same split and
     category. Neither statements are never flipped.


5. **Checks.** The run stops (`assert`) if any entity is in two splits, if a statement appears twice, or if a
   neither pair has both forms in one split.

6. **Near-duplicate dedup** (`dedup_splits.py`; skip with `--no-dedup`). Entity keys miss variants, so a second,
   stricter key is computed per datapack:
   - `med_indications`: drugs (`object_1`) drop salt / ester tokens, numbers and isotope tags after the first word
     (`Prednisolone phosphate` → `prednisolone`; `Calcium carbonate` keeps `calcium`), and also collide on a
     whole-word prefix (`insulin` / `insulin glargine`). Conditions (`object_2`) are singular, American spelling
     (`anaemias` → `anemia`), without possessives (`Parkinson's` → `parkinson`).
   - `word_definitions`: the condition key, on subjects and objects together.
   - `city_locations`: ASCII, no leading "the" / "city of", no state code (`Los Ángeles` = `Los Angeles`).

   Splits are walked test → cal → train; a row is dropped if one of its entities collides with an entity kept in an
   earlier split, so train loses rows and cal / test keep theirs. Rows are never moved between splits. Each split is
   then downsampled to equal true / false / neither, with "is" / "is not" at exactly 50 / 50 inside each class
   (`city_locations`: 50 ± 10, `POLARITY_TOL`), keeping its `source_file` mix. Post-checks: zero overlap on exact
   strings, entity keys and the stricter keys, label consistency, the balance, and `source_file` shares within
   0.5 pp. Matches that share only a cation or a generic word (e.g. `Potassium acetate` / `Potassium chloride`) are
   dropped as well and listed under "Needs human review" in the report; related but distinct drugs
   (omeprazole / esomeprazole) may stay in different splits and are listed under "Not handled".

7. **Write.** See Outputs below. Every original statement is in a split file or in `excluded.parquet`, with a reason:
   - `cut`: its subject and object are in different splits;
   - `dropped (both forms)`: a neither pair with both forms in one split;
   - `dropped (balancing)`: a neither statement removed in step 4;
   - `dropped (near-duplicate)`: an entity collides with one in another split (step 6);
   - `dropped (rebalancing after dedup)`: removed to restore the balance after step 6.

## Outputs

`outputs/<name>-disjoint/`:

| File | Content |
|---|---|
| `train.csv`, `cal.csv`, `test.csv` | The final statements of each split, one file per split. Columns: `ds.COLUMNS` (no `augmented` column). |
| `entity_to_split.csv` | `entity_key`, `split` for every entity in the split files. |
| `excluded.parquet` | Excluded original statements, with `reason`, `subject_split` and `object_split` (the METIS split of each entity). |

`reports/<name>.md`: build log, composition, every near-duplicate collision, rows dropped per rule, balance and
`source_file` tables, post-checks.

## Settings

| Setting | Where | Value | Meaning |
|---|---|---|---|
| `TARGET` | `disjoint_splits.py` | 0.55 / 0.20 / 0.25 | train / cal / test share of the kept statements |
| `SIZE_TOL` | `disjoint_splits.py` | 0.02 | allowed deviation from `TARGET` (absolute) |
| `INCORRECT_WEIGHT` | `disjoint_splits.py` | 2 | METIS edge weight of incorrect pairs (correct: 1) |
| `SEEDS`, `UFACTORS`, `N_ITER` | `make_splits.py` | 0–9, 30/100/300, 4 | METIS runs tried per graph |
| `AUG_CAP` | `make_splits.py` | 0.30 | cap on added true/false statements per split |
| `SEED` | `make_splits.py` | 0 | sampling seed for balancing |
| `PRIORITY` | `dedup_splits.py` | test, cal, train | which split keeps its rows on a near-duplicate |
| `POLARITY_TOL` | `dedup_splits.py` | city_locations: 0.10 | allowed deviation from 50 / 50 "is" / "is not" per class |
| `SOURCE_TOL` | `dedup_splits.py` | 0.5 pp | allowed change of a split's `source_file` shares |

