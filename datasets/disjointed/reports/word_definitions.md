# Splits: word_definitions

## Build

- 8496 statements (4 duplicates dropped)
- true_false/instances: entities=397, pairs=795, statements=998
- true_false/types+synonyms: entities=3118, pairs=3630, statements=3998
- neither: entities=2269, pairs=3324, statements=3500
- true_false/instances: kept 81.8% (seed 7, ufactor 100, sizes ok: True)
- true_false/types+synonyms: kept 94.5% (seed 2, ufactor 30, sizes ok: True)
- neither: kept 87.9% (seed 9, ufactor 100, sizes ok: True)
- entities put into different splits by different graphs: 20
- original statements kept: 7637/8496 (89.9%)
- negation flips available per split: {0: 1978, 1: 675, 2: 867}
- correct-pair statements available per split: {0: 338, 1: 148, 2: 134}
- dropped neither statements: 266 (of which 175 because the pair had both forms)
- added: {'correct_pair': 522, 'negation_flip': 584, 'neither_shuffle': 29}

Composition after balancing:

```
shape: (3, 9)
┌────────┬──────┬─────────┬──────────┬───┬───────────┬────────────┬─────────────┬───────┐
│ _split ┆ n    ┆ true_is ┆ true_not ┆ … ┆ false_not ┆ neither_is ┆ neither_not ┆ share │
│ ---    ┆ ---  ┆ ---     ┆ ---      ┆   ┆ ---       ┆ ---        ┆ ---         ┆ ---   │
│ str    ┆ u32  ┆ u32     ┆ u32      ┆   ┆ u32       ┆ u32        ┆ u32         ┆ f64   │
╞════════╪══════╪═════════╪══════════╪═══╪═══════════╪════════════╪═════════════╪═══════╡
│ cal    ┆ 1706 ┆ 284     ┆ 284      ┆ … ┆ 284       ┆ 285        ┆ 285         ┆ 0.201 │
│ test   ┆ 2126 ┆ 354     ┆ 354      ┆ … ┆ 354       ┆ 355        ┆ 355         ┆ 0.25  │
│ train  ┆ 4674 ┆ 779     ┆ 779      ┆ … ┆ 779       ┆ 779        ┆ 779         ┆ 0.549 │
└────────┴──────┴─────────┴──────────┴───┴───────────┴────────────┴─────────────┴───────┘
```

## Near-duplicate dedup

Priority (first keeps its rows): test > cal > train. Seed: 0.

Normalization: object_1+object_2: `cond` key.

### Inventory

| split | rows | F/A | F/N | T/A | T/N | N/A | N/N |
|---|---|---|---|---|---|---|---|
| train | 4674 | 779 | 779 | 779 | 779 | 779 | 779 |
| cal | 1706 | 284 | 284 | 284 | 284 | 285 | 285 |
| test | 2126 | 354 | 354 | 354 | 354 | 355 | 355 |

### Pre-checks

Label consistency (violating rows):

| check | train | cal | test |
|---|---|---|---|
| one-hot labels | 0 | 0 | 0 |
| multiclass == T + 2N | 0 | 0 | 0 |
| neither <=> real_object == 0 | 0 | 0 | 0 |
| is_true <=> correct (real rows) | 0 | 0 | 0 |

Overlap across splits (shared values):

| check | train-cal | train-test | cal-test | cal-train | test-train | test-cal |
|---|---|---|---|---|---|---|
| object_1 exact | 0 | 0 | 0 |  |  |  |
| object_1 lower | 0 | 0 | 0 |  |  |  |
| object_2 exact | 0 | 0 | 0 |  |  |  |
| object_2 lower | 0 | 0 | 0 |  |  |  |
| statement exact | 0 | 0 | 0 |  |  |  |
| statement lower | 0 | 0 | 0 |  |  |  |
| pair exact | 0 | 0 | 0 |  |  |  |
| pair lower | 0 | 0 | 0 |  |  |  |
| entity_key (both roles) | 0 | 0 | 0 |  |  |  |
| cond key (object_1+object_2) | 2 | 3 | 1 |  |  |  |
| cross-role object_1 -> object_2 exact | 0 | 0 | 0 | 0 | 0 | 0 |
| cross-role object_1 -> object_2 lower | 0 | 0 | 0 | 0 | 0 | 0 |

Duplicates and nulls:

| check | split | n |
|---|---|---|
| duplicate statements | train | 0 |
| duplicate statements | cal | 0 |
| duplicate statements | test | 0 |

### Dropped rows

A row hit by two rules counts once per rule, and once in `total`.

| split | before | cond-key | total | after drops | after rebalance |
|---|---|---|---|---|---|
| train | 4674 | 21 | 21 | 4653 | 4638 |
| cal | 1706 | 2 | 2 | 1704 | 1698 |
| test | 2126 | 0 | 0 | 2126 | 2124 |

### Balance (label x negation)

Target: equal true / false / neither; affirmative share 50 +- 0% inside each class.

| split | stage | F/A | F/N | T/A | T/N | N/A | N/N |
|---|---|---|---|---|---|---|---|
| train | before | 779 | 779 | 779 | 779 | 779 | 779 |
| train | after drops | 776 | 777 | 776 | 776 | 775 | 773 |
| train | after rebalance | 773 | 773 | 773 | 773 | 773 | 773 |
| cal | before | 284 | 284 | 284 | 284 | 285 | 285 |
| cal | after drops | 284 | 284 | 283 | 283 | 285 | 285 |
| cal | after rebalance | 283 | 283 | 283 | 283 | 283 | 283 |
| test | before | 354 | 354 | 354 | 354 | 355 | 355 |
| test | after drops | 354 | 354 | 354 | 354 | 355 | 355 |
| test | after rebalance | 354 | 354 | 354 | 354 | 354 | 354 |

### source_file shares (%)

| split | source_file | before | after | change (pp) |
|---|---|---|---|---|
| train | word_instances | 10.65 | 10.54 | -0.11 |
| train | word_instances_synthetic | 4.75 | 4.72 | -0.03 |
| train | word_synonyms | 28.22 | 28.27 | +0.05 |
| train | word_synonyms_synthetic | 14.29 | 14.29 | +0.00 |
| train | word_types | 27.79 | 27.86 | +0.06 |
| train | word_types_synthetic | 14.29 | 14.32 | +0.02 |
| cal | word_instances | 11.49 | 11.54 | +0.05 |
| cal | word_instances_synthetic | 4.81 | 4.83 | +0.02 |
| cal | word_synonyms | 25.73 | 25.74 | +0.00 |
| cal | word_synonyms_synthetic | 14.30 | 14.25 | -0.05 |
| cal | word_types | 29.37 | 29.39 | +0.02 |
| cal | word_types_synthetic | 14.30 | 14.25 | -0.05 |
| test | word_instances | 10.91 | 10.92 | +0.01 |
| test | word_instances_synthetic | 4.80 | 4.80 | +0.00 |
| test | word_synonyms | 27.23 | 27.26 | +0.03 |
| test | word_synonyms_synthetic | 14.30 | 14.31 | +0.01 |
| test | word_types | 28.46 | 28.48 | +0.03 |
| test | word_types_synthetic | 14.30 | 14.22 | -0.08 |

### Collisions (6)

`keep_split:value ~ lose_split:value` (rule).

- `test:circumstances` ~ `cal:circumstance` (cond-key)
- `test:algier` ~ `train:algiers` (cond-key)
- `test:arging` ~ `train:argings` (cond-key)
- `cal:hemoprotein` ~ `train:haemoprotein` (cond-key)
- `cal:hemorrhoidectomy` ~ `train:haemorrhoidectomy` (cond-key)
- `test:spirit` ~ `train:spirits` (cond-key)

### Needs human review (0)

Drug matches whose shared tokens are only cations or generic words (aluminium, aluminum, calcium, choline, iron, lithium, magnesium, potassium, silver, sodium, strontium, zinc). They may be false positives; the rows were dropped anyway.

None.

### Not handled

Related but distinct drugs (e.g. omeprazole / esomeprazole, chloroquine / hydroxychloroquine) are kept apart only if their keys collide. No local RxNorm / ATC / MeSH resource exists in the repo, so concept-level disjointness is not possible here. Candidates left in the output, by a shared-stem heuristic (one key is a substring of the other, at least 6 characters):

None.

### Post-checks

| check | train-cal | train-test | cal-test | cal-train | test-train | test-cal |
|---|---|---|---|---|---|---|
| object_1 exact | 0 | 0 | 0 |  |  |  |
| object_1 lower | 0 | 0 | 0 |  |  |  |
| object_2 exact | 0 | 0 | 0 |  |  |  |
| object_2 lower | 0 | 0 | 0 |  |  |  |
| statement exact | 0 | 0 | 0 |  |  |  |
| statement lower | 0 | 0 | 0 |  |  |  |
| pair exact | 0 | 0 | 0 |  |  |  |
| pair lower | 0 | 0 | 0 |  |  |  |
| entity_key (both roles) | 0 | 0 | 0 |  |  |  |
| cond key (object_1+object_2) | 0 | 0 | 0 |  |  |  |
| cross-role object_1 -> object_2 exact | 0 | 0 | 0 | 0 | 0 | 0 |
| cross-role object_1 -> object_2 lower | 0 | 0 | 0 | 0 | 0 | 0 |

Overlap counts must be zero for every exact, entity_key and normalized check (lower-case counts are informational). True / false / neither must be equal, with an affirmative share of 50 +- 0% inside each class; source_file shares must change by at most 0.5 pp.

**PASS**

## Excluded original statements

```
shape: (5, 2)
┌─────────────────────────────────┬─────┐
│ reason                          ┆ len │
│ ---                             ┆ --- │
│ str                             ┆ u32 │
╞═════════════════════════════════╪═════╡
│ cut                             ┆ 859 │
│ dropped (balancing)             ┆ 91  │
│ dropped (both forms)            ┆ 175 │
│ dropped (near-duplicate)        ┆ 21  │
│ dropped (rebalancing after ded… ┆ 18  │
└─────────────────────────────────┴─────┘
```

## Result

| split | after balancing | written |
|---|---|---|
| train | 4674 | 4638 |
| cal | 1706 | 1698 |
| test | 2126 | 2124 |

**PASS**
