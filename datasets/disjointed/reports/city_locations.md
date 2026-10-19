# Splits: city_locations

## Build

- 7247 statements (5 duplicates dropped)
- true_false: entities=1592, pairs=4139, statements=5498
- neither: entities=354, pairs=1301, statements=1749
- true_false: kept 71.1% (seed 8, ufactor 30, sizes ok: True)
- neither: kept 70.4% (seed 9, ufactor 30, sizes ok: True)
- entities put into different splits by different graphs: 0
- original statements kept: 5138/7247 (70.9%)
- negation flips available per split: {0: 1115, 1: 457, 2: 524}
- correct-pair statements available per split: {}
- dropped neither statements: 371 (of which 371 because the pair had both forms)
- added: {'negation_flip': 504, 'neither_shuffle': 1348}

Composition after balancing:

```
shape: (3, 9)
┌────────┬──────┬─────────┬──────────┬───┬───────────┬────────────┬─────────────┬───────┐
│ _split ┆ n    ┆ true_is ┆ true_not ┆ … ┆ false_not ┆ neither_is ┆ neither_not ┆ share │
│ ---    ┆ ---  ┆ ---     ┆ ---      ┆   ┆ ---       ┆ ---        ┆ ---         ┆ ---   │
│ str    ┆ u32  ┆ u32     ┆ u32      ┆   ┆ u32       ┆ u32        ┆ u32         ┆ f64   │
╞════════╪══════╪═════════╪══════════╪═══╪═══════════╪════════════╪═════════════╪═══════╡
│ cal    ┆ 1211 ┆ 161     ┆ 242      ┆ … ┆ 162       ┆ 202        ┆ 202         ┆ 0.183 │
│ test   ┆ 1642 ┆ 245     ┆ 302      ┆ … ┆ 245       ┆ 274        ┆ 274         ┆ 0.248 │
│ train  ┆ 3766 ┆ 590     ┆ 665      ┆ … ┆ 590       ┆ 628        ┆ 628         ┆ 0.569 │
└────────┴──────┴─────────┴──────────┴───┴───────────┴────────────┴─────────────┴───────┘
```

## Near-duplicate dedup

Priority (first keeps its rows): test > cal > train. Seed: 0.

Normalization: object_1+object_2: `loc` key.

### Inventory

| split | rows | F/A | F/N | T/A | T/N | N/A | N/N |
|---|---|---|---|---|---|---|---|
| train | 3766 | 665 | 590 | 590 | 665 | 628 | 628 |
| cal | 1211 | 242 | 162 | 161 | 242 | 202 | 202 |
| test | 1642 | 302 | 245 | 245 | 302 | 274 | 274 |

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
| loc key (object_1+object_2) | 0 | 0 | 1 |  |  |  |
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

| split | before | loc-key | total | after drops | after rebalance |
|---|---|---|---|---|---|
| train | 3766 | 0 | 0 | 3766 | 3765 |
| cal | 1211 | 3 | 3 | 1208 | 1200 |
| test | 1642 | 0 | 0 | 1642 | 1641 |

### Balance (label x negation)

Target: equal true / false / neither; affirmative share 50 +- 10% inside each class.

| split | stage | F/A | F/N | T/A | T/N | N/A | N/N |
|---|---|---|---|---|---|---|---|
| train | before | 665 | 590 | 590 | 665 | 628 | 628 |
| train | after drops | 665 | 590 | 590 | 665 | 628 | 628 |
| train | after rebalance | 665 | 590 | 590 | 665 | 627 | 628 |
| cal | before | 242 | 162 | 161 | 242 | 202 | 202 |
| cal | after drops | 242 | 161 | 160 | 241 | 202 | 202 |
| cal | after rebalance | 239 | 161 | 160 | 240 | 200 | 200 |
| test | before | 302 | 245 | 245 | 302 | 274 | 274 |
| test | after drops | 302 | 245 | 245 | 302 | 274 | 274 |
| test | after rebalance | 302 | 245 | 245 | 302 | 273 | 274 |

### source_file shares (%)

| split | source_file | before | after | change (pp) |
|---|---|---|---|---|
| train | city_locations | 66.65 | 66.67 | +0.02 |
| train | city_locations_synthetic | 33.35 | 33.33 | -0.02 |
| cal | city_locations | 66.64 | 66.67 | +0.03 |
| cal | city_locations_synthetic | 33.36 | 33.33 | -0.03 |
| test | city_locations | 66.63 | 66.67 | +0.04 |
| test | city_locations_synthetic | 33.37 | 33.33 | -0.04 |

### Collisions (1)

`keep_split:value ~ lose_split:value` (rule).

- `test:Los Angeles` ~ `cal:Los Ángeles` (loc-key)

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
| loc key (object_1+object_2) | 0 | 0 | 0 |  |  |  |
| cross-role object_1 -> object_2 exact | 0 | 0 | 0 | 0 | 0 | 0 |
| cross-role object_1 -> object_2 lower | 0 | 0 | 0 | 0 | 0 | 0 |

Overlap counts must be zero for every exact, entity_key and normalized check (lower-case counts are informational). True / false / neither must be equal, with an affirmative share of 50 +- 10% inside each class; source_file shares must change by at most 0.5 pp.

**PASS**

## Excluded original statements

```
shape: (4, 2)
┌─────────────────────────────────┬──────┐
│ reason                          ┆ len  │
│ ---                             ┆ ---  │
│ str                             ┆ u32  │
╞═════════════════════════════════╪══════╡
│ cut                             ┆ 2109 │
│ dropped (both forms)            ┆ 371  │
│ dropped (near-duplicate)        ┆ 3    │
│ dropped (rebalancing after ded… ┆ 8    │
└─────────────────────────────────┴──────┘
```

## Result

| split | after balancing | written |
|---|---|---|
| train | 3766 | 3765 |
| cal | 1211 | 1200 |
| test | 1642 | 1641 |

**PASS**
