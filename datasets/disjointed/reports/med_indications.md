# Splits: med_indications

## Build

- 6898 statements (5 duplicates dropped)
- true_false: entities=1793, pairs=4369, statements=5901
- neither: entities=371, pairs=860, statements=997
- true_false: kept 75.4% (seed 8, ufactor 300, sizes ok: True)
- neither: kept 77.8% (seed 5, ufactor 30, sizes ok: True)
- entities put into different splits by different graphs: 0
- original statements kept: 5228/6898 (75.8%)
- negation flips available per split: {0: 1180, 1: 407, 2: 526}
- correct-pair statements available per split: {0: 50, 1: 18, 2: 24}
- dropped neither statements: 129 (of which 129 because the pair had both forms)
- added: {'correct_pair': 16, 'negation_flip': 215, 'neither_shuffle': 1695}

Composition after balancing:

```
shape: (3, 9)
┌────────┬──────┬─────────┬──────────┬───┬───────────┬────────────┬─────────────┬───────┐
│ _split ┆ n    ┆ true_is ┆ true_not ┆ … ┆ false_not ┆ neither_is ┆ neither_not ┆ share │
│ ---    ┆ ---  ┆ ---     ┆ ---      ┆   ┆ ---       ┆ ---        ┆ ---         ┆ ---   │
│ str    ┆ u32  ┆ u32     ┆ u32      ┆   ┆ u32       ┆ u32        ┆ u32         ┆ f64   │
╞════════╪══════╪═════════╪══════════╪═══╪═══════════╪════════════╪═════════════╪═══════╡
│ cal    ┆ 1380 ┆ 230     ┆ 230      ┆ … ┆ 230       ┆ 230        ┆ 230         ┆ 0.196 │
│ test   ┆ 1763 ┆ 293     ┆ 294      ┆ … ┆ 294       ┆ 294        ┆ 294         ┆ 0.251 │
│ train  ┆ 3882 ┆ 647     ┆ 647      ┆ … ┆ 647       ┆ 647        ┆ 647         ┆ 0.553 │
└────────┴──────┴─────────┴──────────┴───┴───────────┴────────────┴─────────────┴───────┘
```

## Near-duplicate dedup

Priority (first keeps its rows): test > cal > train. Seed: 0.

Normalization: object_1: `drug` key + token prefix; object_2: `cond` key.

### Inventory

| split | rows | F/A | F/N | T/A | T/N | N/A | N/N |
|---|---|---|---|---|---|---|---|
| train | 3882 | 647 | 647 | 647 | 647 | 647 | 647 |
| cal | 1380 | 230 | 230 | 230 | 230 | 230 | 230 |
| test | 1763 | 294 | 294 | 293 | 294 | 294 | 294 |

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
| drug key (object_1) | 11 | 5 | 3 |  |  |  |
| cond key (object_2) | 11 | 9 | 8 |  |  |  |
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

| split | before | cond-key | drug-key | drug-prefix | total | after drops | after rebalance |
|---|---|---|---|---|---|---|---|
| train | 3882 | 171 | 42 | 43 | 248 | 3634 | 3378 |
| cal | 1380 | 32 | 11 | 0 | 43 | 1337 | 1308 |
| test | 1763 | 0 | 0 | 0 | 0 | 1763 | 1758 |

### Balance (label x negation)

Target: equal true / false / neither; affirmative share 50 +- 0% inside each class.

| split | stage | F/A | F/N | T/A | T/N | N/A | N/N |
|---|---|---|---|---|---|---|---|
| train | before | 647 | 647 | 647 | 647 | 647 | 647 |
| train | after drops | 602 | 563 | 563 | 612 | 647 | 647 |
| train | after rebalance | 563 | 563 | 563 | 563 | 563 | 563 |
| cal | before | 230 | 230 | 230 | 230 | 230 | 230 |
| cal | after drops | 221 | 218 | 218 | 220 | 230 | 230 |
| cal | after rebalance | 218 | 218 | 218 | 218 | 218 | 218 |
| test | before | 294 | 294 | 293 | 294 | 294 | 294 |
| test | after drops | 294 | 294 | 293 | 294 | 294 | 294 |
| test | after rebalance | 293 | 293 | 293 | 293 | 293 | 293 |

### source_file shares (%)

| split | source_file | before | after | change (pp) |
|---|---|---|---|---|
| train | med_indications | 66.67 | 66.67 | +0.00 |
| train | med_indications_synthetic | 33.33 | 33.33 | +0.00 |
| cal | med_indications | 66.67 | 66.67 | +0.00 |
| cal | med_indications_synthetic | 33.33 | 33.33 | +0.00 |
| test | med_indications | 66.65 | 66.67 | +0.02 |
| test | med_indications_synthetic | 33.35 | 33.33 | -0.02 |

### Collisions (58)

`keep_split:value ~ lose_split:value` (rule).

- `test:Gallium citrate Ga-67` ~ `cal:Gallium chloride Ga-67` (drug-key)
- `test:Mometasone furoate` ~ `cal:Mometasone` (drug-key)
- `test:Morphine` ~ `cal:Morphine glucuronide` (drug-key)
- `test:anemias` ~ `cal:anaemia` (cond-key)
- `test:fungal infection` ~ `cal:fungal infections` (cond-key)
- `test:hairy cell leukemia` ~ `cal:hairy cell leukaemia` (cond-key)
- `test:Pain` ~ `cal:pains` (cond-key)
- `test:pain` ~ `cal:pains` (cond-key)
- `test:rashes` ~ `cal:rash` (cond-key)
- `test:schizoaffective disorder` ~ `cal:schizoaffective disorders` (cond-key)
- `test:stroke` ~ `cal:strokes` (cond-key)
- `test:viral infections` ~ `cal:viral infection` (cond-key)
- `test:Aluminum hydroxide` ~ `train:Aluminum chloride` (drug-prefix)
- `test:Aluminum hydroxide` ~ `train:Aluminum sulfate` (drug-prefix)
- `cal:Choline salicylate` ~ `train:Choline C-11` (drug-prefix)
- `test:Isosorbide mononitrate` ~ `train:Isosorbide` (drug-key)
- `test:Isosorbide mononitrate` ~ `train:Isosorbide dinitrate` (drug-key)
- `test:Lithium citrate` ~ `train:Lithium cation` (drug-prefix)
- `test:Magnesium Aluminum Silicate` ~ `train:Magnesium sulfate` (drug-prefix)
- `cal:Polyethylene glycol 400` ~ `train:Polyethylene glycol` (drug-key)
- `cal:Polyethylene glycol 400` ~ `train:Polyethylene glycol 300` (drug-key)
- `cal:Potassium acetate` ~ `train:Potassium bitartrate` (drug-prefix)
- `cal:Potassium sulfate` ~ `train:Potassium bitartrate` (drug-prefix)
- `cal:Potassium acetate` ~ `train:Potassium chloride` (drug-key)
- `cal:Potassium sulfate` ~ `train:Potassium chloride` (drug-key)
- `cal:Potassium cation` ~ `train:Potassium chloride` (drug-prefix)
- `cal:Potassium acetate` ~ `train:Potassium citrate` (drug-key)
- `cal:Potassium sulfate` ~ `train:Potassium citrate` (drug-key)
- `cal:Potassium cation` ~ `train:Potassium citrate` (drug-prefix)
- `cal:Prednisolone phosphate` ~ `train:Prednisolone` (drug-key)
- `cal:Selenium` ~ `train:Selenium Sulfide` (drug-key)
- `cal:Silver sulfadiazine` ~ `train:Silver` (drug-key)
- `test:Sodium phosphate, dibasic` ~ `train:Sodium acetate` (drug-prefix)
- `cal:Sodium phosphate, monobasic, unspecified form` ~ `train:Sodium acetate` (drug-prefix)
- `test:Sodium stibogluconate` ~ `train:Sodium acetate` (drug-prefix)
- `cal:Sodium phosphate, monobasic, unspecified form` ~ `train:Sodium phosphate, monobasic` (drug-prefix)
- `cal:Strontium chloride` ~ `train:Strontium malonate` (drug-prefix)
- `cal:Urea C-14` ~ `train:Urea C-13` (drug-key)
- `test:anemias` ~ `train:Anemia` (cond-key)
- `cal:acute coronary syndromes` ~ `train:acute coronary syndrome` (cond-key)
- `test:anemias` ~ `train:anemia` (cond-key)
- `cal:anxiety disorders` ~ `train:anxiety disorder` (cond-key)
- `test:bacterial infection` ~ `train:bacterial infections` (cond-key)
- `cal:bipolar disorders` ~ `train:bipolar disorder` (cond-key)
- `test:cardiovascular disease` ~ `train:cardiovascular diseases` (cond-key)
- `cal:oedema` ~ `train:edema` (cond-key)
- `test:fractures` ~ `train:fracture` (cond-key)
- `test:head and neck cancers` ~ `train:head and neck cancer` (cond-key)
- `test:haemorrhage` ~ `train:hemorrhage` (cond-key)
- `cal:idiopathic Parkinson disease` ~ `train:idiopathic Parkinsons disease` (cond-key)
- `cal:infection` ~ `train:infections` (cond-key)
- `cal:insomnias` ~ `train:insomnia` (cond-key)
- `cal:myelodysplastic syndrome` ~ `train:myelodysplastic syndromes` (cond-key)
- `cal:renal diseases` ~ `train:renal disease` (cond-key)
- `cal:seizures` ~ `train:seizure` (cond-key)
- `test:thyroid carcinomas` ~ `train:thyroid carcinoma` (cond-key)
- `test:urinary tract infection` ~ `train:urinary tract infections` (cond-key)
- `test:ventricular tachycardia` ~ `train:ventricular tachycardias` (cond-key)

### Needs human review (18)

Drug matches whose shared tokens are only cations or generic words (aluminium, aluminum, calcium, choline, iron, lithium, magnesium, potassium, silver, sodium, strontium, zinc). They may be false positives; the rows were dropped anyway.

- `test:Aluminum hydroxide` ~ `train:Aluminum chloride` (drug-prefix)
- `test:Aluminum hydroxide` ~ `train:Aluminum sulfate` (drug-prefix)
- `cal:Choline salicylate` ~ `train:Choline C-11` (drug-prefix)
- `test:Lithium citrate` ~ `train:Lithium cation` (drug-prefix)
- `test:Magnesium Aluminum Silicate` ~ `train:Magnesium sulfate` (drug-prefix)
- `cal:Potassium acetate` ~ `train:Potassium bitartrate` (drug-prefix)
- `cal:Potassium sulfate` ~ `train:Potassium bitartrate` (drug-prefix)
- `cal:Potassium acetate` ~ `train:Potassium chloride` (drug-key)
- `cal:Potassium sulfate` ~ `train:Potassium chloride` (drug-key)
- `cal:Potassium cation` ~ `train:Potassium chloride` (drug-prefix)
- `cal:Potassium acetate` ~ `train:Potassium citrate` (drug-key)
- `cal:Potassium sulfate` ~ `train:Potassium citrate` (drug-key)
- `cal:Potassium cation` ~ `train:Potassium citrate` (drug-prefix)
- `cal:Silver sulfadiazine` ~ `train:Silver` (drug-key)
- `test:Sodium phosphate, dibasic` ~ `train:Sodium acetate` (drug-prefix)
- `cal:Sodium phosphate, monobasic, unspecified form` ~ `train:Sodium acetate` (drug-prefix)
- `test:Sodium stibogluconate` ~ `train:Sodium acetate` (drug-prefix)
- `cal:Strontium chloride` ~ `train:Strontium malonate` (drug-prefix)

### Not handled

Related but distinct drugs (e.g. omeprazole / esomeprazole, chloroquine / hydroxychloroquine) are kept apart only if their keys collide. No local RxNorm / ATC / MeSH resource exists in the repo, so concept-level disjointness is not possible here. Candidates left in the output, by a shared-stem heuristic (one key is a substring of the other, at least 6 characters):

- `train:Amphetamine` / `cal:Dextroamphetamine`
- `train:Ascorbic acid` / `cal:Dehydroascorbic acid`
- `train:Cysteine` / `cal:Acetylcysteine`
- `train:Epinephrine` / `cal:Norepinephrine`
- `train:Iodine` / `cal:Povidone-iodine`
- `train:Modafinil` / `cal:Armodafinil`
- `train:Nitrazepam` / `cal:Flunitrazepam`
- `train:Omeprazole` / `cal:Esomeprazole`
- `train:Tannic acid` / `cal:Phentantannic acid`
- `train:Chloroquine` / `test:Hydroxychloroquine`
- `train:Ciprofloxacin` / `test:Technetium Tc-99m ciprofloxacin`
- `train:Ofloxacin` / `test:Technetium Tc-99m ciprofloxacin`
- `train:Salicylic acid` / `test:Acetylsalicylic acid`
- `cal:Polyethylene glycol 400` / `train:Methoxy polyethylene glycol-epoetin beta`
- `cal:Ethanol` / `test:Ethanolamine oleate`
- `cal:Phenytoin` / `test:Pramphenytoin`
- `test:Cholic Acid` / `train:Ursodeoxycholic acid`
- `test:Estradiol valerate` / `train:Ethinylestradiol`
- `test:Ethamide` / `train:Dapineethamide`
- `test:Glucosamine` / `train:N-Acetylglucosamine`
- `test:Leucine` / `train:Isoleucine`
- `test:Methadone` / `train:Normethadone`
- `test:Methylphenidate` / `train:Dexmethylphenidate`
- `test:Morphine` / `train:Apomorphine`
- `test:Morphine` / `train:Diamorphine`

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
| drug key (object_1) | 0 | 0 | 0 |  |  |  |
| cond key (object_2) | 0 | 0 | 0 |  |  |  |
| cross-role object_1 -> object_2 exact | 0 | 0 | 0 | 0 | 0 | 0 |
| cross-role object_1 -> object_2 lower | 0 | 0 | 0 | 0 | 0 | 0 |

Overlap counts must be zero for every exact, entity_key and normalized check (lower-case counts are informational). True / false / neither must be equal, with an affirmative share of 50 +- 0% inside each class; source_file shares must change by at most 0.5 pp.

**PASS**

## Excluded original statements

```
shape: (4, 2)
┌─────────────────────────────────┬──────┐
│ reason                          ┆ len  │
│ ---                             ┆ ---  │
│ str                             ┆ u32  │
╞═════════════════════════════════╪══════╡
│ cut                             ┆ 1670 │
│ dropped (both forms)            ┆ 129  │
│ dropped (near-duplicate)        ┆ 280  │
│ dropped (rebalancing after ded… ┆ 150  │
└─────────────────────────────────┴──────┘
```

## Result

| split | after balancing | written |
|---|---|---|
| train | 3882 | 3378 |
| cal | 1380 | 1308 |
| test | 1763 | 1758 |

**PASS**
