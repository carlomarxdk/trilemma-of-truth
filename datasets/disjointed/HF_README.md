---
pretty_name: "Trilemma of Truth: True, False, and Neither Statements"
license: cc-by-4.0
language:
  - en
multilinguality:
  - monolingual
size_categories:
  - 10K<n<100K
source_datasets:
  - original
task_categories:
  - text-classification
  - zero-shot-classification
task_ids:
  - fact-checking
  - multi-class-classification
annotations_creators:
  - found
  - machine-generated
language_creators:
  - machine-generated
tags:
  - truthfulness
  - veracity
  - probing
  - interpretability
  - negation
  - synthetic-entities
  - geography
  - biomedical
  - lexical-semantics
configs:
  - config_name: city_locations_disjoint
    default: true
    keep_default_na: false  # keep the word "nan" as text
    data_files:
      - split: train
        path: data/city_locations_disjoint/train.csv
      - split: validation
        path: data/city_locations_disjoint/validation.csv
      - split: test
        path: data/city_locations_disjoint/test.csv
  - config_name: med_indications_disjoint
    keep_default_na: false  # keep the word "nan" as text
    data_files:
      - split: train
        path: data/med_indications_disjoint/train.csv
      - split: validation
        path: data/med_indications_disjoint/validation.csv
      - split: test
        path: data/med_indications_disjoint/test.csv
  - config_name: word_definitions_disjoint
    keep_default_na: false  # keep the word "nan" as text
    data_files:
      - split: train
        path: data/word_definitions_disjoint/train.csv
      - split: validation
        path: data/word_definitions_disjoint/validation.csv
      - split: test
        path: data/word_definitions_disjoint/test.csv
  - config_name: city_locations
    sep: "\t"
    keep_default_na: false  # keep the word "nan" as text
    data_files:
      - split: train
        path: data/city_locations/train.tsv
      - split: validation
        path: data/city_locations/validation.tsv
      - split: test
        path: data/city_locations/test.tsv
  - config_name: med_indications
    sep: "\t"
    keep_default_na: false  # keep the word "nan" as text
    data_files:
      - split: train
        path: data/med_indications/train.tsv
      - split: validation
        path: data/med_indications/validation.tsv
      - split: test
        path: data/med_indications/test.tsv
  - config_name: word_definitions
    sep: "\t"
    keep_default_na: false  # keep the word "nan" as text
    data_files:
      - split: train
        path: data/word_definitions/train.tsv
      - split: validation
        path: data/word_definitions/validation.tsv
      - split: test
        path: data/word_definitions/test.tsv
dataset_info:
  - config_name: city_locations_disjoint
    features:
      - name: statement
        dtype: string
      - name: object_1
        dtype: string
      - name: object_2
        dtype: string
      - name: correct_object_2
        dtype: string
      - name: correct
        dtype: bool
      - name: negation
        dtype: bool
      - name: real_object
        dtype: bool
      - name: is_true
        dtype: bool
      - name: is_false
        dtype: bool
      - name: is_neither
        dtype: bool
      - name: multiclass_label
        dtype:
          class_label:
            names:
              '0': 'false'
              '1': 'true'
              '2': 'neither'
      - name: category
        dtype: string
      - name: source_file
        dtype: string
  - config_name: med_indications_disjoint
    features:
      - name: statement
        dtype: string
      - name: object_1
        dtype: string
      - name: object_2
        dtype: string
      - name: correct_object_2
        dtype: string
      - name: correct
        dtype: bool
      - name: negation
        dtype: bool
      - name: real_object
        dtype: bool
      - name: is_true
        dtype: bool
      - name: is_false
        dtype: bool
      - name: is_neither
        dtype: bool
      - name: multiclass_label
        dtype:
          class_label:
            names:
              '0': 'false'
              '1': 'true'
              '2': 'neither'
      - name: category
        dtype: string
      - name: source_file
        dtype: string
  - config_name: word_definitions_disjoint
    features:
      - name: statement
        dtype: string
      - name: object_1
        dtype: string
      - name: object_2
        dtype: string
      - name: correct_object_2
        dtype: string
      - name: correct
        dtype: bool
      - name: negation
        dtype: bool
      - name: real_object
        dtype: bool
      - name: is_true
        dtype: bool
      - name: is_false
        dtype: bool
      - name: is_neither
        dtype: bool
      - name: multiclass_label
        dtype:
          class_label:
            names:
              '0': 'false'
              '1': 'true'
              '2': 'neither'
      - name: category
        dtype: string
      - name: source_file
        dtype: string
  - config_name: city_locations
    features:
      - name: statement
        dtype: string
      - name: object_1
        dtype: string
      - name: object_2
        dtype: string
      - name: correct_object_2
        dtype: string
      - name: correct
        dtype: bool
      - name: negation
        dtype: bool
      - name: real_object
        dtype: bool
      - name: is_true
        dtype: bool
      - name: is_false
        dtype: bool
      - name: is_neither
        dtype: bool
      - name: multiclass_label
        dtype:
          class_label:
            names:
              '0': 'false'
              '1': 'true'
              '2': 'neither'
  - config_name: med_indications
    features:
      - name: statement
        dtype: string
      - name: object_1
        dtype: string
      - name: object_2
        dtype: string
      - name: correct_object_2
        dtype: string
      - name: correct
        dtype: bool
      - name: negation
        dtype: bool
      - name: real_object
        dtype: bool
      - name: is_true
        dtype: bool
      - name: is_false
        dtype: bool
      - name: is_neither
        dtype: bool
      - name: multiclass_label
        dtype:
          class_label:
            names:
              '0': 'false'
              '1': 'true'
              '2': 'neither'
  - config_name: word_definitions
    features:
      - name: statement
        dtype: string
      - name: object_1
        dtype: string
      - name: object_2
        dtype: string
      - name: correct_object_2
        dtype: string
      - name: correct
        dtype: bool
      - name: negation
        dtype: bool
      - name: real_object
        dtype: bool
      - name: is_true
        dtype: bool
      - name: is_false
        dtype: bool
      - name: is_neither
        dtype: bool
      - name: multiclass_label
        dtype:
          class_label:
            names:
              '0': 'false'
              '1': 'true'
              '2': 'neither'
---

# Dataset Card for Trilemma of Truth

[![arXiv](https://img.shields.io/badge/arXiv-2506.23921-b31b1b.svg)](https://arxiv.org/abs/2506.23921)
[![GitHub](https://img.shields.io/badge/GitHub-trilemma--of--truth-181717?logo=github)](https://github.com/carlomarxdk/trilemma-of-truth)
[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F5900-blue)](https://doi.org/10.57967/hf/5900)

## Dataset Description

- **Repository:** [GitHub Repository](https://github.com/carlomarxdk/trilemma-of-truth)
- **Paper:** [The Trilemma of Truth in Large Language Models](https://arxiv.org/abs/2506.23921)
- **Point of Contact:** [Germans Savcisens](mailto:germans@savcisens.com)
- **Language:** English
- **License:** CC BY 4.0 

### Dataset Overview

Most veracity datasets only have two labels: *true* and *false*. The **Trilemma of Truth** (ToT) dataset is a benchmark for evaluating veracity probes across three distinct statement types:

- Factually **true** statements.
- Factually **false** statements.
- **Neither** statements: : statements whose truth value an LLM cannot know.

The dataset covers three domains. Each domain comes in two versions, each with three standard splits: `train`, `validation`, and `test`.

| Domain | Relation | Source | Recommended config | Paper config |
|---|---|---|---|---|
| City locations | city → country | [GeoNames 2.0.0](https://www.geonames.org/) | `city_locations_disjoint` | `city_locations` |
| Medical indications | drug → indicated disease/condition | [DrugBank 5.1.12](https://go.drugbank.com/releases/5-1-12) | `med_indications_disjoint` | `med_indications` |
| Word definitions | noun → synonym / type / instance | [WordsAPI](https://www.wordsapi.com/) (public sample) | `word_definitions_disjoint` | `word_definitions` |

> [!IMPORTANT]
> **(Oct 19, 2026) Use the `*_disjoint` configs.** We introduce an updated configs for each data domain: no entity (city, country, drug, disease, or word) appears in more than one split (with additonal cleaning) . The original configs only keep (`object_1`, `object_2`) *pairs* apart.  Use the original configs to reproduce the results in the [paper](https://arxiv.org/abs/2506.23921). See [Entity-Disjoint Splits](#entity-disjoint-splits) and [Disjoint Documentation on GitHub](https://github.com/carlomarxdk/trilemma-of-truth/blob/main/datasets/disjointed/DOCS.md)).

Every configuration has both *affirmative* and *negated* ("not") statements, incorporating both **real** and **synthetic** entities.

## Uses

### Direct Use
- **Veracity probing:** train and benchmark probes on LLM hidden states for three-way classification (true / false / neither) or for binary true-vs-false classification.
- **Zero-shot evaluation:** prompt LLMs to judge statements and measure whether they abstain on *neither* or *false* statements.

### Out-of-Scope Use
- **Not a medical or factual reference.** Some statements, especially in `med_indications`, are false on purpose. Do not use them as a source of medical, geographic, or lexical facts.
- **Not a model of real-world uncertainty.** *Neither* statements stand in for "unknown to the LLMs". They do not cover ambiguous, contested, time-dependent, or opinion-based claims.

## Dataset Structure

### Data Instances

```python
{
  "statement": "The city of Bangkok is located in Iran.",
  "object_1": "Bangkok",
  "object_2": "Iran",
  'correct_object_2': 'Thailand',    # the object that would make the statement true
  'correct': False,                  # whether the statement is factually correct
  'negation': False,                 # whether the statement uses negation
  'real_object': True,               # True if BOTH object_1 and object_2 are real-world entities
  'is_true': False,                  # correct=True AND real_object=True
  'is_false': True,                  # correct=False AND real_object=True
  'is_neither': False,               # correct=False AND real_object=False (both objects are synthetic)
  'multiclass_label': 0              # 0=false, 1=true, 2=neither
  'category': 'cities',              # (`*_disjoint` only) relation type
  'source_file' : 'city_locations'   # (`*_disjoint` only) source table of the statement
}
```

Examples from each domain:

| Config | Statement | Label |
|---|---|---|
| `city_locations` | The city of Dharān is not located in Sierra Leone. | true (negated) |
| `city_locations` | The city of Dharān is located in Ecuador. | false |
| `city_locations` | The city of Staakess is located in Marbate. | neither |
| `med_indications` | PR-104 is indicated for the treatment of tumors. | true |
| `med_indications` | Guaifenesin is not indicated for the treatment of coughs. | false (negated) |
| `med_indications` | Alostat is indicated for the treatment of hyperampsies. | neither |
| `word_definitions` | Corsage is a synonym of a nosegay. | true |
| `word_definitions` | Towner is not a type of a resident. | false (negated) |
| `word_definitions` | Kharter is a type of a flagrading. | neither |

### Data Fields

| Field | Type | Description |
|---|---|---|
| `statement` | string | The natural-language statement. |
| `object_1` | string | Subject entity (city, drug, or word). |
| `object_2` | string | Object entity (country, disease/condition, or related word). |
| `correct_object_2` | string | Comma-separated list of the objects that make the **affirmative** relation true for `object_1` (e.g., all indications of a drug). It ignores negation: for "The city of Paris is not located in France." the value is `France`. For *neither* rows it holds synthetic filler (sometimes including `object_2` itself) and has no ground-truth meaning. |
| `correct` | bool | Whether the statement, including any negation, is factually true. Always `False` for *neither* rows, where it is a placeholder. |
| `negation` | bool | Whether the statement contains "not". |
| `real_object` | bool | `True` if both entities are real, `False` if both are synthetic. The dataset has no mixed real/synthetic statements. |
| `is_true` | bool | `correct and real_object` |
| `is_false` | bool | `not correct and real_object` |
| `is_neither` | bool | `not real_object` |
| `multiclass_label` | ClassLabel | `0` = false, `1` = true, `2` = neither. Exactly one of `is_true` / `is_false` / `is_neither` is set. |
| `category` | string | *(`*_disjoint` only)* Relation type: `cities`, `indications`, or, for `word_definitions`, `synonyms` / `types` / `instances`. |
| `source_file` | string | *(`*_disjoint` only)* Source table of the statement, e.g. `word_synonyms` or `city_locations_synthetic` (`*_synthetic` = neither statements). |

> **Note on `is_neither`:** This flag requires *both* objects to be synthetic (`real_object=False`). At this point, the dataset **does not** contain statements with mixed entities.


### Data Splits

#### Entity-disjoint configs (recommended)

No entity appears in more than one split, not even as a spelling or plural variant. See [Entity-Disjoint Splits](#entity-disjoint-splits) for how they are built.

| Config | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|---:|
| `city_locations_disjoint` | 3,765 | 1,200 | 1,641 | 6,606 |
| `med_indications_disjoint` | 3,378 | 1,308 | 1,758 | 6,444 |
| `word_definitions_disjoint` | 4,638 | 1,698 | 2,124 | 8,460 |
| **Total** | 11,781 | 4,206 | 5,523 | 21,510 |

Every split has exactly as many true as false as neither statements. Inside each class, affirmative and negated statements are exactly 50 / 50, except in `city_locations_disjoint`, where true and false are about 45 / 55 (at most 40 / 60 in any split). There are not enough city statements to fill the missing polarity.

**Class composition** (A = affirmative, N = negated), all splits combined:

| Config | True (A / N) | False (A / N) | Neither (A / N) |
|---|---|---|---|
| `city_locations_disjoint` | 995 / 1,207 | 1,206 / 996 | 1,100 / 1,102 |
| `med_indications_disjoint` | 1,074 / 1,074 | 1,074 / 1,074 | 1,074 / 1,074 |
| `word_definitions_disjoint` | 1,410 / 1,410 | 1,410 / 1,410 | 1,410 / 1,410 |

**Entities per split** (subjects and objects):

| Config | Train | Validation | Test |
|---|---:|---:|---:|
| `city_locations_disjoint` | 930 | 494 | 521 |
| `med_indications_disjoint` | 1,050 | 481 | 586 |
| `word_definitions_disjoint` | 3,076 | 1,300 | 1,503 |

#### Original configs (used in the paper)

| Config | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `city_locations` | 3,999 | 1,398 | 1,855 | 7,252 |
| `med_indications` | 3,849 | 1,327 | 1,727 | 6,903 |
| `word_definitions` | 4,717 | 1,628 | 2,155 | 8,500 |
| **Total** | 12,565 | 4,353 | 5,737 | 22,655 |

No (`object_1`, `object_2`) pair appears in more than one split, but single entities can.

**Class composition** (A = affirmative, N = negated), all splits combined:

| Config | True (A / N) | False (A / N) | Neither (A / N) |
|---|---|---|---|
| `city_locations` | 1,392 / 1,376 | 1,358 / 1,374 | 876 / 876 |
| `med_indications` | 1,522 / 1,439 | 1,419 / 1,523 | 478 / 522 |
| `word_definitions` | 1,269 / 1,251 | 1,223 / 1,257 | 1,747 / 1,753 |


## Dataset Creation

### Curation Rationale

Existing truthfulness benchmarks may already be in LLM pre-training or fine-tuning data. This dataset was built from scratch to lower that contamination risk and to give full control over provenance. We use simple two-entity relational statements, so every statement has a clear, verifiable truth value that does not depend on context.

### Source Data

#### Data Collection and Processing

**City Locations.** City–country pairs come from GeoNames, accessed through the [`geonamescache`](https://pypi.org/project/geonamescache/) package.
- Cities must have a population of at least 30,000 and an associated country. Cities located in Antarctica are excluded.
- If a city name belongs to several countries, each pair is kept.
- The final set has 1,400 unique cities: the 700 largest plus 700 sampled at random from the rest.
- Template: `The city of [city] is (not) located in [country].` The "The city of" prefix is dropped when the name already contains "City".
- False statements pair a city with an incorrect country.

**Medical Indications.** Drug names and the first sentence of each drug's *Indication* field come from DrugBank 5.1.12 (academic license).
- Diseases and conditions are extracted with two NER models: SciSpaCy `en_ner_bc5cdr_md` and a BioBERT disease NER. A term is kept only if both models tag it.
- Drug names are kept only if SciSpaCy tags them as `CHEMICAL`.
- Pairs are removed if the drug or the indication has a [`wordfreq`](https://pypi.org/project/wordfreq/) Zipf frequency of 0.
- Template: `[drug] is (not) indicated for the treatment of [disease/condition].`
- False statements pair a drug with a disease that shares no words with any of its true indications.

**Word Definitions.** Nouns come from the public WordsAPI sample, which covers about 10% of the database. A noun is kept if it has at least one definition and at least one `synonym`, `typeOf`, or `instanceOf` entry. Templates (with "a"/"an" inserted where needed, and plurals made singular with [`inflect`](https://pypi.org/project/inflect/)):
- `[word] is (not) [instanceOf].`
- `[word] is (not) a type of [typeOf].`
- `[word] is (not) a synonym of [synonym].`

**Synthetic entities (*neither* statements).** Synthetic entities/names come from a n-gram Markov chain trained on the real entity names of each domain, using the [`namemaker`](https://github.com/Rickmsd/namemaker) package.
Synthetic entities are matched with each other at random. The validated pools in the paper are slightly larger (e.g., 138 countries, 243 drugs, 131 diseases) because a few validated names never ended up in a sampled statement.

#### Entity-Disjoint Splits

The `*_disjoint` configs are built from the same source statements as the original configs, in four steps. Steps 1 and 3 are what make them entity-disjoint.

1. **Graph split.** Every entity (`object_1` and `object_2`, in one shared set) is a node of a graph, and every (subject, object) pair is an edge. Entity names are compared ignoring case, whitespace, hyphens, apostrophes, and periods (`dock worker` = `dock-worker` = `dockworker`). In `city_locations`, US states are folded into their country (`the United States, AZ` → `the United States`). True/false and neither statements form separate graphs, since they share no entities; in `word_definitions`, `instances` are cut separately from `types` + `synonyms`, so every relation type appears in every split. [METIS](https://github.com/inducer/pymetis) cuts each graph into train / validation / test with target sizes 55 / 20 / 25% (±2 points), weighting incorrect pairs ×2 so the cut does not keep mostly correct pairs. A statement is kept only if both of its entities fall in the same split; the rest is *cut*.
2. **Balance.** Each split is brought to 1/3 true, 1/3 false and 1/3 neither, and toward 50 / 50 affirmative / negated statements within each class. New statements use the same templates as the originals:
   - *negation flips*: the missing "is" / "is not" twin of a true or false statement, with the label flipped;
   - *correct pairs*: a subject paired with one of its correct objects (`correct_object_2`), in both polarities;
   - *neither shuffles*: a synthetic subject paired with another synthetic object of the same split and relation type.
   Generated statements never use an entity from another split, and candidates whose label contradicts `correct_object_2` are rejected (e.g. a city name that exists in two countries). True/false additions are capped at 30% of a split's true/false statements. A *neither* pair appears in only one polarity per split, and surplus *neither* statements are dropped.
3. **Near-duplicate removal.** A stricter entity key catches variants that step 1 treats as different entities:
   - conditions and words: singular, American spelling, no possessives (`anaemias` = `anemia`, `Parkinson's disease` = `Parkinson disease`);
   - drugs: salt, ester and isotope suffixes removed (`Prednisolone phosphate` = `Prednisolone`, `Urea C-14` = `Urea C-13`), and a match on a whole-word prefix (`Insulin` ~ `Insulin glargine`);
   - places: accents and leading "the" removed (`Los Ángeles` = `Los Angeles`).

   When an entity collides across splits, the statements with it are dropped from the lower-priority split (test, then validation, then train), so train gives up rows and validation and test keep theirs. Statements are never moved between splits. Related but distinct drugs (omeprazole / esomeprazole) are not treated as duplicates.
4. **Rebalance.** Each split is downsampled again to the exact class balance above, keeping its mix of source tables.

The build script and a per-config report are in the [GitHub repository](https://github.com/carlomarxdk/trilemma-of-truth) under `datasets/disjointed/`.

#### Personal and Sensitive Information

The dataset contains no personal data. Some WordsAPI entries name public historical figures (e.g., "Schweitzer"). These are dictionary entries, not personal records.

## Bias, Risks, and Limitations

- **English only.**
- **Simple negations.** Negation uses only the simple "not" construction.
- **Three domains** and only two-entity relations. The results may not carry over to statements that are compositional, involve several relations, or depend on context.
- **Proxy for "unknown".** *Neither* statements do not cover ambiguity, temporal holdout, or contested claims.
- **Residual contamination.** The synthetic names were checked against databases and web search. Because LLM training corpora are not public, a few of them may still resemble strings a model has seen (however, this is unlikely).
- **Source errors.** Labels inherit any errors in GeoNames, DrugBank, and WordsAPI. The NER-based indication extraction may also miss or incorrectly identify conditions.
- **Generated statements (`*_disjoint`).** About 13–28% of the statements in the disjoint configs are generated from the same templates (negation flips, correct pairs, neither shuffles) to balance the splits. They follow the source databases but were not part of the original sampling, and they are not marked in the data.
- **Smaller disjoint configs.** Keeping entities apart costs statements: about 9% fewer in `city_locations_disjoint`, 7% in `med_indications_disjoint` and 0.5% in `word_definitions_disjoint` than in the original configs. Hub entities (e.g. China, India) concentrate in one split.
- **Knowledge cutoff.** The *true* and *false* labels reflect the source databases as they were when the dataset was built. The statements were generated in May 2025. Later changes are not reflected: a drug that gains or loses an indication, a city that is reassigned to a different country, or a dictionary entry that is revised. The labels for such statements may then be outdated. The source snapshots are:
  - `city_locations`: the GeoNames snapshot bundled with the [`geonamescache`](https://pypi.org/project/geonamescache/) package (`v2.0.0`).
  - `med_indications`: DrugBank release 5.1.12, obtained on October 4, 2024.
  - `word_definitions`: the public WordsAPI sample. Its snapshot from October 2024.

### Recommendations
- Use the `*_disjoint` configs for new work, so that test performance is not inflated by entities seen during training.
- Keep affirmative and negated statements. Handling negation is part of what the benchmark tests, and dropping one polarity weakens the evaluation.


## Licensing

The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). GeoNames data is used under CC BY 4.0. `med_indications` is derived from DrugBank 5.1.12, which was accessed under DrugBank's academic license. The raw DrugBank data is not redistributed here, and users who need it should apply for their own license.

## How to Use

```python
from datasets import load_dataset

# Recommended: entity-disjoint splits
ds = load_dataset("carlomarxx/trilemma-of-truth", "city_locations_disjoint")
test = load_dataset("carlomarxx/trilemma-of-truth", "med_indications_disjoint", split="test")

# Splits used in the paper
paper = load_dataset("carlomarxx/trilemma-of-truth", "city_locations")

labels = ds["train"].features["multiclass_label"].names   # ['false', 'true', 'neither']
neither = ds["train"].filter(lambda x: x["is_neither"])
negated = ds["train"].filter(lambda x: x["negation"])
```

## Citation

If you use this dataset, please cite the paper:

```bibtex
@misc{savcisens2025trilemma,
  title         = {The Trilemma of Truth in Large Language Models},
  author        = {Savcisens, Germans and Eliassi-Rad, Tina},
  year          = {2025},
  eprint        = {2506.23921},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2506.23921}
}
```

To cite the dataset itself:

```bibtex
@misc{savcisens2025trilemmadata,
  title     = {trilemma-of-truth},
  author    = {Savcisens, Germans and Eliassi-Rad, Tina},
  year      = {2025},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/5900},
  url       = {https://huggingface.co/datasets/carlomarxx/trilemma-of-truth}
}
```


## Related Work Using This Dataset

- S. Dies, C. Maynard, G. Savcisens, T. Eliassi-Rad. *Epistemic Familiarity is Associated With Belief Stability in Large Language Models.* [arXiv:2511.19166](https://arxiv.org/abs/2511.19166)
- G. Savcisens, S. Dies, C. Maynard, T. Eliassi-Rad. *Belief Coevolution in a Social Network of Generalist and Specialist Large Language Models.* [arXiv:2607.27512](https://arxiv.org/abs/2607.27512) (2026)

---

