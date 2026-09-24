# Tri-Source Business Entity Resolution

Solution pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**.
Business records come from three independent sources that share no identifier. Source 1 is the
deduplicated reference. For every Source-1 entity we find all Source-2 and Source-3 records that
describe the same real-world business. The score is **macro F0.5**: each entity is scored on its own,
precision counts twice as much as recall, and singletons count too.

```
 S1 / S2 / S3 records
        │
        ▼
 1. Normalise    names: legal forms, abbreviations, DBA split, transliteration skeletons
                 addresses: abbreviations, landmarks, postcodes, house numbers
        │
        ▼
 2. Block        union of TF-IDF kNN blockers (name char, name word+phonetic, name+address,
                 address word) + exact keys  ──►  output/candidate_pairs.tsv
        │
        ▼
 3. Featurise    ~100 pairwise features: fuzzy and IDF-weighted similarities, postcode and
                 number agreement or conflict, group-relative ranks
        │
        ▼
 4. Match        LightGBM, grouped 5-fold CV (optional stage 2 on the stage-1 score ranks)
        │
        ▼
 5. Decide       exclusivity (one S1 per S2/S3 record) + threshold or expected-F0.5 rule,
                 tuned on out-of-fold scores  ──►  output/matching_results.tsv
```

Planning and background docs:
[docs/PLAN.md](docs/PLAN.md) (3-day plan and team roles) ·
[docs/RESEARCH.md](docs/RESEARCH.md) (approach research) ·
[docs/METHODOLOGY.md](docs/METHODOLOGY.md) (draft of the methodology write-up we submit) ·
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) (experiment and submission log)

---

## Get the data (team members)

The dataset (2.4 GB, files of 127–510 MB) is too big for git, because GitHub rejects files over 100 MB.
It is attached to the private release
[`dataset-v1`](https://github.com/kksahu444/tri-source-entity-resolution/releases/tag/dataset-v1).
Only collaborators can see it.

```bash
gh auth login                 # once; install gh from https://cli.github.com
python main.py fetch-data     # download (1.1 GB) -> check SHA-256 -> unpack into dataset/train, dataset/test
```

You can also download from the release page in a browser. Either take the `*.tsv.gz` files and
gunzip them into `dataset/train/` and `dataset/test/`, or take the original
`6ab10eb3b23ba_student_resource.zip` and move its `student_resource/dataset/` folder to `dataset/`.
The organisers' validator and documentation template are already committed in
[`student_resource/`](student_resource/).

| split | Source 1 | Source 2 | Source 3 | countries (S1) |
|---|---|---|---|---|
| train | 2,206,821 | 5,034,616 | 5,285,603 | US 60%, India 40% |
| test | 1,732,544 | 4,887,273 | 5,082,316 | India 47%, US 38%, **France 15%** (not in train) |

## Quickstart

```bash
# Python >= 3.11 (developed on 3.14)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py fetch-data   # -> dataset/ (git-ignored)

python main.py eda          # dataset statistics  -> artifacts/eda_train.json
python main.py block        # blocking recall ceiling on train
python main.py train        # CV + decision tuning -> artifacts/model.joblib, train_report.json
python main.py predict      # -> output/matching_results.tsv, output/candidate_pairs.tsv (validated)
python main.py validate --official student_resource/utils/validate_submission.py
python main.py package --team <TEAM_NAME>   # -> <TEAM_NAME>_submission.zip
```

`python main.py all` runs train and predict in one go. Without the real data you can check the whole
pipeline on synthetic data that uses the same format:

```bash
python main.py synth                                   # -> data_synthetic/
python main.py --data-dir data_synthetic all
python main.py score --truth data_synthetic/hidden_labels/test_ground_truth.tsv
python -m pytest -q
```

## Commands

| command | what it does |
|---|---|
| `eda` | singleton rate, matches per entity, whether an S2/S3 record can match more than one S1 (exclusivity check), cross-country pairs, field coverage, example groups |
| `block` | pair recall (overall, S2, S3), reduction ratio, candidates per entity, and the best macro F0.5 a perfect matcher could reach with these candidates |
| `train` | builds features, runs grouped K-fold LightGBM, runs the optional stage-2 model, grid-searches the decision rule on out-of-fold scores, and saves the model plus reports |
| `predict` | scores the test split, writes both TSVs, runs the local rule validator |
| `loco` | leave-one-country-out check that estimates how well the model handles an unseen country (France) |
| `validate` | local re-implementation of the submission rules, and optionally runs the official script too |
| `score` | macro F0.5 of a predictions file against a ground-truth file |
| `package` | builds the zip in the required `output/ + code/business_entity_resolution/ + Documentation_template.md` layout |

Global flags: `--data-dir`, `--artifacts-dir`, `--output-dir`, `--config configs/<file>.json`.
`configs/fast_debug.json` gives quick iterations.

## Repository layout

```
src/entity_resolution/
  config.py      all hyper-parameters (saved with the model)
  io.py          robust TSV reading, submission writing
  normalize.py   name/address normalisation (open-set countries)
  blocking.py    TF-IDF kNN + key blockers, candidate union
  features.py    pairwise + group-context features
  model.py       grouped-CV LightGBM, stage-2 stacking
  decide.py      exclusivity, threshold / expected-F0.5 selection, OOF tuning
  metrics.py     macro F0.5 (exact challenge definition), blocking diagnostics
  pipeline.py    train / predict / LOCO orchestration
  validate.py    submission rule checks
  eda.py         dataset statistics
  package.py     final zip builder
  synthetic.py   format-identical synthetic data for tests
tests/           metric, normalisation, decision and end-to-end tests
configs/         JSON config overrides
docs/            plan, research notes, methodology draft, experiment log
```

## Challenge compliance

- **No external data or APIs.** Only the provided TSVs are used. Normalisation dictionaries
  (abbreviations, legal forms, state codes) are hand-written rules in `normalize.py`. The code makes
  no network calls. The one exception is the optional embedding model, which is downloaded once.
- **Model licence and size.** LightGBM (MIT). The optional embedding model
  `intfloat/multilingual-e5-small` is MIT-licensed with about 118M parameters, well inside the
  "MIT/Apache-2.0, ≤ 8B parameters" rule.
- **Open-set country.** Country is only used as a "same label" feature. Nothing is hard-coded to
  US or India, and every test Source-1 entity always gets a row.
- **Submission format.** Tab-separated files, one row per Source-1 entity, S2/S3 ids only, no
  duplicates, and matches are always a subset of the candidates. `validate` checks all of this.
