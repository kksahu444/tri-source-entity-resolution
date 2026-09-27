# Business entity resolution: reproduction guide

Pipeline for the Amazon ML Challenge 2026 task: for every test Source-1 entity, list the Source-2/3
records of the same business. Every number quoted in the methodology document comes from
`runs/run_log.csv` and cites its `run_id`.

## Layout

```
student_resource/
  dataset/{train,test}/            organiser data (not shipped)
  utils/validate_submission.py     organiser validator
  code/business_entity_resolution/ this folder (src/, tests/, kaggle/, config*.yaml, requirements.txt)
  artifacts/                       parquet intermediates and models (not shipped)
  runs/                            run_log.csv, audits, stage reports (decoy_shift.md, acceptance.jsonl, ...)
  output/                          matching_results.tsv, candidate_pairs.tsv
  submissions/                     every uploaded file plus manifest.csv
```

## Environment

Python 3.12. The tree-model pipeline runs on CPU (LightGBM). With a CUDA GPU the tree models can use
XGBoost (`model: {backend: xgboost_gpu}`), and the optional cross-encoder stage needs one or two GPUs
(it was run on Kaggle with 2x T4, see `kaggle/runner/`).

```bash
pip install -r requirements.txt            # torch + transformers only for the cross-encoder stage
python -m pytest -q tests
python src/decision/f05_decision_rule.py   # must print "count-tilted GFM vs brute force: mismatches = 0 / 200"
```

## Full run (final configuration)

```bash
python -m src.run_pipeline --config config_v7.yaml --stage all
```

`config_v7.yaml` produced the final file (upload 3, Kaggle kernel `amlc26-pipeline-v7b`; its run log is
`runs/kaggle_v7b/run_log.csv`). `config_v7ce.yaml` (+ cross-encoder) and `config_v8.yaml` (+ test-like dev
pool, 45% of the cities) are later variants that did not finish before the deadline.

`--stage all` runs: prep, translit, then on the dev universe block, features, stack, [ce], stackb, decide,
then the same on test, then submit (writes `output/` and runs the organiser validator). The final file
must then pass the pre-upload checks before it is uploaded:

```bash
python -m src.acceptance --config config_v7.yaml --file ../../output/matching_results.tsv     # decoy shape vs train truth
python -m src.france_checks --config config_v7.yaml --matching ../../output/matching_results.tsv \
    --candidates ../../output/candidate_pairs.tsv                                             # label-free, vs upload 1
python -m src.compare_runs --a <runs dir A> --b <runs dir B>                                  # paired dev gain
```

`python -m src.stage_upload` runs the validator, the acceptance check and the France checks on a downloaded
Kaggle run and copies it to `submissions/` with a manifest row.

Each stage can also run alone, e.g. `python -m src.run_pipeline --config config_v7.yaml --stage block --split dev`.
Full run times: 262 minutes for `config_v7.yaml` on a Kaggle 2x T4 session (XGBoost CUDA, 4 cores, 31 GB);
a 4-core CPU session with LightGBM does not finish within Kaggle's 12 h limit. On Kaggle:
`kaggle/runner/run_on_kaggle.py` (reads the data and code from private datasets, writes `output/`, `runs/`
and small checkpoints to /kaggle/working; `ER_CONFIG` picks the config, default `config_v7.yaml`).
`kaggle/runner_tpu/` runs the same pipeline on a TPU VM (224 cores, CPU LightGBM) after installing
`requirements.txt` into a virtual environment, since that image has no LightGBM and ships pandas 3.

## 1% reproduction check

`config_repro1pct.yaml` is `config_v6.yaml` with a dev universe of 1% of the train city groups, its own
folders (`artifacts_repro1pct/`, `runs_repro1pct/`, `output_repro1pct/`) and no cross-encoder. It runs
the full dev path end to end on about 17k dev queries:

```bash
python -m src.run_pipeline --config config_repro1pct.yaml --stage prep
python -m src.run_pipeline --config config_repro1pct.yaml --stage translit
for s in block features stack stackb decide; do
  python -m src.run_pipeline --config config_repro1pct.yaml --stage $s --split dev
done
```

`runs_repro1pct/run_log.csv` then holds one row per stage, the last with the dev OOF macro F0.5
(noisier than the full run's, which uses 30% of the city groups). Prep normalises all six files and is the
slowest step (about 20 minutes with 10 workers).

## Stages

| Stage | What it does |
|---|---|
| prep | View A (NFKC + spaces) and view B (fold, junk flags, legal field, DBA split, order-invariant address parts) for all six files; legal forms, regions and region aliases mined from the data; from config_v7 also numbering markers ("no"/"N°" before the house number), region aliases for countries without training pairs (French departments) and zero-padded numbers |
| translit | Transliteration dictionary mined from aligned train pairs, applied to non-Latin S2/S3 names |
| block | Per country: C1 joint TF-IDF (rare-feature generation, exact rescoring), C2 name crowd, C5 keys, C7 empty-address name channel (reverse + forward), C8 address-only channel, C9 reverse channel for records no other channel retrieved (config_v7); on dev, synthetic decoy entities are added to the pool (`dev.decoys`, see runs/decoy_shift.md) |
| features | Pruner (OOF) to the top N per S1 (N = smallest within 0.005 PC of the union; 11 in the final run) plus each record's top 3 S1s; pair features on the pruned set (string, address, house-number, support, generator-aware, consistency) |
| stack | Stage A gradient boosting, 5-fold out-of-fold by city groups; test = per-fold predictions |
| ce | Optional cross-encoder on the uncertain band (mDeBERTa-v3-base vs XLM-R-base bake-off, 3 grouped folds); any failure leaves the pipeline running without it |
| stackb | Stage B: stage-A context features (+ cross-encoder logit when present), trained with and without it on the same folds |
| decide | Isotonic calibration, one S1 per record, rules (two thresholds, GFM, count-tilted GFM) scored per entity; variants stage A -> stage B -> cross-encoder, each adopted only if the paired bootstrap interval of its gain excludes zero |
| submit | Writer, organiser validator, archive under submissions/ |

Tools: `src.acceptance` (decoy-link check against train truth), `src.stage_upload` (download a Kaggle
run, validator + acceptance, copy to submissions/, manifest row), `src.error_analysis`, `src.transfer`,
`src.ablate`.

## Compliance

- No external data or services. Vocabularies (legal forms, regions, region aliases, numbering markers, name
  endings and name words for the synthetic decoys) are mined from the provided files; the only fixed lists are the problem
  statement's examples and the tag words the spec itself names.
- Non-Latin scripts (Devanagari, Kannada, Telugu, Tamil, Bengali, Gujarati in India S2/S3) are
  romanised with the `anyascii` library (ISC licence, a Unicode transliteration table, no model).
  Flag `TRANSLITERATE` in the config.
- The cross-encoder backbones (mDeBERTa-v3-base, XLM-R-base) are pretrained public models fine-tuned only
  on the provided training data. The cross-encoder is off in the final `config_v7.yaml`.
- Libraries and licences: see LICENSES.md.
