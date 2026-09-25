# Business entity resolution: reproduction guide

Pipeline for the Amazon ML Challenge 2026 task: for every test Source-1 entity, list the Source-2/3
records of the same business. Every number quoted in the methodology document comes from
`runs/run_log.csv` and cites its `run_id`.

## Layout

```
student_resource/
  dataset/{train,test}/            organiser data (not shipped)
  utils/validate_submission.py     organiser validator
  code/business_entity_resolution/ this folder (src/, tests/, config.yaml, requirements.txt)
  artifacts/                       parquet intermediates and models (not shipped)
  runs/                            run_log.csv, audits, stage reports
  output/                          matching_results.tsv, candidate_pairs.tsv
  submissions/                     every uploaded file, archived by run_id, plus manifest.csv
```

## Environment

Python 3.12, CPU only for the tree-model pipeline:

```bash
pip install -r requirements.txt
python -m pytest -q tests
python src/decision/f05_decision_rule.py     # must print "count-tilted GFM vs brute force: mismatches = 0 / 200"
```

## Commands (run from this folder)

```bash
# audits (from student_resource/)
python code/business_entity_resolution/src/audit_d0.py --root dataset > runs/d0_audit.txt
python code/business_entity_resolution/src/france_audit.py --root dataset > runs/france_audit.txt

# pipeline
python -m src.run_pipeline --stage prep                      # D1: vocabularies + normalised parquet
python -m src.run_pipeline --stage block --split dev --frac 0.01   # 1% benchmark, projected runtime
python -m src.run_pipeline --stage block --split dev         # D2 on the dev universe, pair completeness
python -m src.run_pipeline --stage features --split dev      # D3 pruner (OOF) + D4 features
python -m src.run_pipeline --stage stack --split dev         # D6 stage A, 5-fold OOF
python -m src.run_pipeline --stage decide --split dev        # D6-D8 calibration, assignment, thresholds, OOF F0.5
python -m src.run_pipeline --stage loco --split dev          # D9 leave-one-country-out prior for France
python -m src.run_pipeline --stage block --split test
python -m src.run_pipeline --stage features --split test
python -m src.run_pipeline --stage stack --split test
python -m src.run_pipeline --stage decide --split test
python -m src.run_pipeline --stage submit --split test       # writes output/, runs the validator (must PASS)
```

## Stages

| Stage | Spec | What it does |
|---|---|---|
| prep | D1 | View A (NFKC + spaces) and view B (fold, junk flags, legal field, DBA split, order-invariant address parts) for all six files; legal forms, regions and region aliases mined from the data |
| block | D2 | Per country: C1 joint TF-IDF (name char 3-grams + address words, exact top-k), C2 name crowd ranked by address, C5 hash keys (house + street, first name token + city); union with every channel's score and rank |
| features | D3, D4 | Pruner LightGBM on cheap features (OOF), top-N per S1 plus each record's top 3 S1s; full pair features on the pruned set |
| stack | D6 | Stage-A LightGBM, 5-fold out-of-fold by city groups; test = mean of the fold models |
| decide | D6-D8 | Isotonic calibration, one S1 per record, two-threshold rule per country (France: pooled thresholds + offset) |
| loco | D9 | Leave-one-country-out oracle offsets; b_FR = their mean |
| submit | 10 | Writer, organiser validator, archive under submissions/ |

## Compliance

- No external data or services. Vocabularies (legal forms, regions, region aliases) are mined from the
  provided files; the only fixed lists are the problem statement's examples and the tag words the
  spec itself names.
- Non-Latin scripts (Devanagari, Kannada, Telugu, Tamil, Bengali, Gujarati in India S2/S3) are
  romanised with the `anyascii` library (ISC licence, a Unicode transliteration table, no model).
  Flag `TRANSLITERATE` in config.yaml. Open question for the organisers alongside Q3.
- Libraries and licences: see LICENSES.md.
