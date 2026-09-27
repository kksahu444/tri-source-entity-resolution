# Experiment and Submission Log

Rules:
1. Every idea is measured on **OOF macro F0.5** (`artifacts/train_report.json`) before it reaches the leaderboard.
2. Every leaderboard upload gets a git tag `sub/d<day>-<n>` and a row in the submission table.
3. Record the blocking numbers whenever blocking changes.

## Blocking runs

| date/time | commit | config change | pair recall | recall S2 / S3 | cand/S1 | upper-bound F | notes |
|---|---|---|---|---|---|---|---|
| | | default | | | | | |

## Model / decision runs (OOF)

| date/time | commit | change | stage-1 OOF | stage-2 OOF | decision | per-country | notes |
|---|---|---|---|---|---|---|---|
| | | baseline | | | | | |

## Leaderboard submissions (5 per day)

Submissions come from the pipeline under `student_resource/code/business_entity_resolution/` (origin/adi).
Its per-run numbers are in `student_resource/runs/run_log.csv` (iteration 3: `runs/kaggle_iter3/run_log.csv`)
and the uploaded files in `student_resource/submissions/manifest.csv`. Day 1 = 25 Sep. Tags are proposed,
not pushed yet.

| tag | day | commit | run_id | config | OOF macro F0.5 | public LB | decision | ran by | notes |
|---|---|---|---|---|---|---|---|---|---|
| sub/d2-1 | 2 | 0beb90e (code 69f4c91) | 0926-073916-submit | config.yaml (hash 1087ee4653) | 0.97143 [0.97114, 0.97170] (`0926-032111-decide`) | 0.960 (2 decimals) | GFM + count model | Aditya | upload 1, iteration 1: C1+C2+C5 blocking, pruner N=8, stage A LightGBM, b_FR = 0; 25,722,619 candidate pairs; acceptance check pass (run after the fact) |
| sub/d2-2 | 2 | 341091a | probe-of-0926-073916-submit | - | - | 0.821 | - | Aditya | diagnostic: upload 1 with French ids scrambled, F_FR = 0.928 (`0926-144305-probe`); never a final candidate |
| sub/d2-3 | 2 | 3334842 (Kaggle run, code commit not logged) | kaggle-0926-121628-submit | config_v3.yaml + XGBoost CUDA (hash b96fc5c751) | 0.98049 [0.98025, 0.98073] (`kaggle-0926-102147-decide`) | 0.938 | GFM, independent | Aditya | upload 2, iteration 3: + C7, support/generator/consistency features, stage B; over-links test decoys (`runs/decoy_shift.md`); acceptance check fail |
| sub/d3-1 | 3 | 111652f (code 081b693 + bd71889) | v7b-0927-101300-submit | config_v7.yaml + XGBoost CUDA (Kaggle `amlc26-pipeline-v7b`) | 0.98060 [0.98037, 0.98083] on the decoy dev (`v7b-0927-074428-decide`); paired gain vs config_v5 +0.00073 [+0.00056, +0.00090] | **0.971** | two thresholds, stage B | Aditya | upload 3, iteration 5 + France fixes: synthetic decoys (suffix and word-changed), numbering markers and department aliases mined without labels, zero-padded numbers, C9, memory-lean stage A/B; 31,669,732 candidate pairs; validator, acceptance and France checks pass; best public upload so far |

Final ranking uses the private score of the best public upload (organiser answers 25 Sep 18:36, 26 Sep 12:56
and 17:03), so as of 27 Sep 00:45 upload 1 is the one that counts; a later upload counts only if it beats 0.960
public. Since 27 Sep 21:00 that is upload 3 (0.971).
