# Experiment and Submission Log

Rules:
1. Every idea is measured on **OOF macro F0.5** (`artifacts/train_report.json`) before it reaches the leaderboard.
2. Every leaderboard upload gets a git tag `sub/d<day>-<n>` and a row in the submission table.
3. Record the blocking numbers whenever blocking changes.

## Blocking runs

Dev-universe pair completeness from `student_resource/runs/run_log.csv` (Kaggle rows from
`runs/kaggle_iter3/run_log.csv`, cited `kaggle-`).

| run-log timestamp | commit | run_id | change | PC US | PC India | cand/S1 | ran by | notes |
|---|---|---|---|---|---|---|---|---|
| 26 Sep 01:05 | 2c2274b | 0926-005620-block | 1% benchmark, full sparse product | 0.98044 | 0.92805 | 35.41 | Aditya | block_bench slice, 567 s |
| 26 Sep 01:46 | 2e372f1 | 0926-011721-block | v1: rare-feature C1 generation | 0.97407 | 0.94293 | 36.40 | Aditya | full dev, 1,749 s |
| 26 Sep 08:13 | 5edc821 | 0926-074645-block | v2: + transliteration dictionary, rarest-token C5a key | 0.97404 | 0.96404 | 36.05 | Aditya | India +0.021 |
| 26 Sep 09:33 | not logged (Kaggle) | kaggle-0926-091122-block | v3: + C7 reverse, k = 10 | 0.98727 | 0.97591 | 42.46 | Aditya | C7 alone 4.2% (US) / 3.5% (India) of true pairs |
| 26 Sep | a4839a3 | `runs/c7_scale.md` | C7 recall vs S1 pool size, reverse k = 10 vs forward k = 5 | - | - | - | Ashutosh | study, not a pipeline run: at test-sized pools both directions add 12 pt (India) and 7 pt (US) on blank-address targets; led to `forward_k: 5` in config_v5 |

## Model / decision runs (OOF)

Macro F0.5 over all dev-universe S1s with 1,000-resample entity bootstrap intervals, from the `decide` rows.

| run-log timestamp | commit | run_id | change | OOF macro F0.5 [95% CI] | US | India | rule | ran by | notes |
|---|---|---|---|---|---|---|---|---|---|
| 26 Sep 04:33 | 69f4c91 | 0926-032111-decide | v1 stage A | 0.97143 [0.97114, 0.97170] | 0.97918 | 0.95949 | GFM + count model | Aditya | +0.00080 over two thresholds; became upload 1 |
| 26 Sep 10:01 | 04669a9 | 0926-095450-decide | v2 stage A: + transliteration, support features | 0.97745 [0.97720, 0.97771] | 0.98009 | 0.97337 | GFM + count model | Aditya | India +0.014 |
| 26 Sep 10:50 | 76ef313 | 0926-103648-decide | v2 stage B (context stacking) | 0.97898 [0.97876, 0.97923] | 0.98167 | 0.97484 | GFM, independent | Aditya | count model stops helping once stage B is in |
| 26 Sep 10:33 | not logged (Kaggle) | kaggle-0926-102147-decide | v3: + C7, generator/consistency features, stage B, XGBoost CUDA | 0.98049 [0.98025, 0.98073] | 0.98308 | 0.97649 | GFM, independent | Aditya | became upload 2 (public 0.938, decoy over-linking) |
| 27 Sep ~03:50 | - | Kaggle `amlc26-pipeline` v5 and v6 | config_v5/v6: decoy dev, C7 forward, C8, cross-encoder | - | - | - | - | Aditya | both died of memory on the 31 GB sessions (stage B test held ~3 x 10 GB for India; fixed in bd71889), no file; cross-encoder bake-off fold 0: mDeBERTa-v3-base band log-loss 0.156, XLM-R-base 0.165; the stage took 3 h 10 min (`runs/france_research.md` section 6) |
| 27 Sep 10:00 start | 081b693 | Kaggle `amlc26-pipeline-v7` | config_v7: v6 + France parse fixes, zero strip, C9; cross-encoder off | [pending] | | | | Aditya | 2x T4, XGBoost GPU; started before the memory fix bd71889 (11:21) |

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

Final ranking uses the private score of the best public upload (organiser answers 25 Sep 18:36, 26 Sep 12:56
and 17:03), so as of 27 Sep 00:45 upload 1 is the one that counts; a later upload counts only if it beats 0.960
public.
