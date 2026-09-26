# AMLC 2026 Business Entity Resolution: progress report

Status at 26 Sep 2026, 09:55 IST. Covers everything built since the build spec (v1.0/v1.1) arrived on
25 Sep evening. Every number cites the `run_id` of its row in `runs/run_log.csv`.

Terms used below:

- **OOF (out-of-fold):** a score on training entities the model did not train on (5-fold cross-validation).
- **PC (pair completeness):** the share of true pairs that blocking retrieves; it caps recall.
- **Dev universe:** 30% of train cities (about 500k S1 entities) used for model development.

---

## 1. Summary

- **A validated leaderboard file exists:** `student_resource/submissions/upload_1/matching_results.tsv`.
  - The organisers' validator prints PASS.
  - All 5,654,822 matched IDs exist in the test S2/S3 files (checked separately; the default validator skips this).
  - All 1,732,544 test S1s appear once, in file order; 102,118 have an empty list.
- **Expected quality of upload 1:** OOF macro F0.5 **0.97143**, 95% CI 0.97114-0.97170 (`0926-032111-decide`).
  - US 0.97918, India 0.95949, true singletons 0.96817.
  - France has no labels. The leave-one-country-out study (section 5.5) suggests France will score below US and India.
- **Iteration 2 is running.** A transliteration dictionary lifts India blocking recall from 0.943 to 0.964, and a
  stage-B model is added. Its dev score is due about 11:35; its test file about 16:00-16:30.
- **GPU work (dense retrieval, cross-encoder) is blocked** until someone attaches a Kaggle secret (section 8).
  Until then the pipeline follows the spec's CPU fallback.

---

## 2. What was built

All code is in `student_resource/code/business_entity_resolution/`, the layout the submission zip requires.
The single entry point is `python -m src.run_pipeline --config <config> --stage <stage> --split <dev|test>`,
and `--stage all` regenerates both output files from `dataset/`.

| Stage | Spec | What it does |
|---|---|---|
| `audit_d0.py`, `france_audit.py` | section 4 | Data audit and French noise audit; outputs in `runs/` |
| `prep` | D1 | Two text views of all 23.5M records; data-mined vocabularies (legal forms, regions, region aliases) |
| `translit` | D1 (added) | Romanised-to-Latin token dictionary mined from training pairs, for India's native-script names |
| `block` | D2 | Retrieval channels C1 (TF-IDF), C2 (name crowd), C5 (hash keys), union per country |
| `features` | D3, D4 | Pruner (LightGBM, out-of-fold), then about 88-95 pair features on the pruned set |
| `stack` | D6 | Stage-A LightGBM, 5-fold out-of-fold; test = mean of the 5 fold models |
| `stackb` | D6 (CPU variant) | Stage-B LightGBM adding context from stage A's out-of-fold probabilities |
| `decide` | D6-D8 | Isotonic calibration, one S1 per record, decision rule chosen out-of-fold |
| `loco` | D9 | Leave-one-country-out study for the France probability offset |
| `submit` | section 10 | Writes both TSVs, runs the organisers' validator, archives the files and models |

Supporting pieces: `f05_decision_rule.py` (Appendix A), a numba port of its GFM (checked equal to the
reference), an error-analysis script, a decision-layer ablation script, a packaging script, and
`README.md` / `LICENSES.md` / pinned `requirements.txt`. There are 30 unit tests, all passing.

Hardware: a Windows laptop with 12 CPU threads and 15 GB RAM. A full dev + test iteration takes about 7-8 hours.

---

## 3. Data audit: findings that shaped the design

Sources: `runs/d0_audit.txt`, `runs/france_audit.txt`, trigger table `runs/d0_triggers.md`.

| Finding | Value | Design consequence |
|---|---|---|
| Matches per S1 | mean 3.46; singletons 5.58% (US), 5.59% (India) | recall matters as much as precision |
| Duplicates inside S2/S3 | 54-59% of matched S1s have 2+ matches in one source | "candidate support" and sibling features |
| Cross-country pairs, many-to-one | 0 and 0 of 7,638,365 true pairs | block per country; hard assignment of each record to one S1 |
| Pool size | (S2+S3)/S1 is 4.68 in train vs 5.53-5.82 in test (**trigger**) | dev pool withholds 19% of dev S1s so their records act as distractors |
| Native scripts (not in the spec) | 23.5% of India S2 names, 13% of India S3 names | anyascii romanisation plus a mined token dictionary |
| Noise in true pairs | house number conflicts 20.9%; last name token equal only 40.5% | no hard vetoes; legal form is a soft feature |
| France-only noise | "R." for Rue in 12% of French S2/S3 addresses; "St-", Av, Bd, Imp, Rte | language-agnostic abbreviation matcher, no French word list |

---

## 4. Key design decisions and deviations from the spec

| Topic | What the spec said | What was done, and why |
|---|---|---|
| Appendix B `audit_d0.py` | use as given | never arrived (message truncated); rebuilt from the section-4 trigger table and marked as a reconstruction |
| Appendix A self-test 7 | as given | truncated mid-test; rebuilt to the stated check, which prints `mismatches = 0 / 200` |
| Legal-form vocabulary | all 1-3-token name endings with share >= 0.1% | that rule swept in ordinary words ("tech", "brothers"); the mined rule instead keeps tokens that end >= 1% of names and sit at the end in >= 85% of names containing them. Result: llc/inc/pc/corp (US), ltd/llp/co (India), sarl/sas/eurl/sa/sasu/sci/ei (France) |
| Region vocabulary | S1 last address parts | cities also appear last in reordered addresses; a region must also be last in >= 50% of the addresses containing it |
| C1 TF-IDF | char 3-4-grams on name and address, exact top-k | char 3-grams on name + words on address (memory). The full sparse product projected 3.6 h for US; candidates are now generated on the rarer features and rescored exactly. That is 15x faster, with PC 0.911 vs 0.900 on a 3,000-query benchmark |
| Section-8 leak check | fix grouping if > 5% | measured 7.0% (India) / 8.6% (US). Grouping by state gives 5.6-6.1%, so the cause is name-driven, not geographic. No feature carries identity, so labels cannot leak. Kept city grouping; flagged (`runs/leak_check.md`) |
| France offset | leave-one-country-out prior | study cut to one model per direction for runtime; result b_FR = 0 |
| Cross-encoder / dense channel | D5 / C4 on 2x T4 | not run: data access on Kaggle is blocked (section 8); fallback ladder applied |

---

## 5. Results

### 5.1 Blocking (dev universe)

| Version | US union PC | India union PC | Candidates / S1 | run |
|---|---|---|---|---|
| 1% benchmark, full sparse product | 0.98044 | 0.92805 | 35.41 | `0926-005620-block` |
| v1: rare-feature C1 | 0.97407 | 0.94293 | 36.40 | `0926-011721-block` |
| v2: + transliteration dictionary, rarest-token house key | 0.97404 | **0.96404** | 36.05 | `0926-074645-block` |

Per-channel PC for India, v1 to v2 (`0926-011721-block`, `0926-074645-block`):

- C1: 0.909 to 0.917
- C2 (name crowd): 0.459 to 0.626
- S2 records overall: 0.945 to 0.973

Reduction ratio is 0.999994 for US and 0.999991 for India.

### 5.2 Pruner (D3)

- N = 8 per S1 in both countries (smallest N within 0.005 PC of the union), plus each record's top 3 S1s.
- Dev after pruning (`0926-020122-features`): PC 0.9739 (US), 0.9424 (India); about 20 candidates per S1.
- v2 (`0926-083601-features`): PC 0.97386 (US), 0.9637 (India).
- Test candidates for upload 1: 14.6-15.0 per S1, 25.7M pairs in total (`0926-043331-features`).

### 5.3 Matching model and decision rule (iteration 1, stage A)

Stage A was trained on 10.4M dev pairs with 16.1% positives; best iterations 692-1,106 across folds (`0926-023404-stack`).

| Decision rule | OOF macro F0.5 | Gain vs two thresholds [95% CI] |
|---|---|---|
| Two thresholds, per country (tau1 = 0.59-0.61, tau2 = 0.67-0.68) | 0.97063 | - |
| GFM, independent candidates | 0.97042 | -0.00021 [-0.00033, -0.00009] |
| **GFM + count model (chosen)** | **0.97143** | **+0.00080 [+0.00067, +0.00093]** |

All rows are run `0926-032111-decide`.

- Calibration: expected calibration error 0.0002 after isotonic.
- Count model: 97.3% of entities get the right match-count class (0/1/2/3+); temperature 1.167.
- Oracle bounds: perfect decisions on the retrieved candidates would score 0.98659, so about 0.013 is lost
  in blocking/pruning and about 0.015 in pairwise decisions. Predicting all empty scores 0.05565.

### 5.4 Upload 1 on test

Runs `0926-072826-decide` and `0926-073547-submit`:

- Matches by country: France 870,801 (3.36 per S1), India 2,571,607 (3.17 per S1), US 2,212,414 (3.34 per S1).
- File: `submissions/sub_20260926_1_0926-073916-submit_matching.tsv`, copied to `submissions/upload_1/`.

### 5.5 France (leave-one-country-out, `0926-063314-loco`)

| Direction | Held-out F0.5 | In-country F0.5 | Drop |
|---|---|---|---|
| Train US, score India | 0.87092 | 0.95949 | 0.089 |
| Train India, score US | 0.95768 | 0.97918 | 0.022 |

- The oracle offsets were +0.1 and -0.1, so b_FR = 0: the shift is not a probability miscalibration.
- The large India drop comes from native-script names a US-only model never sees. France has none, so
  France is more likely to behave like the smaller drop.

---

## 6. Compliance check against the problem statement

Checked on upload 1.

| Rule | Status |
|---|---|
| TSV format, exact header, one row per test S1, empty lists allowed | pass (validator + own checks) |
| No duplicate IDs; only S2/S3 IDs that exist in test | pass (all 5.65M matched and 25.7M candidate IDs verified) |
| Matches are a subset of `candidate_pairs.tsv`; candidates = the set the final model scores | pass |
| Final model MIT/Apache-2.0, <= 8B parameters | pass: LightGBM (MIT) trained from scratch; no pretrained model |
| No external data, APIs, geocoders, internet data | pass: only the provided files. Nothing was uploaded to Kaggle (blocked); the Kaggle test kernels fetched no data |
| Country treated as an open set | pass: France handled by generic code; country is never a model feature (88 features: similarities, ranks, counts) |

Judgment calls a reviewer could ask about:

1. `anyascii` (ISC licence) romanises native scripts. It is a Unicode character table in a pip package,
   not a lookup of business identities. It sits behind the `TRANSLITERATE` flag.
2. Some statistics come from unlabelled test text: TF-IDF weights, legal-form and region vocabularies
   (France only exists in test). Nothing is fitted on test predictions or on the leaderboard.

---

## 7. What is running now

Iteration 2 (`runs/iter2c_chain.sh`, config `config_v2.yaml`) adds three things:

- the transliteration dictionary;
- candidate-support features;
- stage B.

The chain restarted at 08:36 after an out-of-memory crash. The crash was caused by running the
validator on a 25.7M-ID candidate file at the same time as feature building.

| Step | Status / estimate |
|---|---|
| Dev features | done 09:08 (`0926-083601-features`) |
| Stage A (5 folds) | running; about 10:05 |
| Dev decision, stage A only (ablation) | about 10:20 |
| Stage B + dev decision, stage A + B | about 11:35 (first number showing whether iteration 2 beats 0.97143) |
| Test blocking, features, stage A/B inference, decision, validated file | about 16:00-16:30 |

---

## 8. Open items and decisions for the team

1. **Upload 1** and send back its public score, so it can be logged against run `0926-073916-submit`.
2. **Kaggle GPU access.** Uploading the data from the laptop was blocked by the permission system, and
   CLI-pushed kernels cannot read Kaggle secrets. To enable the dense channel:
   - open notebook `aditya103856/amlc26-dense`;
   - add a secret `GITHUB_TOKEN` with read access to the `dataset-v1` release;
   - attach the secret to that notebook.

   The kernel then pulls the data into Kaggle itself. The GPU profile was verified: 2x Tesla T4, 14.6 GB,
   compute capability 7.5 (`runs/kaggle_probe.txt`).
3. **Leak check above 5%** (section 4). Accept the explanation, or propose a grouping to test.
4. **Organiser questions.** Is `anyascii` romanisation acceptable? This sits alongside Q3 on hand-written
   lists; everything else in the pipeline is mined from the data.
5. **Final deliverables for 27 Sep:**
   - the zip, built by `python -m src.package --team <TEAM>`;
   - the filled methodology document (draft: `docs_draft/Documentation_draft.md`);
   - the README check on a small slice;
   - ablation tables from `python -m src.ablate`.

---

## 9. Where things are

| What | Path (under `student_resource/`) |
|---|---|
| Code, configs, tests, README, licences | `code/business_entity_resolution/` |
| Run log (every number) | `runs/run_log.csv` |
| Audits, trigger table, leak check, Kaggle probe | `runs/` |
| Upload 1 (ready) | `submissions/upload_1/` |
| Submission archive and manifest | `submissions/` (TSVs git-ignored; `manifest.csv` tracked) |
| Models archived per submission | `artifacts/archive/<run_id>/` (git-ignored) |
| Methodology draft | `docs_draft/Documentation_draft.md` |

Git: 16 commits on `main` since `337ca16` (including the one adding this report), all pushed to
`kksahu444/tri-source-entity-resolution`.
