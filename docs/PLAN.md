# Execution Plan: Amazon ML Challenge 2026 (Business Entity Resolution)

**Window:** 25 Sep 2026 00:00 IST to 27 Sep 2026 23:59 IST (72 h)
**Leaderboard budget:** 5 uploads per day, 15 in total
**Final ranking:** private leaderboard (the part of the test set held back from the public board)

---

## 1. The problem in one paragraph

We get three sources of business records (`entity_id, business_name, business_address, country`).
Source 1 is clean and deduplicated. For each Source-1 entity we must list every Source-2/3 record
that is the same business. There can be 0, 1 or many. The data is noisy: legal suffixes, abbreviations,
typos, transliterations, DBA names, reordered words, missing PIN/state, and landmark-only addresses
("Near SBI ATM"). Train covers **US + India**. Test adds **France**, which never appears in train.
We submit `matching_results.tsv`, the only scored file, and `candidate_pairs.tsv`, which is used to
audit our blocking. At the end we also submit a zip with our code and a methodology document.

## 2. What the metric rewards (and what that implies)

| fact | implication for us |
|---|---|
| Macro average over **Source-1 entities** | Every entity has the same weight, however many matches it has. Big groups do not dominate. |
| **Singletons** score 1.0 for an empty list and 0.0 for any prediction | Predicting "no match" is a real decision. First compute the all-empty baseline (= singleton rate). |
| F0.5: precision weighs 2× | One wrong extra match on a 1-match entity gives 0.556. Missing one of 2 true matches gives 0.833. **When unsure, do not merge.** |
| Blocking sets the recall ceiling | Track `upper_bound_macro_f` from `python main.py block`. Aim for ≥ 0.98 pair recall. |
| Source 1 is deduplicated | A given S2/S3 record belongs to at most one S1, so assign it only to its best S1 (**exclusivity**). Check this with EDA. |
| France only in test | No country-specific logic. Use `loco` to estimate the drop on an unseen country. Prefer robust, language-agnostic features. |
| Public LB is a subset of test | Trust out-of-fold (OOF) scores, not public LB swings of ±0.005. |

## 3. Architecture (implemented in this repo)

1. **Normalise** (`normalize.py`): ASCII folding, `&`→`and`, joining initials (`A.B.C.`→`abc`), canonical
   short forms for legal words and street types, DBA split into name variants, consonant skeletons for
   transliterations (`Agarwal/Aggarwal`→`agrvl`), landmark segments split from the core address,
   postcode and house-number extraction.
2. **Block** (`blocking.py`): top-k TF-IDF neighbours per S1 in S2 and in S3, for four representations,
   plus exact keys (postcode+name skeleton, acronym+postcode, full skeleton, exact address). Union →
   `candidate_pairs.tsv`.
3. **Features** (`features.py`): about 100 features: rapidfuzz scores, TF-IDF cosines, IDF-weighted fuzzy
   token coverage, postcode/number conflicts, and group-relative ranks (how a pair ranks among its
   S1's candidates and among the S1s competing for the same candidate).
4. **Model** (`model.py`): LightGBM with GroupKFold by S1 id. Stage 2 re-scores using the stage-1
   probability ranks and is kept only if OOF improves.
5. **Decide** (`decide.py`): exclusivity plus either a global threshold or per-entity expected-F0.5
   top-k (k=0 allowed). A grid search on OOF picks the rule.

## 4. Timeline

### Day 1: Thu 25 Sep (baseline on the board)

| time (IST) | task | output / gate |
|---|---|---|
| 00:00–01:30 | Download data into `dataset/`, `student_resource/` → `python main.py eda` | EDA numbers table (section 8) filled in |
| 01:30–03:00 | `python main.py block`: tune `*_k` until pair recall ≥ 0.98 with ≤ ~100 candidates per S1 | blocking report in EXPERIMENTS.md |
| 03:00–05:00 | `python main.py train` → first OOF score, check per-country and singleton breakdown | OOF macro F0.5 |
| 05:00 | **Sub 1**: baseline model. Run `validate --official` before uploading. | public LB vs OOF gap |
| day | Error analysis on `artifacts/oof_scores.tsv`: top false positives and false negatives by pattern | list of new rules and features |
| day | `python main.py loco` → how much we lose on an unseen country | France risk estimate |
| evening | **Subs 2–3**: normalisation and feature fixes from the error analysis | |
| optional | **One sub** of all-empty predictions tells us the test singleton rate on the public subset. Use it only if the singleton rate varies a lot across train countries. | |

### Day 2: Fri 26 Sep (improve)

Run the backlog in section 6 in priority order. Each experiment runs on OOF first. Only experiments that
gain ≥ 0.002 OOF get a leaderboard upload. Use up to 5 uploads today on the largest confirmed gains.

### Day 3: Sat 27 Sep (freeze and ship)

| time (IST) | task |
|---|---|
| until 16:00 | last experiments (low risk only), seed-averaged final model |
| 16:00 | **Code freeze.** Pick the final model by OOF (tie-break with public LB). |
| 16:00–19:00 | Fill `docs/METHODOLOGY.md` (the organisers' `Documentation_template.md`). Re-run from a clean venv: `pip install -r requirements.txt && python main.py all`. |
| 19:00–21:00 | `python main.py package --team <TEAM>` → check the zip layout → upload |
| by 22:00 | Final leaderboard upload(s). Keep 2 h of buffer before 23:59. |

## 5. Team roles (3–4 people)

| role | owns | success metric |
|---|---|---|
| **Blocking and normalisation** | `normalize.py`, `blocking.py` | pair recall, upper-bound F, candidates per S1 |
| **Features and model** | `features.py`, `model.py`, hyper-parameters | OOF macro F0.5 |
| **Decision and robustness** | `decide.py`, error analysis, LOCO/France checks | OOF gain from decision rules, LOCO gap |
| **Release manager** | submissions, `EXPERIMENTS.md`, git tags, docs, packaging | no failed uploads, reproducible zip |

With 3 people, the release manager also takes decision and robustness.

**Versioning rule (the guidelines require submission history):** every leaderboard upload gets a
commit plus a tag `sub/d<day>-<n>` (for example `sub/d1-1`) and a row in `docs/EXPERIMENTS.md` with
the OOF score, public LB score and config.

## 6. Experiment backlog (ordered by expected gain / effort)

| # | idea | why | effort |
|---|---|---|---|
| 1 | Tune blocking k values from EDA sizes, add blockers where recall is lost | Blocking caps recall | S |
| 2 | Error-driven normalisation rules (new abbreviations, legal forms, city variants, French forms) | Cheap precision and recall gains | S |
| 3 | Check exclusivity with EDA; if S2/S3 records can map to several S1s, loosen to a top-2 or margin rule | Removes look-alike false merges | S |
| 4 | **S2 ↔ S3 transitivity**: match S2–S3 too. Add S3 to an S1 when its S2 match strongly agrees with it. Add "linked via other source" features | Recovers landmark-only and heavily noisy records | M |
| 5 | Dense multilingual embeddings (`use_embeddings: true`, e5-small, MIT) as extra blocker and feature | Semantic and cross-lingual similarity for France | M |
| 6 | Fine-tuned cross-encoder on top-k pairs (e.g. MiniLM / multilingual-e5-small, ≤ 8B, MIT/Apache) as a feature | Ditto-style PLM matchers are strong on dirty data | L |
| 7 | Hyper-parameter search (num_leaves, min_data_in_leaf, feature_fraction), 3–5 seed average | Stability on the private LB | S |
| 8 | Per-entity decision calibration: isotonic calibration of OOF probabilities before expected-F | Better k=0 (singleton) decisions | S |
| 9 | Country-shift hardening: train with country-dropout copies of features, check with LOCO | France generalisation | M |
| 10 | Small open LLM re-ranker (≤ 8B, Apache/MIT, such as Qwen2.5-7B-Instruct, Apache-2.0) only for borderline pairs | Hard cases; expensive, do last | L |

## 7. Risks and mitigations

| risk | mitigation |
|---|---|
| Format rejection wastes an upload | `predict` validates automatically, and we always run `validate --official` first |
| Public LB overfitting | Choose by OOF. Upload only changes with OOF gain. Keep 5-fold grouped CV. |
| France distribution shift | No country branching. LOCO check. Accent folding. French legal and street dictionaries. |
| Test much larger than train (runtime) | Chunked sparse top-k. Features are vectorised (rapidfuzz `cpdist`). `max_candidates` cap. |
| Rule violation (external data) | No network calls in the pipeline. Dictionaries are hand-written. Documented in METHODOLOGY. |
| Plagiarism / code leak during the contest | **Keep the GitHub repo private until results are announced.** |
| Last-minute failure | Freeze at 16:00 on day 3. Zip built and checked by 21:00. |

## 8. EDA numbers to fill on day 1

| quantity | train value | decision it drives |
|---|---|---|
| # S1 / S2 / S3 records | | runtime, k values |
| singleton rate | | all-empty baseline, how conservative the threshold should be |
| matches per S1 (distribution) | | expected-F vs threshold |
| S2 vs S3 share of true pairs | | per-source k |
| S2/S3 ids matched to more than one S1 | | exclusivity on or off |
| cross-country true pairs | | whether blocking may stay within one country |
| % addresses with postcode / landmark | | weight of address blockers |
| blocking pair recall at default k | | blocking tuning |

## 9. Definition of done

- [ ] `output/matching_results.tsv` and `output/candidate_pairs.tsv` pass the official validator
- [ ] Best submission tagged in git, with the full ledger in EXPERIMENTS.md
- [ ] `docs/METHODOLOGY.md` filled in (methodology, blocking, model and features, other details)
- [ ] Clean-environment rerun reproduces the submitted files
- [ ] `<TEAM>_submission.zip` uploaded with the required layout
