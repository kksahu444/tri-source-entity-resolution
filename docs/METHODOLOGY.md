# Methodology: Business Entity Resolution (Amazon ML Challenge 2026)

> **Draft.** Copy these sections into the organisers' `Documentation_template.md` when it is
> available, and replace every `TBD` with the final numbers from `artifacts/train_report.json`.

**Team:** TBD · **Final public LB (macro F0.5):** TBD · **OOF macro F0.5 (5-fold grouped CV):** TBD

## 1. Methodology overview

A four-stage pipeline: normalisation → blocking (candidate generation) → pairwise matching
(gradient-boosted trees) → an F0.5-aware decision rule. All learning uses only the provided training
labels. All text statistics are fitted unsupervised on the records of the split being processed.
No external data, APIs or lookups are used.

## 2. Normalisation

- Unicode/accent folding, lower-casing, `&` → `and`, apostrophes removed, punctuation → space, runs of
  single letters joined (`A.B.C.` → `abc`).
- **Names:** canonical short forms for legal and business words (Private/Pvt/Pvte → `pvt`,
  Incorporated → `inc`, Enterprises → `ent`, Shri/Shree/Sree → `sri`, …). French forms (SARL, SAS, SA,
  EURL, SCI) are handled the same way. A *core name* has legal forms and stop words removed. DBA /
  "trading as" names are split into variants. A consonant *skeleton* per token absorbs transliteration
  (`Aggarwal`/`Agarwal`, `Laxmi`/`Lakshmi`).
- **Addresses:** canonical street types (Street/Str/St → `st`, Road → `rd`, Boulevard/Bd → `blvd`,
  Rue, Chemin, …) and state names → state codes. Landmark phrases (`Near/Nr/Opp/Behind …`) are
  split out of the *core address*. Postcodes (US ZIP, ZIP+4, Indian PIN incl. `560 001`, French CP)
  and house numbers are extracted.
- **Country** is treated as an open label set and only used for a "same country" feature.

## 3. Candidate generation / blocking strategy

For each Source-1 record, candidates are retrieved **separately from Source 2 and Source 3** and
unioned:

| blocker | representation | k per source |
|---|---|---|
| name_char | TF-IDF char 2–4-grams of the core name | TBD (default 15) |
| name_word | TF-IDF of name tokens + phonetic skeleton tokens | TBD (10) |
| full_char | TF-IDF char 3–4-grams of core name + core address | TBD (15) |
| addr_word | TF-IDF word 1–2-grams of the core address | TBD (8) |
| keys | exact: postcode + first name skeleton, acronym + postcode, full skeleton, exact address (blocks larger than 30 skipped) | all |

Cosine top-k uses chunked sparse matrix products. `candidate_pairs.tsv` is exactly the union above,
which is the set the model scores.

| blocking metric (train) | value |
|---|---|
| pair recall (all / S2 / S3) | TBD |
| reduction ratio | TBD |
| mean candidates per S1 | TBD |
| macro F0.5 upper bound (perfect matcher on these candidates) | TBD |

## 4. Model architecture and feature engineering

**Model:** LightGBM binary classifier (MIT licence), 5-fold `GroupKFold` grouped by Source-1 id,
early stopping on the held-out fold. Test predictions average the fold models. An optional stage-2
LightGBM adds rank/gap features of the stage-1 out-of-fold probability within each Source-1 group
and each candidate group. It is kept only if it improves OOF macro F0.5.

**Features (about 100):**
- *Name:* rapidfuzz ratio, partial, token-sort and token-set ratios, Jaro-Winkler, normalised
  Levenshtein (on full and core names); TF-IDF cosines (char and word); IDF-weighted fuzzy token
  Jaccard, coverage of each side, unmatched IDF mass, rarest shared token; acronym match; legal-form
  agreement and conflict; digit conflict; best similarity over DBA variants; lengths.
- *Address:* fuzzy ratios on core and full address; IDF-weighted overlap and coverage (partial or
  landmark-only addresses); postcode both / match / conflict / same 3-digit prefix; house-number
  match / conflict / Jaccard / first-number equality; landmark flags; TF-IDF cosines.
- *Context:* for key similarities, rank and gap to best within the Source-1 group, within
  (Source-1, source), and within the candidate's competing Source-1 records; group sizes.
- *Provenance:* which blockers proposed the pair and their similarities.
- *Country:* same-label flag only.

Top features by gain: TBD (from `artifacts/feature_importance_stage1.tsv`).

## 5. Decision rule (precision-heavy metric)

1. **Exclusivity:** each Source-2/3 record is kept only for the Source-1 entity with the highest
   probability, because Source 1 is deduplicated.
2. **Selection:** either a global probability threshold, or per entity the top-k (k ≥ 0) that
   maximises expected F0.5, where k = 0 wins when P(singleton) = Π(1 − pᵢ) is higher.
3. Strategy, threshold and exclusivity are grid-searched on out-of-fold probabilities.
   Selected: TBD.

## 6. Validation and results

| experiment | OOF macro F0.5 | public LB |
|---|---|---|
| baseline | TBD | TBD |
| … | | |

- Per-country OOF: TBD. Singletons vs non-singletons: TBD.
- Leave-one-country-out (proxy for unseen France): TBD.

## 7. Other relevant information

- **Compliance:** only provided data; hand-written normalisation dictionaries; no geocoding, registry
  or ER APIs. Models: LightGBM (MIT). Optional embeddings: `intfloat/multilingual-e5-small`
  (MIT, ~118M params).
- **Reproducibility:** `pip install -r requirements.txt && python main.py all` regenerates both output
  files. The resolved config is saved in `artifacts/model.joblib`. Seeds are fixed.
- **Runtime:** TBD on TBD hardware.
