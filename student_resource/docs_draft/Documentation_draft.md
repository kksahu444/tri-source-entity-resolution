# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [team name]
**Team Members:** [members]
**Submission Date:** 27 Sep 2026

Every number below cites the `run_id` of the row in `runs/run_log.csv` that produced it (or the audit
file it comes from). Out-of-fold (OOF) scores are macro F0.5 over all dev-universe Source-1 entities,
exactly as the leaderboard computes it, with 1,000-resample bootstrap intervals over entities.

---

## 1. Executive Summary

A four-stage pipeline: a union of complementary retrieval channels (character/word TF-IDF, name crowds,
hash keys), a learned pruner, gradient-boosted pair matchers (stage A on 100+ similarity features, stage B
adding context from stage A's own out-of-fold probabilities), and an entity-level decision rule that
maximises expected F0.5 per Source-1 entity using a learned "how many matches" model. Every vocabulary
(legal forms, regions, region aliases, a romanised-token dictionary for India's native-script records)
is mined from the provided data, so the same code runs on France, which never appears in training.

---

## 2. Methodology

### 2.1 Problem Analysis

Audit outputs: `runs/d0_audit.txt`, `runs/france_audit.txt`; trigger table `runs/d0_triggers.md`.

| Finding | Value | Consequence |
|---|---|---|
| Matches per S1 | mean 3.46; singletons 5.58% (US), 5.59% (India) | recall matters as much as precision; 94% of entities are scored on multi-link sets |
| Duplicates inside S2/S3 | 54% (S2) and 59% (S3) of matched S1s have 2+ matches in one source | S2/S3 records of one business resemble each other: "candidate support" and sibling features |
| Cross-country pairs; many-to-one | 0 and 0 of 7,638,365 true pairs | block per country; each record goes to at most one S1 (hard assignment) |
| Pool size | (S2+S3)/S1 = 4.68 (train) vs 5.53-5.82 (test) | dev pool withholds 19% of dev S1s so their records act as extra distractors |
| Native scripts | 23.5% of India S2 names, 13% of India S3 names (Devanagari, Kannada, Telugu, Tamil, Bengali, Gujarati) | romanisation + a token dictionary mined from training pairs |
| Name crowds | 28-43% of test S1 names repeat; France "same core name + same city" collisions are the largest | no channel may retrieve on the name alone; crowd-size features |
| Noise in true pairs | first house number conflicts on 20.9%; last name token equal on only 40.5% | no hard vetoes; legal form is a soft feature |
| France-only noise (unlabelled test audit) | "R." for Rue in 12% of French S2/S3 addresses, "St-" for Saint-, Av/Bd/Imp/Rte | a language-agnostic subsequence-abbreviation matcher instead of a French word list |

### 2.2 The metric has two regimes

F0.5 is computed per S1 and averaged. For the first link of an entity the cost is symmetric (a wrongly
linked singleton and an unlinked one-match entity both lose 1.0), so the break-even probability is about
0.5. Each further link faces the precision weighting: a second link needs p of about 0.73 once the first is
certain. A single global threshold therefore throws away correct first links. The OOF-tuned two-threshold
rule lands exactly there: first link at tau1 = 0.59-0.61, further links at tau2 = 0.67-0.68
(run `0926-032111-decide`). The final rule goes one step further (section 4.4).

### 2.3 Solution Strategy

**Approach type:** blocking (union of channels) + learned pruner + two-stage gradient boosting + calibrated,
entity-level F0.5 decision.

**Core innovation:** (1) the decision layer optimises expected F0.5 per entity with a count model
(how many true matches does this S1 have among its candidates?) instead of a threshold; (2) all
normalisation vocabularies are mined from the data, including a romanised-to-Latin token dictionary
learned from aligned training pairs, which lifts India blocking recall from 0.943 to 0.964.

---

## 3. Candidate Generation (Blocking)

Per country: queries = S1, index = S2 u S3 of the same country.

| Channel | Representation | Notes |
|---|---|---|
| C1 TF-IDF | name character 3-grams + address words; score = 0.5 cos_name + 0.5 cos_addr | exact top-k via sparse products: 100 candidates generated from the rarer features (posting lists <= max(20,000, 0.5% of the index)), rescored exactly on the full vectors, top 30 kept (15x faster than the full product with higher recall on a 3,000-query benchmark) |
| C2 name crowd | exact sorted-token core name | crowds <= 10 records: all; larger crowds: the 10 with the highest address cosine |
| C5a key | house number + rarest street token (document frequency) | blocks > 200 records skipped; top 10 per S1 by C1 score |
| C5b key | first core-name token + city | as above |

Pair completeness (PC, share of true pairs retrieved) on the dev universe:

| Version | US union PC | India union PC | Candidates / S1 | run |
|---|---|---|---|---|
| 1% benchmark, full sparse product | 0.980 | 0.928 | 35.4 | `0926-005620-block` |
| v1 (rare-feature C1) | 0.9741 | 0.9429 | 36.4 | `0926-011721-block` |
| v2 (+ transliteration dictionary, rarest-token key) | [v2] | 0.9640 | [v2] | [v2 run] |

Reduction ratio: 0.999994 (US), 0.999991 (India) (`0926-011721-block`).

**Pruner.** LightGBM on channel scores/ranks and cheap similarities, trained out-of-fold. Per S1 it keeps the
smallest top-N whose PC is within 0.005 of the union (N = 8 in both countries; France uses the larger
value), plus each record's top 3 S1s so that the assignment step sees the competing entities
(`0926-020122-features`: PC after pruning 0.9739 US, 0.9424 India, ~20 candidates per S1).
`candidate_pairs.tsv` is exactly this pruned set; every match is a subset of it (the writer asserts it).

---

## 4. Matching Model

### 4.1 Normalisation (two views)

View A (NFKC, whitespace) is kept for any text model. View B (all features): lower-case, accent folding,
romanisation of native scripts (anyascii, then the mined token dictionary), junk flags (leading "#", "<<",
quotes; trailing "!"; "(India)"/"(France)" tags), "&" = "and", dotted abbreviations joined (L.L.C. = LLC),
long digit runs (phone numbers) removed, DBA split, legal-form tokens moved to their own field wherever
they sit (the noise moves them to the front), and an order-invariant address parser (house number and
bis/ter suffix, street, unit, landmark, city, region via mined regions and region aliases such as
"texas" -> tx, "krnatk" -> karnataka).

### 4.2 Features (D4), never country, ids or row order

- Names (core and full): Jaro-Winkler, Levenshtein ratio, token-set/sort ratios, partial ratio, char-3-gram
  TF-IDF cosine, token containment both ways, consonant-skeleton containment, subsequence-abbreviation token
  matching both ways, acronym match, DBA best-part similarity.
- Legal form: equal / missing on either side / conflicting, containment.
- Address: house number equal / conflicting / missing and its string ratio, bis/ter agreement, all-number
  overlap, street similarity, city equality and similarity, city found anywhere in the record address,
  region equal / conflicting / missing, unit and landmark similarity, component counts, token containment,
  abbreviation matching.
- Context: channel ranks and scores, rank and gap within the S1's candidates and within the record's S1s,
  name crowd sizes, number of strong candidates, candidate support (other candidates of the same S1 that
  share the record's name, address or house number).

### 4.3 Models

- **Stage A:** LightGBM, 5-fold out-of-fold with folds grouped by (country, city) and stratified by
  country x match-count bucket; test = mean of the 5 fold models.
- **Stage B:** LightGBM on the same features plus context from stage A's out-of-fold probabilities (rank and
  gap within the S1, competition for the record, best probability of sibling candidates sharing the name,
  address or house number). Cross-fitted: stage B never sees probabilities produced with its own labels.
- **Calibration:** isotonic regression on the OOF probabilities (US + India pooled); expected calibration
  error 0.0002 after calibration (`0926-032111-decide`).
- **Assignment:** each S2/S3 record is kept only for the S1 with its highest calibrated probability.

### 4.4 Decision rule

Three rules are compared out-of-fold with a paired bootstrap; the simpler one is kept unless the gain's
interval excludes zero (`0926-032111-decide`, stage A):

| Rule | OOF macro F0.5 | Gain vs two thresholds [95% CI] |
|---|---|---|
| Two thresholds (tau1, tau2 per country) | 0.97063 | - |
| GFM, independent candidates | 0.97042 | -0.00021 [-0.00033, -0.00009] |
| GFM + count model (chosen) | 0.97143 | +0.00080 [+0.00067, +0.00093] |

GFM (Dembczynski et al., 2011, generalised to F-beta) returns the subset with the highest expected F0.5,
including the empty set. The count model (4-class LightGBM on the entity's calibrated candidate
probabilities, temperature-scaled, no country feature) decides how many matches an entity has; the pair
model decides which.

### 4.5 France

No French labels exist. Leave-one-country-out (`0926-063314-loco`): a model trained on US only scores
F0.5 0.871 on India (vs 0.959 in-country), trained on India only it scores 0.958 on US (vs 0.979). The
oracle probability offsets are +0.1 and -0.1, so the prior offset b_FR = 0: the shift is not a
probability-level miscalibration, and France uses the pooled count model and thresholds unchanged. The
India drop is concentrated in native-script records that the US model never sees; France has none.

---

## 5. Results & Error Analysis

| Version | OOF macro F0.5 [95% CI] | US | India | singletons | run |
|---|---|---|---|---|---|
| v1: stage A + GFM/count | 0.97143 [0.97114, 0.97170] | 0.97918 | 0.95949 | 0.96817 | `0926-032111-decide` |
| v2: + transliteration, support features, stage B | [v2] | | | | |

Oracles (`0926-032111-decide`): perfect decisions on the retrieved candidates would score 0.98659, so
about 0.013 is lost in blocking and pruning and about 0.015 in the pairwise decisions; predicting every
true singleton empty would add 0.0018.

[error analysis and failure-mode table: from runs/error_analysis.md]

---

## 6. Conclusion

[to fill after the final run]

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/run_pipeline.py` is the single entry point:
`python -m src.run_pipeline --config config_v2.yaml --stage all` regenerates both output files from
`dataset/` (prep, transliteration, dev stages, test stages, submit with the organisers' validator).
Stage-by-stage commands, runtimes and hardware are in the README.

### B. Additional Results

[run-log tables: blocking per channel, pruner PC(N) curve, ablations]
