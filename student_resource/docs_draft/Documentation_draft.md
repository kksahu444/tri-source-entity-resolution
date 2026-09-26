# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [team name]
**Team Members:** [members]
**Submission Date:** 27 Sep 2026

Every number below cites the `run_id` of the row in `runs/run_log.csv` that produced it (or the audit
file it comes from). Iteration-3 runs were made on a Kaggle T4 session; their rows are in
`runs/kaggle_iter3/run_log.csv` and are cited as `kaggle-<run_id>`. Out-of-fold (OOF) scores are macro
F0.5 over all dev-universe Source-1 entities, exactly as the leaderboard computes it, with 1,000-resample
bootstrap intervals over entities.

Gaps marked **[FINAL: ...]** are filled from the last run (`config_v5.yaml`) before the zip is built.

---

## 1. Executive Summary

A four-stage pipeline: a union of complementary retrieval channels (character/word TF-IDF, name crowds,
hash keys, a two-way name channel for records with no address, an address-only channel), a learned pruner,
gradient-boosted pair matchers (stage A on 100+ similarity features, stage B adding context from stage A's
own out-of-fold probabilities, optionally a fine-tuned multilingual cross-encoder on the uncertain band),
and an entity-level decision rule that maximises expected F0.5 per Source-1 entity. Normalisation
vocabularies (legal forms, regions, region aliases, a romanised-token dictionary for India's native-script
records) are mined from the provided data, so the same code runs on France, which never appears in
training. Our first two leaderboard uploads taught us that dev OOF can be blind to a test-only failure: the
better-scoring model on dev (0.98049 vs 0.97143) scored lower on the public board (0.938 vs 0.960) because
test holds groups of near-duplicate decoy businesses that dev lacked. The final run trains and selects on
a dev set with synthetic decoys at the test rate, and every file must pass a structural acceptance check
before upload (section 5.3).

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
| Empty record addresses | 48% of US and 21% of India blocking misses after iteration 3 are records with no address (`runs/kaggle_iter3/error_budget_kaggle-0926-1219.md`) | a name channel for empty-address records in both directions (C7, section 3.2) |
| France-only noise (unlabelled test audit) | "R." for Rue in 12% of French S2/S3 addresses, "St-" for Saint-, Av/Bd/Imp/Rte | a language-agnostic subsequence-abbreviation matcher instead of a French word list |
| Near-duplicate decoys (test only) | same name and street, house number 1-30 apart, in groups of 2+ records: 0.84 group records per US S1 on test vs 0.06 on dev (`runs/decoy_shift.md`) | dev with synthetic decoys; acceptance check before upload (section 5.3) |

### 2.2 The metric has two regimes

F0.5 is computed per S1 and averaged. For the first link of an entity the cost is symmetric (a wrongly
linked singleton and an unlinked one-match entity both lose 1.0), so the break-even probability is about
0.5. Each further link faces the precision weighting: a second link needs p of about 0.73 once the first is
certain. A single global threshold therefore throws away correct first links. The OOF-tuned two-threshold
rule lands exactly there: first link at tau1 = 0.59-0.61, further links at tau2 = 0.67-0.68
(run `0926-032111-decide`). The final rule goes one step further (section 4.5).

### 2.3 Solution Strategy

**Approach type:** blocking (union of channels) + learned pruner + two-stage gradient boosting (+ an
optional cross-encoder feature) + calibrated, entity-level F0.5 decision.

**Core innovation:** (1) the decision layer optimises expected F0.5 per entity instead of thresholding
pairs; (2) normalisation vocabularies are mined from the data, including a romanised-to-Latin token
dictionary learned from aligned training pairs, which lifts India blocking recall from 0.943 to 0.964;
(3) every model change is adopted only when a paired bootstrap over entities says the gain is real, on a
dev set rebuilt to contain the decoy structure measured on test.

---

## 3. Candidate Generation (Blocking)

### 3.1 Channels

Per country: queries = S1, index = S2 u S3 of the same country. Countries are read from the data (an open
set), so France gets its own partition without any code for it.

| Channel | Representation | Notes |
|---|---|---|
| C1 TF-IDF | name character 3-grams + address words; score = 0.5 cos_name + 0.5 cos_addr | exact top-k via sparse products: 100 candidates generated from the rarer features (posting lists <= max(20,000, 0.5% of the index)), rescored exactly on the full vectors, top 30 kept (15x faster than the full product with higher recall on a 3,000-query benchmark) |
| C2 name crowd | exact sorted-token core name | crowds <= 10 records: all; larger crowds: the 10 with the highest address cosine |
| C5a key | house number + rarest street token (document frequency) | blocks > 200 records skipped; top 10 per S1 by C1 score |
| C5b key | first core-name token + city | as above |
| C7 empty address, reverse | name TF-IDF of records with no address against all S1 | each empty-address record takes its top-k S1s (k = 10 in iteration 3, 25 from iteration 4), cosine >= 0.3 |
| C7 empty address, forward | same vectors, other direction | each S1 takes its top-5 empty-address records (`reverse_empty.forward_k: 5`, config_v5) |
| C8 address only | address TF-IDF alone | each S1 takes its top-8 records by address cosine >= 0.6 (50 generated); aimed at "name replaced" misses, 26-27% of blocking misses in both countries (`error_budget_kaggle-0926-1219.md`) |

### 3.2 Why C7 needs both directions

The reverse direction lets each empty-address record choose its S1s. Its recall depends on how many S1s
compete for those k slots, and a test partition has every S1 of the country, about 4x the dev universe.
Measured by Ashutosh on train with the S1 pool grown to test size (`runs/c7_scale.md`, commit a4839a3 on
origin/main; recall on blank-address true pairs of the same 3% of dev queries, dev share 0.30, min cosine
0.3, reverse k = 10):

| Country | S1 pool | S1 | reverse PC | forward PC (k = 5) | both PC |
|---|---|---|---|---|---|
| India | 3% sample | 6,004 | 0.9765 | 0.8379 | 0.9765 |
| India | dev universe | 198,122 | 0.9007 | 0.8379 | 0.9307 |
| India | all train S1 (test-sized) | 883,188 | 0.7739 | 0.8379 | 0.8967 |
| US | 3% sample | 9,021 | 0.9850 | 0.8292 | 0.9850 |
| US | dev universe | 305,473 | 0.8880 | 0.8292 | 0.9105 |
| US | all train S1 (test-sized) | 1,323,633 | 0.8074 | 0.8292 | 0.8784 |

Dev OOF overstates reverse-only recall on test; the forward pass is flat by construction because each S1
always gets its own 5 slots. Both directions together add 12 points (India) and 7 points (US) of recall on
blank-address targets at test size, for about 5 extra pairs per S1 before pruning. A dev OOF cannot show
this gain, because on dev the reverse direction still has the recall it loses on test. It enters the final
run for that reason (config_v5), not because of an OOF number.

### 3.3 Pair completeness

Pair completeness (PC, share of true pairs retrieved) on the dev universe:

| Version | US union PC | India union PC | Candidates / S1 | run |
|---|---|---|---|---|
| 1% benchmark, full sparse product | 0.980 | 0.928 | 35.4 | `0926-005620-block` |
| v1 (rare-feature C1) | 0.9741 | 0.9429 | 36.4 | `0926-011721-block` |
| v2 (+ transliteration dictionary, rarest-token key) | 0.9740 | 0.9640 | 36.1 | `0926-074645-block` |
| v3 (+ C7 reverse, k = 10) | 0.9873 | 0.9759 | 42.5 | `kaggle-0926-091122-block` |
| v5 (C7 k = 25 + forward, C8, decoy dev) | [FINAL] | [FINAL] | [FINAL] | [FINAL run_id] |

C7 alone retrieves 4.2% (US) and 3.5% (India) of all true pairs in v3 (`kaggle-0926-091122-block`); its
records are the ones no other channel reaches. Reduction ratio: 0.9999933 (US), 0.9999894 (India) in v3.

### 3.4 Pruner and the candidate set

LightGBM on channel scores/ranks and cheap similarities, trained out-of-fold. Per S1 it keeps the smallest
top-N whose PC is within 0.005 of the union, plus each record's top 3 S1s so that the assignment step sees
the competing entities. N = 8 in v1-v2 (`0926-020122-features`: PC after pruning 0.9739 US, 0.9424 India,
about 20 candidates per S1 on dev) and N = 9 in v3 (`kaggle-0926-093309-features`: 0.9857 US, 0.9742
India, 23.2 per S1 on dev).

`candidate_pairs.tsv` is exactly this pruned set, the input of the stage-A matcher; every match is a subset
of it (the writer asserts it, and the organisers' validator passes). The organisers measure candidate-set
size as total pairs:

| File | Total candidate pairs | Per test S1 | run |
|---|---|---|---|
| Upload 1 | 25,722,619 | 14.85 | `0926-073547-submit` |
| Upload 2 | 27,593,574 | 15.93 | `kaggle-0926-121628-submit` |
| Final | [FINAL] | [FINAL] | [FINAL run_id] |

Upload 2 per country (`kaggle-0926-111311-features`): US 15.85, India 16.08, France 15.64 pairs per S1.

---

## 4. Matching Model

### 4.1 Normalisation (two views)

View A (NFKC, whitespace) is kept for any text model. View B (all features): lower-case, accent folding,
romanisation of native scripts (anyascii, then the mined token dictionary), junk flags (leading "#", "<<",
quotes; trailing "!"; "(India)"/"(France)" tags), "&" = "and", dotted abbreviations joined (L.L.C. = LLC),
long digit runs (phone numbers) removed, DBA split, legal-form tokens moved to their own field wherever
they sit (the noise moves them to the front), and an order-invariant address parser (house number and
bis/ter suffix, street, unit, landmark, city, region via mined regions and region aliases such as
"texas" -> tx, "krnatk" -> karnataka). Legal-form and region vocabularies are mined from Source-1 names and
addresses of both splits (label-free token statistics); region aliases and the transliteration dictionary
are mined from aligned training pairs. The only hand-written lists are small: five legal words from the
problem statement (corporation, private, limited, incorporated, company) and the regular expressions for
DBA markers, landmark and unit words, and bis/ter (section 7).

### 4.2 Features, never country, ids or row order

- Names (core and full): Jaro-Winkler, Levenshtein ratio, token-set/sort ratios, partial ratio, char-3-gram
  TF-IDF cosine, token containment both ways, consonant-skeleton containment, subsequence-abbreviation token
  matching both ways, acronym match, DBA best-part similarity.
- Legal form: equal / missing on either side / conflicting, containment.
- Address: house number equal / conflicting / missing, relative difference and string ratio, bis/ter
  agreement, all-number overlap, street similarity, city equality and similarity, city found anywhere in the
  record address, region equal / conflicting / missing, unit and landmark similarity, component counts,
  token containment, abbreviation matching.
- Context: channel ranks and scores, rank and gap within the S1's candidates and within the record's S1s,
  name crowd sizes, number of strong candidates.
- Candidate support (iteration 2): other candidates of the same S1 that share the record's name, address or
  house number (`sup_house`, `sup_addr`). Section 5.3 shows why these features broke on test.
- Generator-aware and consistency features (iteration 3): agreement of the record with the S1's anchor
  fields and with the S1's other strong candidates (`cons_*`).
- Final run: signed house offset (record minus S1) and the S1's count of exact-house candidates (commit
  c253f72), which describe the decoy shape directly.

### 4.3 Models

- **Stage A:** gradient-boosted trees, 5-fold out-of-fold with folds grouped by (country, city) and
  stratified by country x match-count bucket; test = mean of the 5 fold models. LightGBM on CPU through
  iteration 2; XGBoost with the CUDA `hist` method on a Kaggle T4 from iteration 3 (`src/gbm.py`, same
  depth and leaf caps).
- **Stage B:** the same learner on stage A's features plus context from stage A's out-of-fold
  probabilities: rank and gap within the S1, competition for the record, sum of the other candidates'
  probabilities, best probability of sibling candidates sharing the name, address or house number.
  Cross-fitted, so stage B never sees probabilities produced with its own labels. Test context comes from
  each stage-A fold model separately, never from a full-data refit (`stack_b_test_context: per_fold`).
- **Cross-encoder (T6, iteration 4):** a multilingual transformer fine-tuned on pairs whose stage-A
  probability lies in the uncertain band 0.005-0.995, plus a 2% sample of confident pairs. Folds are 3
  groups by (country, sorted core name of the S1), so the text model cannot memorise a business name across
  folds. An fp16-vs-fp32 check runs first, then a bake-off on fold 0 between mDeBERTa-v3-base and
  XLM-R-base (one T4 each; lower held-out band log-loss wins); the winner trains folds 1 and 2. One epoch,
  learning rate 2e-5, batch 64. Every band pair gets an out-of-fold logit; test logits are the mean of the
  winner's fold-1 and fold-2 models. The logit enters stage B as a feature (missing outside the band).
  Bake-off winner and held-out log-loss: [FINAL]. Any failure writes a marker and the pipeline runs without
  it.
- **Calibration:** isotonic regression on the OOF probabilities (US + India pooled); expected calibration
  error 0.00014-0.00017 after calibration (`kaggle-0926-102147-decide`).
- **Assignment:** each S2/S3 record is kept only for the S1 with its highest calibrated probability.

### 4.4 What is kept

Stage B and the cross-encoder are candidates, not defaults. `decide` starts from stage A and adopts stage
B, then stage B with the cross-encoder logit, only when the paired bootstrap interval of the per-entity
F0.5 gain on the decoy dev excludes zero. Variant kept in the final run: [FINAL: stage A / stage B / stage B
+ CE, with the gain and its interval].

### 4.5 Decision rule

Three rules are compared out-of-fold with a paired bootstrap; the simpler one is kept unless the gain's
interval excludes zero:

| Model | Rule | OOF macro F0.5 | Gain vs two thresholds [95% CI] | run |
|---|---|---|---|---|
| v1 stage A | two thresholds | 0.97063 | - | `0926-032111-decide` |
| v1 stage A | GFM, independent candidates | 0.97042 | -0.00021 [-0.00033, -0.00009] | `0926-032111-decide` |
| v1 stage A | **GFM + count model** | 0.97143 | +0.00080 [+0.00067, +0.00093] | `0926-032111-decide` |
| v2 stage B | **GFM, independent** | 0.97898 | +0.00009 [+0.00005, +0.00013] | `0926-103648-decide` |
| v2 stage B | GFM + count model (K = 4) | 0.97887 | -0.00003 [-0.00012, +0.00008] | `0926-103648-decide` |
| v3 stage B | **GFM, independent** | 0.98049 | +0.00009 [+0.00003, +0.00015] | `kaggle-0926-102147-decide` |
| v3 stage B | GFM + count model (K = 4) | 0.98039 | -0.00001 [-0.00011, +0.00009] | `kaggle-0926-102147-decide` |
| final | [FINAL] | [FINAL] | [FINAL] | [FINAL run_id] |

GFM (Dembczynski et al., 2011, generalised to F-beta) returns the subset with the highest expected F0.5,
including the empty set. The count model (4-class LightGBM on the entity's calibrated candidate
probabilities, temperature-scaled, no country feature) decides how many matches an entity has; the pair
model decides which. It helps a stage-A model and stops helping once stage B is in; our reading is that
stage B's context features (the competing candidates' probabilities) already carry the count information.

### 4.6 France

No French labels exist. Leave-one-country-out (`0926-063314-loco`): a model trained on US only scores
F0.5 0.871 on India (vs 0.959 in-country), trained on India only it scores 0.958 on US (vs 0.979). The
oracle probability offsets are +0.1 and -0.1, so the prior offset b_FR = 0: the shift is not a
probability-level miscalibration, and France uses the pooled rule and thresholds unchanged. The India drop
is concentrated in native-script records that the US model never sees; France has none.

We measured France on the leaderboard once, with a diagnostic upload that is never a final candidate:
upload 1 with every French S1's list replaced by wrong ids (`0926-144305-probe`). Public 0.821 against
0.960 gives F_FR = 0.928 (France is 14.98% of test S1s), with +-0.004 from the French share of the public
subset, +-0.0033 from rounding of the probe score, and +-0.033 because the upload-1 score is known to two
decimals only. The same probe puts the US +
India contribution at 0.821 against 0.823 expected from dev (gap -0.0023), so upload 1 transferred from dev
to test in the two labelled countries.

---

## 5. Results & Error Analysis

### 5.1 Scores

| Version | OOF macro F0.5 [95% CI] | US | India | singletons | public | run |
|---|---|---|---|---|---|---|
| v1: stage A + GFM/count | 0.97143 [0.97114, 0.97170] | 0.97918 | 0.95949 | 0.96817 | 0.960 (upload 1) | `0926-032111-decide` |
| v2: + transliteration, support features, stage A only | 0.97745 [0.97720, 0.97771] | 0.98009 | 0.97337 | 0.97253 | - | `0926-095450-decide` |
| v2: stage B | 0.97898 [0.97876, 0.97923] | 0.98167 | 0.97484 | 0.97274 | - | `0926-103648-decide` |
| v3: + C7, generator/consistency features, stage B | 0.98049 [0.98025, 0.98073] | 0.98308 | 0.97649 | 0.97424 | 0.938 (upload 2) | `kaggle-0926-102147-decide` |
| final: config_v5 on decoy dev | [FINAL] | [FINAL] | [FINAL] | [FINAL] | [FINAL] | [FINAL run_id] |

Oracles for v3 (`kaggle-0926-102147-decide`): perfect decisions on the retrieved candidates would score
0.99402, and predicting every true singleton empty would score 0.98192.

The OOF of the final run is measured on a harder dev (with decoys), so it is not comparable with the v1-v3
rows; the comparison that matters there is the paired one inside `decide`.

### 5.2 Where v3 loses F0.5

`runs/kaggle_iter3/loss_decomposition.json` and `error_budget_kaggle-0926-1219.md`; each cell is a
contribution to 1 - macro F0.5 = 0.01951.

| Cause | India | US | all | share |
|---|---|---|---|---|
| false merge on a singleton | 0.00067 | 0.00076 | 0.00143 | 7.3% |
| extra wrong link | 0.00153 | 0.00157 | 0.00310 | 15.9% |
| abstained on a real match | 0.00097 | 0.00128 | 0.00224 | 11.5% |
| missed link inside the candidates | 0.00279 | 0.00402 | 0.00681 | 34.9% |
| blocking miss | 0.00330 | 0.00263 | 0.00593 | 30.4% |

Blocking misses by noise type: in the US, 48% are records with an empty address and 27% are records whose
name was replaced (no shared token); in India, 26% name replaced, 21% empty address, 20% house-number
conflicts. That table produced C7 top-25 and C8.

Common false positives (wrong merges): sibling businesses at one address whose names differ by one token
("Capital Fellowship IV" vs "EV", "Rocky Council III" vs "II"), one-letter name changes at an identical
address ("TN Produce" vs "FN Produce"), and records with an empty address whose name alone looks right.
On test, add the decoy groups of section 5.3.

Common false negatives (missed matches): records whose address is empty and whose name carries noise
("Endyne-Llc", "Martin (Esports)"), records whose name was replaced by an unrelated token or a web domain,
and India records with a different house number and a changed city at the same time
(`runs/kaggle_iter3/error_analysis.md` lists examples per country).

### 5.3 Upload 2: a dev-blind failure and the acceptance check

Upload 2 (v3, OOF 0.98049) scored 0.938 public, below upload 1 (v1, OOF 0.97143, public 0.960). Aditya
traced it (`runs/decoy_shift.md`):

1. Upload 2 links far more: 3.708 links per US S1 against 3.336 in upload 1 and 3.461 true links per S1 in
   train; 259,053 US pairs appear only in upload 2. The local v2 file already has 3.715, so the change came
   with the support features and stage B (iteration 2), not with C7.
2. 94% of the added US links have a house number different from the S1's: same name, same street,
   e.g. 3419 vs 3423. Stage B raised them (median stage-A p 0.73, stage-B 0.91).
3. Test holds about twice the near-duplicate decoys of dev, and on test they come as groups of 2+ records
   at one offset (an entity whose S1 is not in the test file). With the broad definition (house 1-30
   apart, same city, core token-set ratio >= 85), records in such groups per S1: US 0.837 test vs 0.060 dev,
   India 0.209 vs 0.100, France 0.624.
4. On dev the model separates decoys well (median stage-B p 0.001). On test the same pattern has median
   p 0.65, because the support features see the decoy's own siblings as "other candidates confirm this
   house". The shift is largest on `sup_house` (standardised mean difference +1.90) and `sup_addr` (+0.84).
   A feature that was a reliable sign of a true match on dev became a sign of a decoy group on test.

**Fix, in the final run.** (a) The dev pool gets synthetic decoy entities at the measured excess rate (US
0.78, India 0.11 records per query; groups of 2, of 3 with probability 0.1): the S1's name without its
legal form plus an ending drawn from the country's S1 endings, placed at one of the S1's true addresses with
one house offset per entity (log-uniform 1-21, negative 3%), normalised like every real record. Synthetic
pairs built from the provided records are explicitly allowed (section 7). (b) `decide` starts from stage A
and adopts stage B or the cross-encoder only on a paired-bootstrap gain on this dev. (c) Two features
describe the decoy shape directly (signed house offset, count of exact-house candidates).

**Acceptance check** (`src/acceptance.py`). Nothing is fitted. For each country it counts, per S1,
`near_beside`: links whose house number is 1-30 from the S1's (same sorted core name and street) in S1s that
also have a link at the exact house number, which is the decoy shape. A file fails if any country exceeds
the level of true train links by more than 50% relative (France is compared with the US/India average).

| File | US (train truth 0.0052) | India (0.0030) | France (vs 0.0041) | result |
|---|---|---|---|---|
| Upload 1 | 0.0064 | 0.0026 | 0.0033 | pass |
| Upload 2 | 0.0466 | 0.0041 | 0.0162 | fail |
| Final | [FINAL] | [FINAL] | [FINAL] | [FINAL] |

Had this check existed on 26 Sep, upload 2 would not have gone up. From then on no file that fails it is
uploaded, whatever its OOF.

---

## 6. Conclusion

[FINAL: 2-3 sentences after the last run: final OOF on decoy dev, public score, variant kept.]

What we would keep from this: the leaderboard caught a failure that 1,000 bootstrap resamples of dev could
not, because dev and test differed in structure, not in noise level. Measuring the structure of the
submission file against train truth (links per S1, the decoy shape) is cheap and needs no labels, and we
should have done it before the first upload that changed the feature set.

---

## 7. Compliance

Organiser answers are taken from the challenge Q&A sheet (re-read 27 Sep 00:45 IST; the latest answer on a
topic wins).

| Rule | What we do | Organiser answer |
|---|---|---|
| No external data, APIs or lookups | No network call anywhere in the pipeline. The only download is the pretrained cross-encoder weights from Hugging Face, once, before the run | 25 Sep: "Prohibited: external databases, APIs, geocoding/entity lookup, internet-sourced augmentation, and packages bundling external geo/postal/business data" |
| libpostal | Not used (`ALLOW_LIBPOSTAL: false`); our own address parser (section 4.1) | named as prohibited, 25 Sep 01:34 and 02:33 |
| cleanco | Not used | prohibited (bundles a legal-term database), 25 Sep 23:14 |
| Transliteration | anyascii 0.3.3 (ISC licence), a Unicode character table with no geo, postal or business data | anyascii named as allowed, 25 Sep 11:44 and 18:36 |
| Hand-written lists | Five legal words from the problem statement, regular expressions for DBA, landmark and unit words, bis/ter. No city, postal or country tables. Every other vocabulary is mined from the provided records | "small hand-written normalization dictionaries" allowed (25 Sep); state abbreviations fine, city-level or postal tables not (25 Sep 14:43); hand-typed abbreviation and legal-form lists allowed (26 Sep 12:56, 17:03) |
| Label-free statistics on test | TF-IDF vocabularies, document frequencies, legal-form and region vocabularies mined from Source-1 text of both splits | allowed, 25 Sep 02:03; 26 Sep 12:56, 18:43 |
| Synthetic pairs | Synthetic decoy entities in the dev pool, built from provided records only (section 5.3) | "Self-training and generating synthetic pairs from the provided records are also fine", 25 Sep 02:03 |
| Pretrained models | Only in the cross-encoder, fine-tuned on the provided training pairs only | MIT/Apache-2.0, <= 8B parameters per model, offline, fine-tuned only on provided data (25 Sep 00:04, 10:26) |
| Country as an open set | Countries are read from the data; no feature encodes country; France runs through the same code | 25 Sep 01:42, 02:42 |
| candidate_pairs.tsv | The pruned set fed to stage A, the first pair matcher; every matched id is in it | 25 Sep 03:14: "in a cascade, the input to the first scoring model"; 25 Sep 17:59 and 20:47: "the last candidate set fed to your matching model ... a pruned later stage is fine". We follow the later answers. The pruner scores candidates but is part of candidate generation |
| Candidate-set size | Reported as total pairs (section 3.4) | "Total pairs. This is a separate judging criteria", 26 Sep 12:56 |
| Which submission counts | The private score of our best public submission | 25 Sep 18:36, 26 Sep 12:56 and 17:03 |

### 7.1 Models and licences

| Model | Licence | Parameters | Use |
|---|---|---|---|
| LightGBM / XGBoost tree ensembles | MIT / Apache-2.0 (libraries) | trained here from scratch | pruner, stage A, stage B, count model |
| microsoft/mdeberta-v3-base | MIT | 86M backbone + 190M embeddings | cross-encoder bake-off candidate |
| FacebookAI/xlm-roberta-base | MIT | 270M | cross-encoder bake-off candidate |
| intfloat/multilingual-e5-small | MIT | 118M | optional dense channel C4, measured separately, not in the final run [FINAL: confirm] |

No model outside MIT / Apache-2.0 and none above 8B parameters is used, in training or at inference.

### 7.2 Leak check on the grouped folds

Folds are grouped by (country, city). The share of validation candidate pairs whose record's true S1 sits in
another fold is 7.0% (India) and 8.6% (US) over all union pairs (`runs/leak_check.md`,
`0926-020122-features`), above the 5% we set ourselves. Grouping by state instead still gives 5.6% and 6.1%
(on union rank <= 10), so these are name-driven cross-city candidates, not a city-parsing artefact; grouping
by name chains into one giant group. We kept city grouping because no feature carries an entity or record
identity: every feature is a similarity or rank computed from the two records' text and the blocking
scores, so a model cannot learn "record X belongs to S1 B" in one fold and reuse it in another. The
public probe (section 4.6) puts upload 1's US + India score within 0.0023 of dev.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/run_pipeline.py` is the single entry point:
`python -m src.run_pipeline --config config_v5.yaml --stage all` regenerates both output files from
`dataset/` (prep, transliteration, dev stages, test stages, submit with the organisers' validator). The
final run used `kaggle/runner/run_on_kaggle.py` on a Kaggle GPU session [FINAL: accelerator] (the XGBoost
CUDA backend and the cross-encoder need a GPU; everything else runs on CPU). `python -m src.acceptance --config
config_v5.yaml --file <matching_results.tsv>` is the pre-upload check. Stage-by-stage commands, runtimes
and hardware are in the README. [FINAL: final run wall time and hardware.]

### B. Additional Results

Blocking per channel, dev universe (PC of each channel alone; v1 `0926-011721-block`, v2
`0926-074645-block`, v3 `kaggle-0926-091122-block`):

| Channel | US v1 | India v1 | India v2 | US v3 | India v3 |
|---|---|---|---|---|---|
| C1 TF-IDF | 0.9455 | 0.9089 | 0.9169 | 0.9455 | 0.9169 |
| C2 name crowd | 0.5835 | 0.4590 | 0.6256 | 0.5835 | 0.6256 |
| C5a house + street key | 0.6929 | 0.2618 | 0.2732 | 0.6900 | 0.2732 |
| C5b name + city key | 0.6184 | 0.2762 | 0.3369 | 0.6184 | 0.3369 |
| C7 empty address (reverse) | - | - | - | 0.0419 | 0.0349 |
| union | 0.9741 | 0.9429 | 0.9640 | 0.9873 | 0.9759 |

Pruner PC at top-N, dev universe (`0926-020122-features`, v1):

| N | 5 | 10 | 15 | 20 | 30 | union |
|---|---|---|---|---|---|---|
| US | 0.9213 | 0.9735 | 0.9740 | 0.9740 | 0.9741 | 0.9741 |
| India | 0.8837 | 0.9414 | 0.9425 | 0.9427 | 0.9429 | 0.9429 |

C6 second hop (candidates of a confident candidate) was measured and left off: depending on the anchor rule
it recovers 2-6% of blocking misses for 3.5-5.4 extra candidates per S1 (`runs/second_hop.json`), a poor
trade under the candidate-size criterion.
