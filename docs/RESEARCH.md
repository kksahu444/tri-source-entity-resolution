# Research Notes: Multi-Source Business Entity Resolution

Why the pipeline is built the way it is, and what the literature and practice suggest trying next.

## 1. Problem framing

This is **record linkage** between one clean reference table (S1) and two dirty tables (S2, S3), with
a **one-to-many** cardinality: S1 → {S2, S3}, where each S2/S3 record has at most one S1 owner,
because S1 is deduplicated. The standard decomposition (Christen, *Data Matching*, 2012;
Papadakis et al., *ACM CSUR* 2020) is:

1. **Blocking / candidate generation.** Reduce |S1|×|S2∪S3| comparisons to a small candidate set with
   high recall. Measured by *pair completeness* (recall) and *reduction ratio*. The organisers audit
   exactly these through `candidate_pairs.tsv`.
2. **Pairwise matching.** A classifier scores each candidate pair.
3. **Clustering / decision.** Turn pair scores into consistent groups. Here that means a list per S1,
   subject to the metric.

## 2. Blocking: evidence and choices

| technique | reference / practice | used here |
|---|---|---|
| Token / standard blocking on keys | Fellegi & Sunter (1969); Christen (2012) | exact keys: postcode+name skeleton, acronym+postcode, full skeleton, exact address |
| Sorted-neighbourhood and q-gram blocking | Hernández & Stolfo (1995); Christen (2012) | char 2–4-gram TF-IDF kNN (a similarity-join version of q-gram blocking) |
| Meta-blocking / union of complementary blockers | Papadakis et al. (2020) | union of 4 kNN views + keys. Each view catches a different noise type. |
| Dense-embedding kNN blocking | DeepBlocker, Thirumuruganathan et al., *PVLDB* 2021 | optional `use_embeddings` (multilingual-e5-small, MIT) |
| Phonetic keys for transliteration | Soundex/Metaphone family; Indian-name transliteration is heavy on vowel and aspirate variation | custom consonant skeleton (`agarwal/aggarwal`, `lakshmi/laxmi`) |

**Key point from the organisers' video:** blocking favours recall and pulls in *look-alikes*: a similar
name at a different address, or a different business at the same address. The matcher's job is to
reject them, which is why the address-only blocker exists and why postcode and house-number
*conflict* features matter.

## 3. Matching: feature-based GBDT vs deep matchers

| family | representative work | trade-off |
|---|---|---|
| Similarity features + classifier | Magellan (Konda et al., *PVLDB* 2016), Dedupe, Splink | Fast, strong on structured short strings, easy to debug. The standard baseline. |
| Deep attribute matchers | DeepMatcher (Mudgal et al., *SIGMOD* 2018) | Helps on dirty or textual data. Needs many labels. |
| Fine-tuned pre-trained LMs (cross-encoders) | **Ditto** (Li et al., *PVLDB* 2021) | State of the art on dirty benchmarks. Serialise the pair as `[COL] name [VAL] … [SEP] …`. Needs a GPU. |
| LLM matchers | Peeters, Steiner & Bizer, "Entity Matching using Large Language Models" (EDBT 2025) | Strong zero/few-shot. Costly. The challenge limits us to ≤ 8B, MIT/Apache models. |

**Choice:** start with a GBDT over rich similarity features. It is quick to iterate within 72 h, and
names plus addresses are short strings where engineered similarities are strong. A cross-encoder
score (Ditto-style) becomes one more GBDT feature later if compute allows (PLAN backlog #6).

Features that matter most in business ER, from practice:
- **IDF-weighted token agreement.** Sharing the rare token "Zenith" is strong evidence. Sharing
  "Traders" is weak. A rare *unmatched* token ("Bakery" vs "Robotics") is strong counter-evidence.
- **Legal-form-free names.** "Pvt Ltd" vs "Private Limited" vs nothing should not move the score.
- **Numeric agreement.** Postcode or house-number *conflict* is an almost certain non-match for
  look-alikes. *Missing* is neutral, and the model has to see the difference.
- **Group-relative features.** Because S1 is deduplicated, what matters is whether this S1 is the
  *best* S1 for the candidate, not only whether the pair is similar. Rank and gap features within
  both groups encode this, and the stage-2 model uses the same idea on probabilities.

## 4. Decision rule under macro F0.5

- Macro F-beta with singletons included is an **instance-wise** F-measure. Plug-in thresholding and
  expected-utility maximisation are the two standard approaches (Ye, Chai, Lee & Chieu, "Optimizing
  F-measures: A Tale of Two Approaches", *ICML* 2012; Jansche, 2007). We implement both:
  - `threshold`: global cut on calibrated probabilities, tuned on OOF.
  - `expected_f`: per entity, choose the top-k (k ≥ 0) that maximises an approximation of expected F0.5.
    k = 0 is chosen when P(singleton) = Π(1−pᵢ) beats every k ≥ 1.
- **Exclusivity** is a matching constraint from S1 being deduplicated. It is the one-to-many version
  of the "unique mapping clustering" used for clean-vs-dirty ER (for example in the JedAI toolkit).
  A greedy best-owner-per-candidate step is exact for this cardinality.

## 5. Distribution shift: France only in test

- Features must not depend on the country identity. We only use `country_eq`.
- TF-IDF / IDF statistics are fitted per split, unsupervised, on that split's own records, so French
  n-grams and token rarities come from the test data itself. This is not external data.
- French conventions handled in normalisation: accents, SARL/SAS/SA/EURL/SCI, rue/av/bd/pl/chemin,
  "St" (saint) vs "St" (street), which is safe because both collapse to `st`.
- **Leave-one-country-out** (`python main.py loco`) is our proxy for "unseen country" performance.
  On synthetic data, holding out India dropped macro F0.5 from 0.999 to 0.933, so the shift is real.

## 6. Compliance notes

- No geocoding, registry lookups, commercial ER APIs or web data. Everything is derived from the
  provided TSVs plus hand-written normalisation dictionaries.
- Allowed models: MIT / Apache-2.0, ≤ 8B parameters. Examples: LightGBM (MIT),
  `intfloat/multilingual-e5-small` (MIT), `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (Apache-2.0), `BAAI/bge-m3` (MIT), `Qwen2.5-7B-Instruct` (Apache-2.0). Llama and Gemma licences are
  **not** MIT/Apache, so avoid them.

## 7. Sources consulted for this challenge

- Organisers' problem statement, guidelines PDF, and the explainer video (6 slides: problem, blocking
  example with look-alikes, dataset, deliverables, macro F0.5 tips, "no external lookups").
- AWS Builder Center prep guide (AWS Builder ID, $200 Free Tier credits, SageMaker notebook demo).
  It is useful for compute (ml.t3.medium notebooks, 50 h training) but has no modelling guidance.

## References

- Christen, P. *Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection.* Springer, 2012.
- Fellegi, I. & Sunter, A. "A Theory for Record Linkage." *JASA*, 1969.
- Hernández, M. & Stolfo, S. "The Merge/Purge Problem for Large Databases." *SIGMOD*, 1995.
- Papadakis, G., Skoutas, D., Thanos, E. & Palpanas, T. "Blocking and Filtering Techniques for Entity Resolution: A Survey." *ACM Computing Surveys*, 2020.
- Konda, P. et al. "Magellan: Toward Building Entity Matching Management Systems." *PVLDB*, 2016.
- Mudgal, S. et al. "Deep Learning for Entity Matching: A Design Space Exploration." *SIGMOD*, 2018.
- Li, Y., Li, J., Suhara, Y., Doan, A. & Tan, W.-C. "Deep Entity Matching with Pre-Trained Language Models." *PVLDB*, 2021.
- Thirumuruganathan, S. et al. "Deep Learning for Blocking in Entity Matching: A Design Space Exploration." *PVLDB*, 2021.
- Peeters, R., Steiner, A. & Bizer, C. "Entity Matching using Large Language Models." *EDBT*, 2025.
- Jansche, M. "A Maximum Expected Utility Framework for Binary Sequence Labeling." *ACL*, 2007.
- Ye, N., Chai, K., Lee, W. S. & Chieu, H. L. "Optimizing F-measures: A Tale of Two Approaches." *ICML*, 2012.
