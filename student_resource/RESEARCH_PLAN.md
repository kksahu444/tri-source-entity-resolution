# Research memo: what the state of the art says, and what to execute before 27 Sep

Written 26 Sep 2026, 10:15 IST, while iteration 2 runs. The goal is the highest private-leaderboard
macro F0.5 plus a research-grade method write-up. Numbers from our pipeline cite `runs/run_log.csv` run IDs.

---

## 1. Where we lose points today (measured)

Iteration 2, stage A only, run `0926-095450-decide`:

| Quantity | Value |
|---|---|
| OOF macro F0.5 | **0.97745** [0.97720, 0.97771] (was 0.97143 in upload 1) |
| India / US | 0.97337 / 0.98009 |
| Ceiling: perfect decisions on our candidates | 0.9905 |
| Lost in blocking + pruning | about 0.0095 |
| Lost in pairwise decisions | about 0.013 |

Retrieval and matching both still matter, and the matching loss is slightly larger. France is the unknown:
the leave-one-country-out drop was 0.022 (India to US) to 0.089 (US to India) in run `0926-063314-loco`,
before transliteration was added.

---

## 2. What the literature says (2024-2026) and whether it applies here

| Line of work | Key claim | Applies to us? |
|---|---|---|
| **Cross-encoders vs LLMs for in-domain matching**: *Beyond Scale and Generation* (arXiv 2607.24688), 1,215 fine-tuning runs | Fine-tuned cross-encoders beat bi-encoders and generative LLMs in-domain; larger models lean on shortcuts and are not reliably better | **Yes, strongly.** Supports the spec's plan: mDeBERTa-v3-base / XLM-R-base cross-encoder on the uncertain band only. Needs the Kaggle T4s |
| **Group / multi-source matching**: GraLMatch (EDBT 2025), TransClean (arXiv 2506.04006) | Pairwise decisions must be checked for consistency within the group; graph signals (bridges, weak transitive support) catch false positives, and naive transitive closure chains errors | **Yes.** Each S1 has 3.5 true records that are noisy copies of each other. A record that agrees with the S1 but with none of its confident siblings is suspicious, and vice versa. We already added sibling features in stage B; a record-to-record check is the next step |
| **Graph refinement + label propagation** (arXiv 2605.25814) | Propagate confident labels over the candidate graph; spend expensive labelling only on uncertain pairs | **Yes, LLM-free:** use stage-A probabilities as seeds, and the cross-encoder as the "expensive labeller" on the band |
| **Hybrid sparse + dense blocking**: Sparkly (BM25 3-grams), SC-Block (supervised contrastive, ESWC 2024), UniBlocker (2024), LSBlock (2025) | Sparse and dense retrievers miss different pairs; the union is best | **Yes for recall.** Our remaining 0.0095 blocking loss is concentrated in India cross-script pairs, where dense multilingual embeddings are strongest. Needs the Kaggle T4s |
| **Lessons from production pipelines** (arXiv 2607.26298) | Precision and recall need separate fixes (vetoes for precision, diverse retrieval for recall); never trust transitive chains; no single matcher wins everywhere | **Yes.** Confirms: diverse retrieval channels, no transitive closure, model bake-off |
| **LLM matching** (Match/Compare/Select, COLING 2025; AnyMatch 2024; AgenticER 2026) | Selecting among candidates beats independent pairwise decisions; LLMs help zero-shot | **Partly.** The selection idea is what our GFM + count model and stage-B context features do. LLMs over 25M pairs are infeasible on 2x T4, and <= 8B MIT/Apache models are weaker than a fine-tuned cross-encoder in-domain |

---

## 3. Out-of-the-box ideas specific to this data

These come from the data itself rather than the literature.

1. **Second-hop retrieval (query expansion through confident matches).** An S1's true records are noisy copies,
   and copies often share what the S1 lacks: the same native-script name, the same truncated address, the same
   website-style name. A confident match can then retrieve its siblings even when they look unlike the S1.
   This is pseudo-relevance feedback from IR, applied to ER. Cheap version: hash joins on (name key, house)
   and (name key, city) from anchors with p >= 0.8. **Measure first:** how many blocking misses have a
   retrieved sibling sharing such a key.
2. **Generator-aware generic features.** These are generic formulations, not dataset-specific rules:
   - space-free name containment, for website-style names ("anchorsmartcommunities.com" vs "Anchor Smart Communities");
   - house-number substring and numeric closeness (3977 vs 977, 924 vs 922, ranges 3977-3979);
   - "record name has no token overlap but the address matches the S1 and a confident sibling", for fully
     replaced names ("Beloumbra" for "Kip K. Hannigan, Ph.D.").
3. **Record-to-record consistency as a verifier (TransClean-style, without chaining).** For each selected
   record, the max similarity to the other selected records of the same S1. It is low for a false positive
   pulled in by name alone. This feeds the count model and stage B; nothing is ever chained.
4. **France robustness without test labels.** All features are language-agnostic. The one French-specific
   risk found in the audit ("R." for Rue, "St-" for Saint-) is covered by the subsequence-abbreviation
   matcher. The leave-one-country-out prior says no probability offset is needed (b_FR = 0).

---

## 4. Execution plan (ranked by expected gain per hour of compute)

| # | Change | Expected effect | Cost | Needs |
|---|---|---|---|---|
| 1 | Finish iteration 2 (stage B) and upload it | +0.006 OOF already measured for stage A | running, file at ~16:00 | nothing |
| 2 | Generator-aware generic features + record-to-record consistency features | recovers part of the 0.013 decision loss | 1 iteration (~8 h) | nothing |
| 3 | Second-hop retrieval | recovers part of the 0.0095 blocking loss | same iteration as #2 if the measurement is positive | nothing |
| 4 | Cross-encoder on the uncertain band (mDeBERTa-v3-base, fp16, 3 folds) | literature's strongest in-domain matcher; the largest single lever left | ~6-8 GPU hours | Kaggle secret |
| 5 | Dense channel (BGE-M3 / multilingual-e5-small) | recall for cross-script pairs | ~2-3 GPU hours | Kaggle secret |

Schedule: iteration 3 (#2 + #3) runs 16:30 to about 00:30. That leaves 27 Sep for a final run and the
write-up, with the G9 buffer (final upload by 21:00). #4 and #5 can only join if the Kaggle secret is
attached today.

---

## 5. Measured outcomes (26 Sep, 10:30-13:15)

| Experiment | Result | Decision |
|---|---|---|
| Stage B context stacking | OOF **0.97898** [0.97876, 0.97923] vs stage A 0.97745 (`0926-103648-decide`, `0926-095450-decide`) | kept |
| K-class count model (Appendix A update), K = 8 vs K = 4 on stage-B probabilities | gain -0.00001 [-0.00007, +0.00005]; with stage B the best rule is GFM with independence (+0.00009 [+0.00005, +0.00013] vs two thresholds) | bug fix kept; rule chosen per run by the paired bootstrap |
| Second-hop retrieval (C6), measured before building into a run | recovers 4.8-5.8% of India misses and 5.5% of US misses at ~3.5-5.5 extra candidates per S1 (`runs/second_hop.json`) | built, left off |
| Loss decomposition of 1 - 0.97898 | blocking 0.0095 (45%), model misses 0.0068 (33%), false links 0.0047 (22%); **59% of the blocking loss sits in the 5.9% of entities with a missed empty-address record** (`runs/error_analysis.md`) | new channel C7 |

**C7 (reverse name channel).** A record with an empty address has only a name. In C1 its combined score is
capped at 0.5 of a fully matching record. In large name crowds C2 ranks members by address similarity, which
is zero for it. So these records fall out of their S1's lists. In C7 each empty-address record (~3% of the pool)
retrieves its top-10 S1s by name alone, and the model and hard assignment then decide among them.
It runs in iteration 3 together with the generator-aware features, anchor agreement and faster test inference.

---

## Sources

- [Beyond Scale and Generation: Understanding Language Model-based Entity Matching](https://arxiv.org/abs/2607.24688)
- [GraLMatch: Matching Groups of Entities with Graphs and Language Models (EDBT 2025)](https://openproceedings.org/2025/conf/edbt/paper-10.pdf)
- [TransClean: Finding False Positives in Multi-Source Entity Matching via Transitive Consistency](https://arxiv.org/pdf/2506.04006)
- [Adaptive Graph Refinement and Label Propagation with LLMs for Cost-Effective Entity Resolution](https://arxiv.org/pdf/2605.25814)
- [Entity Resolution in Practice: Lessons from a Self-Serve Pipeline](https://arxiv.org/abs/2607.26298)
- [Towards Universal Dense Blocking for Entity Resolution (UniBlocker)](https://arxiv.org/abs/2404.14831)
- [SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines](https://arxiv.org/html/2303.03132v1)
- [Match, Compare, or Select? LLMs for Entity Matching (COLING 2025)](https://aclanthology.org/2025.coling-main.8/)
- [AnyMatch: Efficient Zero-Shot Entity Matching with a Small Language Model](https://arxiv.org/pdf/2409.04073)
- [Heterogeneity in Entity Matching: A Survey and Experimental Analysis](https://arxiv.org/pdf/2508.08076)
