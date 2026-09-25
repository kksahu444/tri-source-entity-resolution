# D0 trigger table (spec section 4), filled from runs/d0_audit.txt and runs/france_audit.txt

Audit script: `code/business_entity_resolution/src/audit_d0.py` is a reconstruction (Appendix B did
not reach the agent; the spec message was truncated). Heuristics as the spec states them.

| Block | Measured | Trigger? | Action taken |
|---|---|---|---|
| 1 Labels | train {US, India}; test {US, India, France}; no variants | no | none |
| 2 Pool ratio | (S2+S3)/S1 train 4.68 (US 4.674, India 4.680) vs test 5.76 (US), 5.82 (India), 5.53 (France): +23% / +24% | **yes** (>20%) | Dev pool resampled: 19% of dev S1 entities are dropped from the queries while their S2/S3 records stay in the pool as distractors (`dev.orphan_share`), which lifts records per queried S1 from 4.68 to ~5.8 |
| 3 Integrity | 0 S1 missing from GT; 0 of 7,638,365 matched ids missing | no | none |
| 4 Singletons | US 5.58%, India 5.59% | no (gap 0.01 pt) | distribution matching stays a valid France estimator (only if ALLOW_TEST_PRED_FIT) |
| 5 Per source | share of matched S1 with >= 2 matches in one source: S2 54.2%, S3 58.8% | no (not "at most 1") | count model predicts the total count directly; no S2/S3 convolution |
| 6 Cross-country | 0 of 7,638,365 true pairs | no | COUNTRY_BLOCKING: true |
| 7 Many-to-one | 0 records matched to 2+ S1 | no | HARD_ASSIGNMENT: true |
| 8 ID leakage | median id gap true 292.9M vs shuffled 292.8M | no | ids are never features (C-8) |
| 9 Fields/scripts (added) | India S2 names non-Latin 23.5%, S3 13.2%; addresses ~23% (native state names); empty S2/S3 addresses 2.3-3.7%; France 0% non-Latin | new finding | anyascii romanisation behind flag TRANSLITERATE (open question for the human, see README) |
| 10 Crowds | exact-name crowd share train India 43.9% / US 34.6% vs test India 43.3% / US 28.0% / France 34.1% | no (train not smaller) | none; France crowds are the largest (core name + city collisions) |
| 11 Number conflicts | first-digit-run conflict on true pairs r = 20.89% (both present in 87.2%) | informs D4 | no house-number veto; conflicts are features only |
| 11 Legal flips | last name token equal on true pairs 40.5% | yes | legal agreement is a soft feature only |

## French noise audit: patterns absent from US/India training noise

| Pattern | French S1 | French S2/S3 | US/India S2/S3 | Handling |
|---|---|---|---|---|
| "R." for Rue | 0% | 12.1-12.7% | ~0.1% | layer (c) subsequence-abbreviation token match; char n-grams |
| "St-" for Saint- | 0.01% | 0.83-0.85% | ~0% | layer (c) ("st" is a subsequence of "saint") |
| Av / Bd / Imp / Rte / Ch abbreviations | 0-0.85% | 0.4-3.9% | ~0% | layer (c) |
| bis/ter | 4.97% | 3.8% (dropped ~24%) | 0% | house_sfx agreement feature, missing != conflict |
| Legal form SNC appears | 0% | 1.0% | n/a | legal is a soft feature (flips already common in train) |
| Components dropped (mean parts) | 3.04 | 2.61-2.63 | US 3.16 -> 2.98-3.10; India 5.67 -> 4.72-4.87 | missing-component counts; order-invariant matching |

Patterns shared with training noise (same mechanism, similar rates): accent injection (names 15.7% -> 24%),
all-caps, legal forms moved to the front (sarl 2.0% leading vs llc 1.3%), "(France)" tag kept in 6.4%,
leading junk, website-style names (3.5%).
