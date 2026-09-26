# Upload 2 regression: near-duplicate decoys on test (26 Sep, 22:00 IST)

Upload 2 (iteration 3, dev OOF 0.98049) scored **0.938** public, below upload 1 (iteration 1, dev OOF 0.97143,
public 0.960). Dev could not see the cause. Measured on the submission files, the local iteration-2 artifacts
(same failure: 3.715 links per US S1) and the normalised train/test tables.

## 1. Upload 2 links far more than upload 1

| Country | links/S1 upload 1 | links/S1 upload 2 | pairs only in upload 2 | pairs only in upload 1 |
|---|---|---|---|---|
| US | 3.336 | 3.708 | 259,053 | 12,419 |
| India | 3.175 | 3.397 | 207,833 | 27,853 |
| France | 3.356 | 3.559 | 64,083 | 11,426 |

Train holds 3.461 true links per S1. The local iteration-2 file already has 3.715 (US), so the change came in
with iteration 2 (support features, stage B), not iteration 3.

## 2. The added links are near-duplicates with a different house number

US added links: 94% have a house number different from the S1's (same name, same street, e.g. 3419 vs 3423
Tschache Lane). Stage B pushed them up: stage-A median p 0.73, stage-B median p 0.91, 89% raised by stage B.

## 3. Test has twice the near-duplicate decoys of dev, and they come in groups

Same sorted core name + street, house number 1-30 apart, among the candidates (per S1):

| Country | dev (all) | dev decoys | test |
|---|---|---|---|
| US | 0.1557 | 0.1401 | 0.2878 |
| India | 0.0122 | 0.0058 | 0.0180 |
| France | - | - | 0.1457 |

On dev the model separates them perfectly (decoys: median stage-B p 0.001, 0.4% above 0.8; true
different-house matches are digit-drop/typo noise, 96% above 0.8). On test the same pattern has median p 0.65.
The shifted features (dev decoys vs test pairs of the same pattern, standardised mean difference):
`sup_house` 0.087 -> 1.106 (+1.90), `sup_addr` 0.028 -> 0.394 (+0.84), `q_n_strong` +0.55,
`q_n_name_addr_strong` +0.55. Train decoys are single records; test decoys are entities of 2+ records at the
nearby address (an entity whose S1 is not in the test file), so "other candidates support this house" - a
reliable sign of a true match on dev - is faked by the decoy's own sibling records on test.

## 4. Decoy shape (calibration of the synthetic decoys)

Same sorted core + street, house 1-30 apart; dev counts decoys only (y = 0). Groups = 2+ records at one offset.

| Country | split | near records/S1 | groups/S1 | records in groups/S1 | mean size | offset > 0 | abs offset q10/25/50/75/90 |
|---|---|---|---|---|---|---|---|
| US | dev | 0.1401 | 0.0009 | 0.0018 | 2.00 | 0.999 | 2/3/7/11/14 |
| US | test | 0.2878 | 0.0270 | 0.0551 | 2.04 | 0.972 | 1/3/5/11/16 |
| India | dev | 0.0058 | 0.0002 | 0.0006 | 2.46 | 0.901 | 1/3/5/11/16 |
| India | test | 0.0180 | 0.0018 | 0.0040 | 2.22 | 0.768 | 1/2/5/11/20 |
| France | test | 0.1457 | 0.0077 | 0.0166 | 2.16 | 0.925 | 2/3/7/11/21 |

Examples (test US): S1 "Misaon Pelican | 6236 Highway 114" with its records at 6236, and "Group Misaon Pelican",
"Corp Misaon Pelican", "Partners Misaon Pelican" at 6237 (stage B 0.99); S1 "Green Rmg Inc. | 594 Mccloud Road"
with "Green Rmg Ltd", "Group Green Rmg" at 598. Groups sit next to the S1's own records in 53% of US cases
(dev: 26%), 18x the dev rate, which rules out "the S1's house number is perturbed" as the main mechanism.
French street-type abbreviations (r./av/bd/imp) are not a problem: same name + house + street rest scores
p = 1.000 at the median whether the record writes "r" or "rue".

## 5. Broad definition (the rate actually used)

Sections 3-4 required an identical sorted name and street, which misses decoys carrying name/street noise
("Group Misaon Pelican", "Hgihway"). Broad: house 1-30 apart, same city, core token-set ratio >= 85.

| Country | split | near/S1 | in groups/S1 | offset > 0 | p > 0.5 per S1 | dev: true/S1 | dev: decoys in groups/S1 |
|---|---|---|---|---|---|---|---|
| US | dev | 0.664 | 0.060 | 0.964 | 0.040 | 0.039 | 0.037 |
| US | test | 1.235 | 0.837 | 0.981 | 0.497 | - | - |
| India | dev | 0.341 | 0.100 | 0.691 | 0.078 | 0.078 | 0.056 |
| India | test | 0.473 | 0.209 | 0.791 | 0.113 | - | - |
| France | test | 1.215 | 0.624 | 0.903 | 0.251 | - | - |

Excess group records per S1 on test: US 0.78, India 0.11 (France ~0.6, no dev). Upload 1 (no support features)
did not link them and its US/India public score matched dev, so they are decoys, not true matches.

The acceptance check (`python -m src.acceptance`) on the two uploads, near_beside = near links of S1s that
also have an exact-house link, per S1 (train truth level in brackets): upload 1 US 0.0064 (0.0052), India
0.0026 (0.0030), France 0.0033 (0.0041): pass. Upload 2 US 0.0466, India 0.0041, France 0.0162: fail.

## 6. Fix (iteration 5, config_v5)

- Test-like dev: `dev.decoys` adds unowned decoy entities to the dev pool at the excess rate (US 0.78, India
  0.11 records per query; 2 records, 3 with probability 0.1). Each record: the S1's name without its legal
  form plus an ending drawn from the country's S1 endings (LLC, Inc, Group, Corp, PC, Center, Partners, ...;
  front 15%, upper case 25%, one vowel -> U+FFFD 5%), address = one of the S1's true records' addresses with
  one house offset for the entity (log-uniform 1..21, negative 3%), normalised like every real record.
- `decide` starts from stage A and adopts stage B, then the cross-encoder, only when the paired bootstrap
  interval of the per-entity gain on this dev excludes zero.
- `src.acceptance` must pass before any upload.
