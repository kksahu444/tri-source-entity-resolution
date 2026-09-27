# France: where the points go, measured without labels (27 Sep, 10:00 IST)

France has no training labels. The wrong-ID probe gives France F0.5 about **0.928** on upload 1, against
dev-level transfer for US + India (probe 0.821 vs 0.823 expected). France is 15% of the test S1s, so closing the
gap to the US/India level is worth about +0.006 on the leaderboard. Scripts: `runs/france_research/` (copied from
the session scratchpad); all numbers on the local iteration-2 artifacts and the upload-1 file.

## 1. The model is confidently wrong on France (concept shift, not calibration)

The GFM expected F0.5 under the model's own calibrated probabilities matches the realised dev F0.5 (US 0.9878 vs
0.9871, India 0.9846 vs 0.9843). On test France it is **0.972** on the candidates, while the probe says ~0.928.
The uncertain mass per S1 (sum of min(p, 1-p)) is 0.24 in France vs 0.07-0.09 on dev. So France is not a
probability offset (the LOCO prior b_FR = 0 agrees); the same feature pattern means something else there.

## 2. France is a different world: generic names, crowded streets

| (test S1 tables) | US | India | France |
|---|---|---|---|
| S1s sharing their exact core name with another S1 of the same city | 1.1% | 16.9% | **26.3%** |
| S1s sharing their exact address with another S1 | 8.7% | 57.2% | **20.7%** |
| S1s per street (mean) | 2.7 | (street unparsed) | **40.8** |

Names are "<city> + generic word": 420 S1s are called "lille club" in Lille, 54 "calais loisirs" in Calais.
Only the address (house + street) separates them. Checked by hand: in the big crowds, pairs with exact name +
house + city but a different street (26,291 pairs, model p ~0.03) are coincidences on other streets, so the
model is right to reject them; the name is worthless there and the address has to carry everything.

## 3. Measured leaks (label-free natural experiments)

Surface noise is applied independently of which entity a record belongs to, so a noise marker should not change a
record's link rate, and a France statistic should look like its US/dev counterpart when the generator is the same.

| Leak | Evidence | Rough size (France F0.5) | v7 |
|---|---|---|---|
| **N° prefixes** ("no 5 r etienne marcel", "ndeg 94 r ...") in 11.5% of French S2/S3 addresses: the part is not read as the house part, the street is lost (12.2% of records have a house but no street; US 1.0%) or the part becomes the city | link gap -1.3 pt on those records | +0.002 | numbering markers mined per country (France: no, ndeg; US/India: none) |
| **Departements instead of the region** ("nord", "gironde", "loire atlantique", "pas de calais"); aliases were mined from labelled pairs, so France had none, and the departement became the city when the city came first | city = a real S1 city in 89.4% of French records; 96.4% after both fixes; region found 34% -> 67% | with the above | label-free alias miner from the S1 city -> region map (finds exactly those 4, nothing for the US) |
| **Zero-padded house numbers** ("0011 rue nerton") invisible to the word-level address index | 3.3% of French records, 5.0% US, 5.4% India | small, all countries | leading zeros stripped in the address text |
| **Records in no candidate list**: generic names on dense streets push a record with a misspelt street out of every S1's top-k | 4.1% of French records vs 2.6% US; orphans are mostly crowd names with a street typo ("rue fonrfede", "r chares marionneau") or an empty address | up to +0.0035 | C9: uncovered records (in a query city) retrieve their top-3 S1s |
| **Recall falls with name-crowd size**; dev truth is flat (3.44-3.47 true links per S1 in every crowd bucket) | France links/S1: 3.49 (unique name) -> 3.07 (crowd 30+); dev drop only 3.35 -> 3.20 | up to +0.009 if all recoverable (part is irreducible) | partly via the parse fixes and C9 |
| **Near-house word-changed decoys** (France 0.64 near-house word-changed candidates per S1 vs 0.027 on dev US) | upload 1 linked ~0.03 per S1 of them, dev truth rate for the pattern 33-61% | ~+0.004 | iteration 5/6 synthetic decoys (already in v6) |

Checked and ruled out: same-house decoy groups (France 0.0066 per S1, 99% true on dev), hidden singletons (S1s
without an exact-house candidate: France 3.9% vs dev US 6.6%), exact S1 twins (0.05%), file-order or ID leakage in
train (rows shuffled, ID correlation 0.0005), probe construction (every French row gets one wrong ID, so French
S1s score 0 in the probe and F_FR = (S_upload - S_probe) / 0.1498 holds).

## 4. What this means for 0.99

France's gap is not one bug: it is the sum of several structural leaks worth +0.002 to +0.009 each, plus an
irreducible part (records whose only evidence is a generic name, in a city with hundreds of same-name S1s). v7
fixes the parse and retrieval leaks; together with the v6 decoy handling a realistic France target is 0.94-0.95,
not 0.99. On the whole leaderboard, the larger remaining levers are India (dev 0.976) and the cross-encoder.

## 5. How to judge v7 on France without labels

`python -m src.france_checks --matching <v7 matching_results.tsv> --candidates <v7 candidate_pairs.tsv>` (compares
with upload 1):
- French records in no candidate list: expect below 4.1% (upload 1), towards the US level 2.6%;
- links per S1 by name-crowd bucket: the drop from unique names to crowds 30+ (3.49 -> 3.07) should shrink;
- near-house links (acceptance check, `src.acceptance`) must stay at the train-truth level;
- links per S1 overall should not exceed ~3.46 (train truth) by much: extra links are more likely decoys.

## 6. Other runs

The v5 GPU kernel with the cross-encoder died on 27 Sep ~03:50 after stage-B base folds (no traceback, likely out of
memory), without a file: the cross-encoder stage alone took 3h10m (mDeBERTa band logloss 0.156, XLM-R 0.165 on
fold 0). v7 (`aditya103856/amlc26-pipeline-v7`, 2x T4, XGBoost GPU, cross-encoder off) started 10:00 IST.
