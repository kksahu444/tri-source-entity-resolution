# France research scripts (27 Sep)

Diagnostics behind `runs/france_research.md`. They read the local iteration-2 artifacts (`artifacts/block`,
`artifacts/feat`, `artifacts/pred`, `artifacts/decision.pkl`) and the upload-1/2 files; paths are absolute to
the laptop checkout. Run from this folder in order: `patterns.py` writes `pat_*.parquet`, which `strata.py`,
`altgroups.py`, `markers.py` (dev part) and `strongkeys.py` read.

| Script | Question |
|---|---|
| `selfest.py` | model's own expected F0.5 per country (GFM) vs realised dev F0.5 |
| `structure.py` | name / address collision structure of the S1 tables |
| `show_fr.py` | print S1s with their candidates (uncertain band or random) |
| `patterns.py` | pair classes (name relation x house relation): prevalence, link rate, dev truth rate |
| `upload_patterns.py` | the same classes on the upload-1/2 links vs dev truth |
| `strata.py` | dev precision per stratum transported to test links (expected false links per S1) |
| `markers.py` | noise-marker natural experiment: link rate with vs without each surface-noise marker |
| `altgroups.py` | alternative-name groups beside the S1 (same-house decoy check) |
| `strongkeys.py` | near-certain key pairs (name + house) through retrieval, pruning and linking |
