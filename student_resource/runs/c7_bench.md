# C7 blank-address channel: dev-sample pair completeness

Run by Ashutosh, 26 Sep 13:12-13:16 IST, on a MacBook M2 (16 GB). Code: `blocking-ports` branch,
`src/blocking/union.py` at the commit that adds this file.

Setup: `prep` normalisation on train only (no translit dictionary, `norm_dir: norm`), `build_dev_universe`
as in config_v3, 3% of the dev queries (India 6,049, US 9,141), full S2 u S3 per country as the index.
One blocking pass per country; "without C7" drops the pairs that only C7 proposed.
The baseline US union PC (0.9736) matches the 0.974 in PROGRESS_REPORT.

| country | C7 setting | PC union | PC on blank-address targets | cands/S1 |
|---|---|---|---|---|
| India | off | 0.94217 | 0.59045 | 38.65 |
| India | k_blank 3, min cos 0.5 | 0.94971 | 0.78894 | 41.10 |
| India | k_blank 5, min cos 0.3 | 0.95171 | 0.84171 | 43.13 |
| US | off | 0.97359 | 0.62329 | 34.76 |
| US | k_blank 3, min cos 0.5 | 0.98271 | 0.80324 | 37.21 |
| US | k_blank 5, min cos 0.3 | 0.98419 | 0.83250 | 39.32 |

Blank-address targets are 3.8% (India) and 5.1% (US) of dev true pairs. C7 costs about 1 s per country.

Not measured: PC after the learned pruner, and OOF macro F0.5. Those need a features + stack run with
config_v4.yaml. The India baseline here is below the norm_tl number (0.964) because the translit
dictionary was not applied; the dictionary does not touch empty addresses, so the C7 gain should carry over.
