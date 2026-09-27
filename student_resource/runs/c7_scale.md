# C7 recall vs the number of competing S1 queries

Run by Ashutosh, 26 Sep 13:40-14:10 IST, MacBook M2 16 GB, `python -m src.analysis_c7_scale`
(norm_tl, train-only vocab because the test split is not on this machine; min cos 0.3).

Recall on blank-address true pairs of the same 3% of dev queries, while the S1 pool the channel runs
over grows. Reverse = main's C7 (each empty-address record takes its top-10 S1s). Forward = each S1
takes its top-5 empty-address records (`reverse_empty.forward_k: 5`).

India (dev share 0.30 as in config_v3; 765 blank-target true pairs from 6,004 eval S1):

| S1 pool | S1 | reverse PC | reverse pairs/S1 | forward PC | both PC |
|---|---|---|---|---|---|
| 3% sample | 6,004 | 0.9765 | 153.8 | 0.8379 | 0.9765 |
| dev universe | 198,122 | 0.9007 | 6.1 | 0.8379 | 0.9307 |
| all train S1 (test-sized) | 883,188 | 0.7739 | 1.4 | 0.8379 | 0.8967 |

US (same settings; 1,464 blank-target true pairs from 9,021 eval S1):

| S1 pool | S1 | reverse PC | reverse pairs/S1 | forward PC | both PC |
|---|---|---|---|---|---|
| 3% sample | 9,021 | 0.9850 | 197.2 | 0.8292 | 0.9850 |
| dev universe | 305,473 | 0.8880 | 7.2 | 0.8292 | 0.9105 |
| all train S1 (test-sized) | 1,323,633 | 0.8074 | 1.7 | 0.8292 | 0.8784 |

Reading: a test run has every S1 of a country in the pool, so reverse C7 recall on test sits near the
last row, not the dev row. Dev OOF overstates it. Forward is flat by construction. Using both is
+12 points (India) and +7 points (US) on blank-address targets at test size (~4-5% of true pairs, so
about +0.3 to +0.5 points of union PC) for ~5 extra pairs per S1 before pruning.

Not measured: PC after the pruner, OOF. A dev OOF would not show this gain, because on dev the reverse
direction already has the recall it loses on test.
