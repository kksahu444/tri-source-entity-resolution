# Error budget (work order T4), tag `kaggle-0926-1219`, rule `gfm_indep`

Dev OOF macro F0.5 0.98049; total loss 0.01951. Each cell = contribution to 1 - macro F0.5.

| Cause | India | US | all | share |
|---|---|---|---|---|
| false merge on a singleton | 0.00067 | 0.00076 | 0.00143 | 7.3% |
| blocking miss | 0.00330 | 0.00263 | 0.00593 | 30.4% |
| abstained on a real match | 0.00097 | 0.00128 | 0.00224 | 11.5% |
| extra wrong link | 0.00153 | 0.00157 | 0.00310 | 15.9% |
| missed link inside the candidates | 0.00279 | 0.00402 | 0.00681 | 34.9% |

## Blocking misses by country x script x noise type (count, share of that country's misses)

| Country | Script | Noise type | Misses | Share |
|---|---|---|---|---|
| India | Latin | name replaced (no shared token) | 4,582 | 25.9% |
| India | Latin | empty record address | 3,643 | 20.6% |
| India | Latin | house number conflict | 3,449 | 19.5% |
| India | Latin | city changed | 2,338 | 13.2% |
| India | native | house number conflict | 1,296 | 7.3% |
| India | Latin | other | 903 | 5.1% |
| India | native | city changed | 552 | 3.1% |
| India | Latin | name crowd (>= 3 in city) | 413 | 2.3% |
| India | native | name crowd (>= 3 in city) | 256 | 1.4% |
| India | native | other | 211 | 1.2% |
| India | native | empty record address | 59 | 0.3% |
| US | Latin | empty record address | 7,309 | 48.3% |
| US | Latin | name replaced (no shared token) | 4,035 | 26.7% |
| US | Latin | city changed | 1,950 | 12.9% |
| US | Latin | house number conflict | 1,595 | 10.5% |
| US | Latin | other | 221 | 1.5% |
| US | Latin | name crowd (>= 3 in city) | 11 | 0.1% |
