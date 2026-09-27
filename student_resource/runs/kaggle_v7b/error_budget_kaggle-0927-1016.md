# Error budget (work order T4), tag `kaggle-0927-1016`, rule `two_threshold`

Dev OOF macro F0.5 0.98060; total loss 0.01940. Each cell = contribution to 1 - macro F0.5.

| Cause | India | US | all | share |
|---|---|---|---|---|
| false merge on a singleton | 0.00064 | 0.00059 | 0.00123 | 6.3% |
| blocking miss | 0.00229 | 0.00209 | 0.00438 | 22.6% |
| abstained on a real match | 0.00107 | 0.00132 | 0.00238 | 12.3% |
| extra wrong link | 0.00207 | 0.00197 | 0.00403 | 20.8% |
| missed link inside the candidates | 0.00297 | 0.00440 | 0.00737 | 38.0% |

## Blocking misses by country x script x noise type (count, share of that country's misses)

| Country | Script | Noise type | Misses | Share |
|---|---|---|---|---|
| India | Latin | name replaced (no shared token) | 3,342 | 26.4% |
| India | Latin | house number conflict | 2,687 | 21.2% |
| India | Latin | empty record address | 2,114 | 16.7% |
| India | Latin | city changed | 1,995 | 15.7% |
| India | native | house number conflict | 1,100 | 8.7% |
| India | native | city changed | 489 | 3.9% |
| India | Latin | other | 375 | 3.0% |
| India | Latin | name crowd (>= 3 in city) | 211 | 1.7% |
| India | native | name crowd (>= 3 in city) | 190 | 1.5% |
| India | native | other | 149 | 1.2% |
| India | native | empty record address | 16 | 0.1% |
| US | Latin | empty record address | 5,776 | 47.4% |
| US | Latin | name replaced (no shared token) | 2,921 | 24.0% |
| US | Latin | city changed | 1,908 | 15.7% |
| US | Latin | house number conflict | 1,515 | 12.4% |
| US | Latin | other | 67 | 0.5% |
| US | Latin | name crowd (>= 3 in city) | 4 | 0.0% |
