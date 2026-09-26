# NEXTSESSION

Handoff for teammates and for any fresh Claude/Codex session after a context reset. Read this
first, then `CLAUDE.md`. Update the "State" and "Open items" sections before you stop working.

Last updated: 26 Sep 2026, 18:10 IST, by Ashutosh (blocking lane).

Git rule: work on your own branch, commit small and often, push the branch. Before anything goes to
`main`, merge `origin/main` into the branch and run both test suites (repo root and
`student_resource/code/business_entity_resolution`). Aditya's running chain reads main.

## Where submissions come from

Aditya's pipeline on `origin/main`, under `student_resource/code/business_entity_resolution/`
(configs `config.yaml`, `config_v2.yaml`, `config_v3.yaml`; chains in `student_resource/runs/iter*_chain.sh`).
His `student_resource/PROGRESS_REPORT.md` is the detailed record of that pipeline.

`src/entity_resolution/` (the `main.py` pipeline) is **not** the submission path. My blocking work in it
(branch `blocking-v2`) is a testbed; anything that wins there gets ported into Aditya's blocking as a
reviewed branch, never pushed into his running chain.

## State

| What | Number | Source |
|---|---|---|
| Upload 1 (iteration 1) | public LB 0.960, OOF 0.97143 (US 0.97918, India 0.95949) | manifest.csv, team chat |
| Aditya blocking v2 union | PC US 0.974, India 0.964 at ~36 cands/S1 | PROGRESS_REPORT |
| After learned pruner (N=8 + reverse 3) | ~15-20 cands/S1, PC almost unchanged | PROGRESS_REPORT |
| Where the loss is | ~0.013 in blocking+pruning, ~0.015 in decisions | PROGRESS_REPORT oracle bounds |
| Cross-encoder (xlm-roberta-base, 1 epoch on 681k out-of-dev pairs, Kaggle `ashuwhy/amlc26-ce`) | held-out 4,956 S1: pair AUC 0.9994; CE alone + one threshold, macro F0.5 India 0.981, US 0.974 (blocking ceiling 0.987 / 0.991) | `ce_report.json`, kernel v1 |
| My blocking-v2, 2% train sample | recall 97.43%, UB macro F 0.991, 46.4 cands/S1, 745 s, 8.9 GB peak | `runs/block_s002_v2.txt` |
| My blocking-v2 by country | India 95.58% @ 50.3/S1, US 98.67% @ 43.8/S1 | same |
| My cut curve (per source cap) | 10: 95.72% @ 20.8; 15: 96.41% @ 28.5; 20: 96.78% @ 35.7; 30: 97.43% @ 46.4 | same |

The 0.960 vs 0.971 gap is not explained yet. France is in test and absent from OOF, so OOF cannot see
it. Do not chase it with public-LB tweaks.

Running now (Aditya's machine): iteration 2 test file expected ~16:00-16:30 26 Sep, then
`iter3_chain.sh` (config_v3: generator-aware features, anchor agreement, K-class count model,
all-dev model for test; C7 on, C6 off as of 771695b).

## Decisions (26 Sep)

1. **candidate_pairs.tsv = the pruned set the final model scores. Keep `submit.py` as it is.**
   The organiser sheet has two answers. Early ones (25 Sep 03:14, 06:35) said "input to the first
   scoring model in a cascade". Later ones (25 Sep 17:59, 20:47, 22:28, 22:41) say "the last candidate
   set fed to your matching model; a pruned later stage is fine", and add that a smaller set per S1
   ranks higher. The later answer wins, and it matches the README ("`candidate_pairs.tsv` is the *last*
   one"). I had flagged the pruned file as non-compliant earlier today; that flag is withdrawn.
   Practical rule: never run the validator on the 25M-ID file while features are building (that was
   the OOM).
2. **Final pick is by OOF, not public LB.** The private leaderboard decides. Choose the final
   submission by dev OOF macro F0.5 per country, and only use public LB to catch a broken file.
3. **Candidate set size is a ranking criterion.** Do not loosen the pruner to buy public-LB points.
   If a change adds candidates, it has to show OOF gain.
4. **India recall is the biggest lever left.** 57% of my India misses are native-script names that
   romanisation cannot recover. The dense channel (multilingual e5, MIT) on Kaggle GPU is the fix, and
   it is blocked only on a Kaggle secret (open item 1). Cut-off: if the dense channel has no dev PC
   number by 27 Sep 10:00 IST, drop it and ship without it.
5. **anyascii stays, disclosed.** It is a Unicode character table (ISC), not geo/postal/business
   data, which is what the rulings ban. Name it in the methodology compliance section.
6. **Leak check 7.0-8.6% by city grouping: accept.** Grouping by state only drops it to 5.6-6.1%,
   the cause is name overlap, and no feature carries identity. Note it in the methodology.
7. **C7 (empty-address records): add the forward pass for the test run.** Aditya built C7 (771695b):
   each empty-address record takes its top-10 S1s. That recall depends on how many S1s compete, and a
   test run has ~3x the S1s of the dev universe. Measured on India (`student_resource/runs/c7_scale.md`):
   reverse PC on blank-address targets 0.90 on dev but 0.77 with a test-sized pool; adding the forward
   pass (each S1 takes its top-5 empty-address records) gives 0.90 at test size. US: reverse 0.89 on dev,
   0.81 at test size, 0.88 with both. Code is on main, off by default: `reverse_empty.forward_k: 5`. Dev OOF will
   not show this gain, because on dev the reverse direction has not lost it yet.
   My earlier forward-only port (`blocking-ports`, config_v4) is superseded and not merged.
   My legal-typo rules are not ported: the translit dictionary already maps `limitet`, `praibhet`,
   `elelpi`, `li`, `pra` to ltd/pvt/llp.

## Lanes (proposed 26 Sep 14:25, confirm in team chat)

One owner per lane, and one owner per file. If you need a change in someone else's files, ask them in
chat or open a branch for them to review; do not edit their files on main. Numbers behind the split are in
`student_resource/RESEARCH_PLAN.md` section 1: of the 0.021 we still lose, 0.0095 is blocking, 0.0068 model
misses, 0.0047 false links.

| Lane | Owner | Owns (files) | Delivers | Deadline |
|---|---|---|---|---|
| Pipeline core and runs | Aditya | `student_resource/code/business_entity_resolution/` except `src/blocking/` channels below and `kaggle/ce_*`; all `config_v*.yaml`; `runs/iter*_chain.sh`; `submissions/manifest.csv` | iteration 3 and the final run, every upload, OOF per run | final run starts 27 Sep 06:00 |
| Retrieval and GPU models | Ashutosh | new channels in `src/blocking/` (C7 forward is on main), dense channel, cross-encoder under `kaggle/ce_*` on my own Kaggle account, `src/analysis_c7_scale.py`, the `main.py` testbed | (a) dense candidates and (b) a cross-encoder score, each as a parquet keyed `s1_id, cand_id` that the stack reads as one extra column | dev file by 27 Sep 06:00; cut-off 10:00, else we ship without it |
| Release and compliance | Krishnkant | `docs_draft/` -> final methodology doc, `LICENSES.md` check, `docs/EXPERIMENTS.md`, tags `sub/d<day>-<n>`, the packaging dry run, README check on a small slice, organiser Q&A sheet | a zip that passes the official validator, methodology with the compliance section (decisions 1, 5, 6) | zip checked by 27 Sep 21:00 |
| Decisions and error analysis | 4th teammate (name?) | `src/ablate.py` outputs, `src/error_analysis.py` outputs, France/LOCO write-up | ablation tables and the loss decomposition on the iteration-3 run, for the methodology doc | 27 Sep 12:00 |

Interfaces, so nobody blocks on anybody:
- Aditya exports the pruned dev and test pairs of the current run (`s1_id, cand_id`, plus the stage-B `p` for
  dev) as parquet; I score them on Kaggle and hand back `s1_id, cand_id, ce_p`. He adds one feature column.
  The cross-encoder only scores the uncertain band (0.05 < p < 0.95 on stage B), which keeps it inside a T4 budget.
- Config values (for example `reverse_empty.forward_k: 5`) are set by Aditya. Others propose them with a number.
- Anyone who runs something records it under their own name in `EXPERIMENTS.md` and the manifest.

## Open items

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | Kaggle GPU for the dense channel and cross-encoder | Ashutosh (own Kaggle account and secret, so it no longer waits on Aditya's notebook) | proposed |
| 2 | Log upload 1 public score (0.960) against run `0926-073916-submit` in `student_resource/submissions/manifest.csv` | Aditya | open |
| 3 | Iteration 2 test file, upload 2, log OOF + LB | Aditya | running |
| 4 | Iteration 3 chain | Aditya | queued behind iteration 2 |
| 5 | C7 forward pass, measured at test-sized pool | Ashutosh | on main, off by default |
| 5b | Set `reverse_empty.forward_k: 5` for the next test run | Aditya (his config) | open |
| 5c | Export iteration-3 dev and test pairs for the cross-encoder (`python -m src.ce_export --split dev`, then `--split test`); test scoring takes ~5 h so it must start tonight | Aditya runs the export, I score | open |
| 5d | Dense channel on Kaggle `ashuwhy/amlc26-dense` (Aditya's kernel, data from private dataset `ashuwhy/amlc26-raw`) | Ashutosh | running |
| 6 | Methodology doc (`docs_draft/Documentation_draft.md` in his tree): compliance section with decisions 1, 5, 6 | Krishnkant (proposed) | open |
| 7 | Final zip via `python -m src.package --team <TEAM>`, README check on a small slice | Krishnkant (proposed) | 27 Sep |
| 7b | Ablation tables (`python -m src.ablate`) | 4th teammate (proposed) | 27 Sep |
| 8 | Zip deleted in 4ee3179 | Ashutosh | restored on blocking-v2 (ae8b38a) |

## My branch (`blocking-v2`)

Committed and pushed (feb5c80..):

- `blocking.py`: hashed TF-IDF (2^22 features, two passes, 500k-row chunks in parallel),
  chunked top-k via `sparse_dot_topn`, blank_name and script channels, number+city key,
  agreement-based score. `char_max_df` 0.02 is the speed knob (kNN 2.6x faster than 0.05).
- `normalize.py`: parallel chunked normalisation, legal typo fixes, affix stripping.
- `requirements.txt`: `pyarrow` (Arrow strings, 419 B vs 1,204 B per record) and `sparse_dot_topn`.
- 30 tests pass: `python -m pytest -q`.

Known cost: features are ~459 B per pair, so ~100M test pairs would be ~46 GB. Test inference in
this pipeline must stream per S1 block. Aditya's featurize already chunks (1M-pair chunks, 6 workers).

## Resume commands

```bash
git fetch origin && git log --oneline origin/main | head        # what Aditya pushed since
cat student_resource/submissions/manifest.csv                    # uploads so far (on origin/main)

# my lane
python -m pytest -q
python -u main.py block                                          # block report on the configured sample
cat runs/block_s002_v2.txt                                        # last full 2% report

# organiser Q&A, check for new answers before trusting any rule above
curl -sL "https://docs.google.com/spreadsheets/d/1wMo3bmj5bAdkDI9FvqMnLRmANRGrXNuS6u69ce_VvR0/export?format=csv" -o /tmp/rulings.csv
```

Dataset-derived files (`runs/*.parquet`, `runs/*.tsv`, `dataset/`) stay out of git. The repo stays
private until results are out.
