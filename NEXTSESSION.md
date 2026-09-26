# NEXTSESSION

Handoff for teammates and for any fresh Claude/Codex session after a context reset. Read this
first, then `CLAUDE.md`. Update the "State" and "Open items" sections before you stop working.

Last updated: 26 Sep 2026, 13:20 IST, by Ashutosh (blocking lane).

Git rule: nobody but Aditya pushes to `main`. Everyone else works on their own branch, commits small
and often, and pushes the branch. Changes to Aditya's pipeline go on a branch off `origin/main` for
him to review and merge.

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
| My blocking-v2, 2% train sample | recall 97.43%, UB macro F 0.991, 46.4 cands/S1, 745 s, 8.9 GB peak | `runs/block_s002_v2.txt` |
| My blocking-v2 by country | India 95.58% @ 50.3/S1, US 98.67% @ 43.8/S1 | same |
| My cut curve (per source cap) | 10: 95.72% @ 20.8; 15: 96.41% @ 28.5; 20: 96.78% @ 35.7; 30: 97.43% @ 46.4 | same |

The 0.960 vs 0.971 gap is not explained yet. France is in test and absent from OOF, so OOF cannot see
it. Do not chase it with public-LB tweaks.

Running now (Aditya's machine): iteration 2 test file expected ~16:00-16:30 26 Sep, then
`iter3_chain.sh` (config_v3: generator-aware features, anchor agreement, K-class count model,
all-dev model for test; `REBLOCK=1` adds the C6 second-hop channel).

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
7. **Ports from my blocking lane: only C7 goes forward.** Branch `blocking-ports` (off origin/main,
   2 commits) adds C7, name TF-IDF against blank-address records, off unless `blocking.k_blank > 0`,
   and `config_v4.yaml` = v3 + C7. Measured on a 3% dev sample with Aditya's own blocking
   (`student_resource/runs/c7_bench.md` on that branch):
   - US union PC 0.9736 -> 0.9842, India 0.9422 -> 0.9517, about +4.5 union cands/S1;
   - blank-address targets are 3.8-5.1% of true pairs, and their PC goes from ~0.6 to ~0.84.
   Not measured yet: PC after the pruner and OOF F0.5 (needs a features + stack run on config_v4).
   Dropped after checking: my legal-typo rules. Aditya's translit dictionary already maps `limitet`,
   `praibhet`, `elelpi`, `li`, `pra` to ltd/pvt/llp. House number + city key and affix stripping:
   too small to be worth a change this late.
8. **C6 is not on in iteration 3.** `config_v3.yaml` has no `blocking.second_hop` entry, so
   `REBLOCK=1 iter3_chain.sh` blocks without C6 although the chain comment says it is enabled.

## Open items

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | Add Kaggle secret `GITHUB_TOKEN` (read on release `dataset-v1`) to notebook `aditya103856/amlc26-dense` and attach it | Aditya (owns the Kaggle notebook) | open; 2-minute manual step, unblocks dense channel |
| 2 | Log upload 1 public score (0.960) against run `0926-073916-submit` in `student_resource/submissions/manifest.csv` | Aditya | open |
| 3 | Iteration 2 test file, upload 2, log OOF + LB | Aditya | running |
| 4 | Iteration 3 chain | Aditya | queued behind iteration 2 |
| 5 | C7 port on `blocking-ports`, dev union PC measured | Ashutosh | done; needs Aditya's review, then a features + stack run on config_v4 for OOF |
| 5b | Decide on C6 for iteration 3 (config_v3 lacks `second_hop`) | Aditya | open |
| 6 | Methodology doc (`docs_draft/Documentation_draft.md` in his tree): compliance section with decisions 1, 5, 6 | unassigned | open |
| 7 | Final zip via `python -m src.package --team <TEAM>`, README check on a small slice, ablation tables (`python -m src.ablate`) | unassigned | 27 Sep |
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
