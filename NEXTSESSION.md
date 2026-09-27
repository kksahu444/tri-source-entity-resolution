# NEXTSESSION

Handoff for teammates and for any fresh Claude/Codex session after a context reset. Read this
first, then `CLAUDE.md`. Update the "State" and "Open items" sections before you stop working.

Last updated: 27 Sep 2026, 14:50 IST, by Ashutosh.

Git rule: work on your own branch, commit small and often, push the branch. Before anything goes to
`main`, merge `origin/main` into the branch and run both test suites (repo root and
`student_resource/code/business_entity_resolution`).

**Before starting any piece of work, check every remote branch for it** (`git fetch --all` then
`git log origin/<branch> --since=yesterday -- <paths>`). On 26 Sep I built and validated a cross-encoder
(14:55-19:17) that Aditya had already started as his work order T6 on `origin/adi` at 14:43; his
in-pipeline version replaced mine at 21:21. The lanes below were only ever proposed, never confirmed.

## Read first: which upload counts (organiser Q&A, re-read 27 Sep 00:45)

**Final ranking = private score of our best PUBLIC upload.** Answered three times (25 Sep 18:36, 26 Sep 12:56,
26 Sep 17:03). There is no team pick and the last upload does not count by default. Right now that is
upload 1 (0.960). An upload replaces it only if it beats 0.960 public, and then its private score is what we
get, overfit or not. Decision 2 below is rewritten for this. Candidate-set size is judged separately as
**total pairs** (26 Sep 12:56).

**Q&A sheet re-read 27 Sep 12:30.** New since 00:45: (1) 27 Sep 01:39, "Final ranking (beyond public/private
leaderboard) will be decided based on complete evaluation of the approach, computationally efficient candidate
generation process and code", so the doc, total candidate pairs and code quality count after the score;
(2) 27 Sep 07:51, an OOM question answered with "refer to the update in the Problem Statement section on the
Unstop portal". That update came by mail (updates@unstop.events, 27 Sep 12:41): candidate_pairs.tsv is part
of the final submission, we review it and the code that produces it, and "the approach that generates a smaller
candidate set per Source 1 entity will be ranked higher in the final evaluation beyond the public/private
leaderboard". Nothing new otherwise (5 uploads per day, window closes 27 Sep 23:59). (3) 26 Sep 19:14: the
candidate_pairs file is uploaded in the final code submission, not on the leaderboard.

## Where submissions come from

**`origin/adi`, not `origin/main`.** Aditya's pipeline under `student_resource/code/business_entity_resolution/`
lives on his branch. On 27 Sep 01:30 I merged origin/adi (through c253f72) into main (cac50f2): main's
`student_resource/code` is now his tree plus my additive files. On 27 Sep 14:45 I removed my dead ones from
the submission tree (standalone CE lane `src/ce_pairs.py`, `src/ce_lane_export.py`, `kaggle/ce/`, its test,
and the C4 kernel `kaggle/dense_ds/`); they stay in history on the `cross-encoder` and `dense-channel` branches. Anything he pushes after c253f72 is
on adi only until the next merge; he can `git merge origin/main` without conflicts (re-checked with
`git merge-tree` at 14:50 on 27 Sep). **At 14:50 adi lacks 44 commits of methodology-0927**, among them the latest
Documentation_draft.md and the `src.package` guard against `[FINAL` gaps (78c9a18). A zip built on adi without
that merge ships the older doc and no guard, so merge origin/main (or methodology-0927) into adi before
`python -m src.package`. He runs it on a Kaggle T4 session (`kaggle/runner/run_on_kaggle.py`, XGBoost CUDA backend).
Read his files there: `git show origin/adi:<path>`. His `student_resource/PROGRESS_REPORT.md` (stale at
09:55 26 Sep), `runs/decoy_shift.md` and `runs/kaggle_iter3/` are the current record.

`src/entity_resolution/` (the root `main.py` pipeline) is **not** the submission path. It is my testbed.

## State (27 Sep 00:30)

| What | Number | Source |
|---|---|---|
| Upload 1 (iteration 1) | public 0.960, OOF 0.97143 | manifest.csv on origin/adi |
| France probe (upload 1 with French ids scrambled) | public 0.821, so F_FR ~ 0.928 | same, commit fd6a06b |
| Upload 2 (iteration 3, Kaggle) | public **0.938**, OOF 0.98049 [0.98025, 0.98073]; PC US 0.987, India 0.976 | same, 3334842 |
| Iteration 3 loss (dev) | 0.0195 total: model misses 0.0090, blocking 0.0060, false links 0.0045 | `runs/kaggle_iter3/loss_decomposition.json` |
| Cross-encoder (mine, standalone) | xlm-r-base on 681k out-of-dev pairs: held-out macro F0.5 alone India 0.981, US 0.974, AUC 0.9994 | Kaggle `ashuwhy/amlc26-ce` v1 output |
| Uploads left | 5 on 27 Sep; code freeze 16:00 IST | CLAUDE.md |

**Why upload 2 dropped** (Aditya, `runs/decoy_shift.md`): test has about twice the near-duplicate decoys of
dev (same name and street, house number 1-30 apart), and on test they come as groups of 2+ records. The
support features (`sup_house`, `sup_addr`) that separate them on dev are faked by the decoy's own siblings
on test, so stage B over-links: 3.71 links per US S1 against 3.46 true. Upload 1 had no support features and
did not over-link. Dev OOF cannot see this; `src/acceptance.py` on origin/adi now checks each file before upload
(upload 1 passes, upload 2 fails: US 0.0466 near-house links vs 0.0052 in train truth).

## What Aditya built since 14:00 26 Sep (all on origin/adi, his credit)

- Iteration 4 (aa1ca5b): cross-encoder stage T6 inside the pipeline (band 0.005 < stage-A p < 0.995,
  3 folds by country + core name, mDeBERTa vs XLM-R bake-off, kept only if a paired bootstrap says so),
  C8 address-only retrieval channel, C7 top-25, `config_v4.yaml`.
- `config_v5.yaml` (ab3b00d) = v4 + `reverse_empty.forward_k: 5`, i.e. **my C7 forward pass** (d7378b8 on
  main, measured in `runs/c7_scale.md`) ported into his tree. This is the config for the final run.
- Iteration 5 (ffa311d, 4ba10e0, 6892c12): dev with synthetic decoy entities at the test rate, and
  `src/acceptance.py`.
- c253f72 (23:00): signed house offset and exact-house candidate count features; per-entity dev F0.5 saved
  for paired comparisons across runs.

## My work, and where it stands

| Item | Status |
|---|---|
| Blocking v2 testbed, C7 scale study, C7 forward pass | on main; forward pass is in config_v5 |
| Cross-encoder (`src/ce_pairs.py`, `src/ce_export.py`, `kaggle/ce/train_ce.py`) | **superseded by T6**; removed from main 27 Sep 14:45, kept on branch `cross-encoder`, the export/score/join path is not used. Model is in the `ashuwhy/amlc26-ce` output if anyone wants a second opinion on T6's bake-off |
| Dense channel C4 (e5-small top-30, `blocking.dense`, off by default) | branch `dense-channel` (2 commits, touches Aditya's `features.py`, not merged). Written against main's `union.py`; Aditya rewrote his since, so it does not apply to origin/adi as is. Kaggle `ashuwhy/amlc26-dense` v4 (train split) started running 27 Sep 00:25. `dense_test.parquet` exists only in an older version's output (download from the Kaggle UI; the CLI fetches the latest version). Low value now: the loss that matters is precision, and C4 adds candidates. Cut-off stays 27 Sep 10:00 |
| `blocking-ports`, `c7-forward`, `cross-encoder`, `blocking-v2` branches | stale or merged; nothing to do |
| Local scratch artifacts (adi_art30, CE model copy, dense_test) | lost when /private/tmp was cleared |

## Decisions (26 Sep, still valid)

1. `candidate_pairs.tsv` = the pruned set the final model scores (organiser sheet, later answers win).
2. ~~Final pick by dev OOF~~ Superseded by the organiser ruling above: the best public upload counts.
   So an upload is only worth making if dev OOF (on the decoy dev) and `src.acceptance` both say it beats
   upload 1; a failing file does not go up, whatever its OOF.
3. Candidate set size is a ranking criterion; a change that adds candidates must show OOF gain.
4. anyascii stays, disclosed in the methodology compliance section (Unicode table, ISC licence).
5. Leak check 7.0-8.6% by city grouping: accept, note it in the methodology.

## Open items

Nobody but Aditya and me has committed since 25 Sep 04:12 (Krishnkant's last commit). The release lane has
no owner in practice, and it holds the one deliverable we cannot ship without.

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | Final run: **config_v7** (Kaggle `aditya103856/amlc26-pipeline-v7`, 2x T4, cross-encoder off, started 10:00), config_v6 is the runner default and fallback. The v5 and v6 GPU runs died of memory; the memory fix bd71889 landed at 11:21, **after** v7 started, so v7 may die the same way at stage B test. Since then on adi: config_v7ce (d542f6b, 12:06: v7 + cross-encoder on band [0.02, 0.98], mDeBERTa only) and config_v8 (8e38dfc, 14:03: v7 on a test-like dev pool, 5.8 records per query S1 instead of 20.3; its OOF is not comparable with v7). Manifest on adi still lists only uploads 1-2 at 14:30; his kernels are private, so status is his to report. Uploads 3-5 | Aditya | in progress on origin/adi |
| 2 | **Methodology doc.** 27 Sep 12:20: brought up to iteration 7 (config_v6 word decoys and broad acceptance table, config_v7 France fixes: N° markers, open-set region aliases, zero strip, C9; France research findings; cross-encoder v5 bake-off result and why it is off; upload tooling in appendix A; compliance rows for the new mined vocabularies). Before that: done through iteration 5 on branch `methodology-0927` (cut from origin/adi, touches only `student_resource/docs_draft/` and `docs/`): v2/v3 numbers, C7 both directions + c7_scale table, C8, stage B, T6, upload-2 decoys + acceptance check. Every number cites a run_id. 27 Sep 14:40 on methodology-0927: v7ce and v8 written in (header, 2.1 pool density row, 4.3, new 5.4 test-like dev pool, 5.1 v8 row, 3.4 Unstop update, 7 compliance, appendix A). Final-run numbers are `[FINAL]` gaps (18 now; grep for it) | Ashutosh (written); Aditya fills `[FINAL]` after his last run | done except gaps |
| 3 | Compliance section (section 7 of the draft): anyascii, libpostal, cleanco, hand lists (the draft discloses LEGAL_CANON and the regexes honestly; "all mined" was not quite true), test statistics, synthetic decoys, model licences, candidate_pairs, size = total pairs, best-public ranking, leak check | Ashutosh | done |
| 4 | Bookkeeping: `docs/EXPERIMENTS.md` blocking and OOF tables filled from the run logs (v1-v3, the failed v5/v6 GPU runs, v7 in flight; 27 Sep 12:20); rows for upload 1, the France probe, upload 2 on `methodology-0927`. Tags **proposed, not pushed** (they sit on Aditya's commits; ask first): `git tag sub/d2-1 0beb90e && git tag sub/d2-2 341091a && git tag sub/d2-3 3334842 && git push origin sub/d2-1 sub/d2-2 sub/d2-3` | Ashutosh | rows done, tags waiting |
| 4b | **Zip traps.** (b) configs and (c) LICENSES.md fixed by Aditya in d667c9b (07:45). (a) fixed by me on `methodology-0927` (78c9a18): `src.package` now defaults to `docs_draft/Documentation_draft.md` and refuses a document that still has `[FINAL` gaps (33 right now), so the zip cannot ship a blank or half-filled doc | Aditya (b, c), Ashutosh (a) | done |
| 5 | Final zip: merge origin/main into adi first (see above), then `python -m src.package --team <TEAM>` on origin/adi, official validator, README run on a small slice. If v7ce is the final config, LICENSES.md line 11 ("used only if it wins the bake-off") needs a word: v7ce skips the bake-off and runs mDeBERTa directly | same | 27 Sep |
| 6 | Dense channel C4: **dropped** (27 Sep 01:20). A number that could change the submission needs an OOF gain (decision 3), i.e. a full Kaggle chain with C4 ported into origin/adi's rewritten `union.py`. That run would compete with Aditya's final config_v5 run on the last day's GPU time and uploads. A PC-only number is not possible on this machine either: Aditya's prep needs the test split, which is not here, and the disk has 10 GB free. C4 also adds up to 30 pairs per S1 before pruning, and size is now judged as total pairs. Kaggle `ashuwhy/amlc26-dense` v4 finished 02:08; `dense_train.parquet` (e5-small top-30 per train S1, 66,204,630 rows, 462 MB) is in `runs/dense/` on my machine, git-ignored, unmeasured. The Kaggle CLI hung on it; fetch through the MCP `download_notebook_output` URL with curl | Ashutosh | dropped |

## Resume commands

```bash
git fetch --all --prune
git log --oneline origin/main..origin/adi | head -20          # what Aditya pushed
git show origin/adi:student_resource/submissions/manifest.csv   # uploads and public scores
git show origin/adi:student_resource/runs/decoy_shift.md        # upload-2 diagnosis

.venv/bin/kaggle kernels status ashuwhy/amlc26-dense            # dense train pass
.venv/bin/kaggle kernels output ashuwhy/amlc26-dense -p <dir>   # when COMPLETE

# organiser Q&A, check for new answers before trusting any rule above
curl -sL "https://docs.google.com/spreadsheets/d/1wMo3bmj5bAdkDI9FvqMnLRmANRGrXNuS6u69ce_VvR0/export?format=csv" -o rulings.csv
```

If the Kaggle CLI says "Permission denied" or "Authentication required" while claiming to be logged in,
run `! .venv/bin/kaggle auth login --force`.

Dataset-derived files (`runs/*.parquet`, `runs/*.tsv`, `dataset/`) stay out of git. The repo stays
private until results are out.
