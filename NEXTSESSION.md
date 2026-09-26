# NEXTSESSION

Handoff for teammates and for any fresh Claude/Codex session after a context reset. Read this
first, then `CLAUDE.md`. Update the "State" and "Open items" sections before you stop working.

Last updated: 27 Sep 2026, 00:30 IST, by Ashutosh.

Git rule: work on your own branch, commit small and often, push the branch. Before anything goes to
`main`, merge `origin/main` into the branch and run both test suites (repo root and
`student_resource/code/business_entity_resolution`).

**Before starting any piece of work, check every remote branch for it** (`git fetch --all` then
`git log origin/<branch> --since=yesterday -- <paths>`). On 26 Sep I built and validated a cross-encoder
(14:55-19:17) that Aditya had already started as his work order T6 on `origin/adi` at 14:43; his
in-pipeline version replaced mine at 21:21. The lanes below were only ever proposed, never confirmed.

## Where submissions come from

**`origin/adi`, not `origin/main`.** Aditya's pipeline under `student_resource/code/business_entity_resolution/`
lives on his branch, which is 17 commits ahead of main and 18 behind (he has not merged main since
26 Sep ~14:00). He runs it on a Kaggle T4 session (`kaggle/runner/run_on_kaggle.py`, XGBoost CUDA backend).
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
| Cross-encoder (`src/ce_pairs.py`, `src/ce_export.py`, `kaggle/ce/train_ce.py`) | on main; **superseded by T6**, the export/score/join path is not used. Model is in the `ashuwhy/amlc26-ce` output if anyone wants a second opinion on T6's bake-off |
| Dense channel C4 (e5-small top-30, `blocking.dense`, off by default) | branch `dense-channel` (2 commits, touches Aditya's `features.py`, not merged). Written against main's `union.py`; Aditya rewrote his since, so it does not apply to origin/adi as is. Kaggle `ashuwhy/amlc26-dense` v4 (train split) started running 27 Sep 00:25. `dense_test.parquet` exists only in an older version's output (download from the Kaggle UI; the CLI fetches the latest version). Low value now: the loss that matters is precision, and C4 adds candidates. Cut-off stays 27 Sep 10:00 |
| `blocking-ports`, `c7-forward`, `cross-encoder`, `blocking-v2` branches | stale or merged; nothing to do |
| Local scratch artifacts (adi_art30, CE model copy, dense_test) | lost when /private/tmp was cleared |

## Decisions (26 Sep, still valid)

1. `candidate_pairs.tsv` = the pruned set the final model scores (organiser sheet, later answers win).
2. Final pick by dev OOF per country, public LB only to catch a broken file. After upload 2, add: a
   file that fails `src.acceptance` does not go up, whatever its OOF.
3. Candidate set size is a ranking criterion; a change that adds candidates must show OOF gain.
4. anyascii stays, disclosed in the methodology compliance section (Unicode table, ISC licence).
5. Leak check 7.0-8.6% by city grouping: accept, note it in the methodology.

## Open items

Nobody but Aditya and me has committed since 25 Sep 04:12 (Krishnkant's last commit). The release lane has
no owner in practice, and it holds the one deliverable we cannot ship without.

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | Final run (config_v5, decoy-aware dev), uploads 3-5, final pick | Aditya | in progress on origin/adi |
| 2 | **Methodology doc.** `student_resource/docs_draft/Documentation_draft.md` (origin/adi) stops at iteration 1 (08:09 26 Sep): `[v2]` placeholders, no C7/C8, no cross-encoder, no decoy finding, no compliance section. `docs/METHODOLOGY.md` on main is a 25 Sep skeleton | unowned (proposed for Krishnkant, no activity) | open, due before 16:00 freeze |
| 3 | Compliance section: anyascii (ISC), no libpostal, hand lists allowed, model licences (xlm-r-base MIT, mDeBERTa-v3 MIT, e5-small MIT), leak check, candidate_pairs definition | same | open |
| 4 | Upload bookkeeping: tags `sub/d<day>-<n>` (none exist, only `dataset-v1`), `docs/EXPERIMENTS.md` rows (table is empty) | same | open |
| 5 | Final zip: `python -m src.package --team <TEAM>` on origin/adi, official validator, README run on a small slice | same | 27 Sep |
| 6 | Dense channel measurement or drop by 10:00 | Ashutosh | running |

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
