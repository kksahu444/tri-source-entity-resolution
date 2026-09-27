# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Team entry for the Amazon ML Challenge 2026 (Business Entity Resolution). For every Source-1 business record, find all Source-2 and Source-3 records describing the same business. Scored by macro F0.5 per Source-1 entity. Window: 25 Sep 2026 00:00 IST to 27 Sep 2026 23:59 IST, 5 leaderboard uploads per day. Organisers' rules are in `student_resource/README.md`; the 72h plan, EDA numbers and backlog are in `docs/PLAN.md`.

Read `NEXTSESSION.md` first: current state, team decisions, owners, and which pipeline submissions come from (Aditya's, under `student_resource/code/business_entity_resolution/` on origin/main). Update it before ending a session.

## Commands

```bash
pip install -r requirements.txt            # pinned; Python >= 3.11 (developed on 3.14)
python main.py fetch-data                  # dataset/ from the private GitHub Release dataset-v1 (needs `gh auth login`)

python main.py eda | block | train | predict | loco
python main.py all                         # train + predict
python main.py --config configs/fast_debug.json train    # quick iteration (3 folds, smaller k, no stage 2)
python main.py validate --official student_resource/utils/validate_submission.py
python main.py package --team <TEAM>       # <TEAM>_submission.zip in the organisers' layout

# no real data needed:
python main.py synth && python main.py --data-dir data_synthetic all
python main.py score --truth data_synthetic/hidden_labels/test_ground_truth.tsv

python -m pytest -q                        # all tests (pyproject puts src/ on the path)
python -m pytest tests/test_decide.py -q   # one file; add -k <name> for one test
```

No linter or formatter is configured. `main.py` just prepends `src/` to `sys.path`, so the package works without `pip install -e .`.

## Architecture

One linear pipeline, orchestrated in `pipeline.py:prepare()`: `normalize_records` -> `build_text_index` -> `generate_candidates` -> `build_features` -> LightGBM -> `select_matches`. The working unit everywhere is a pairs DataFrame keyed by `s1_id, cand_id`; blocking adds `blk_*` columns, features add the rest, training adds `label` and `p`.

- **Config is the model's contract.** Every tunable lives in `config.py` dataclasses. `train` saves the resolved config inside `artifacts/model.joblib`, and `predict` rebuilds blocking and features from that saved config, not from `--config`. Only `data_dir`/`output_dir` are taken from the command line at predict time. Changing blocking or feature code without retraining silently mismatches the model. `Config.from_dict` raises on unknown keys.
- **Two-stage model** (`model.py`): stage 1 is GroupKFold-by-`s1_id` LightGBM; stage 2 adds group-relative ranks of the stage-1 OOF score and is kept only if its tuned OOF macro F beats stage 1. Test predictions average the fold models.
- **Decision layer** (`decide.py`): `tune_decision` grid-searches strategy (`threshold` or per-entity `expected_f`, where k=0 is allowed) x threshold x exclusivity on OOF scores and stores the winner in the bundle. Exclusivity = each S2/S3 id goes to at most one S1 (its best-scoring one); EDA confirmed 0 multi-S1 ids in train.
- **Group-context features** (`features.py:add_group_context`) rank a pair among its S1's candidates and among the S1s competing for the same candidate, so features depend on the whole candidate set, not just the pair.
- **Outputs**: `candidate_pairs.tsv` must be exactly the pairs the model scored, and every match must be a subset of it. `predict` writes both from the same `scored` frame and runs `validate.py` automatically on the test split; exit code 1 on validation errors.

## Constraints that shape the code

- **Country is an open set.** Test adds France, which is absent from train. The only country feature is `country_eq`. Never hard-code, filter or one-hot on `{US, India}`; every test S1 id must get a row. Normalisation dictionaries in `normalize.py` already mix US, Indian and French forms. Use `loco` to estimate unseen-country loss.
- **No external data or network calls** in the pipeline (disqualification). Dictionaries are hand-written. The only allowed download is the optional embedding model (`intfloat/multilingual-e5-small`, MIT). Any model used must be MIT or Apache-2.0 and <= 8B params.
- **F0.5 is precision-heavy and singletons count** (1.0 for an empty list, 0.0 for any prediction). When unsure, do not merge. Choose changes by OOF macro F0.5 in `artifacts/train_report.json`, not by public leaderboard swings.
- **Always read TSVs through `io.read_tsv`**: everything stays a string. Quoted fields are CSV-escaped (`""Niagara""`), so default quoting is the correct parse; `QUOTE_NONE` would keep the escape quotes.
- **pyarrow must be installed.** pandas 3 then stores `str` columns as Arrow (~420 B per normalised record instead of ~1,200 with Python strings); without it the India partition alone swaps on 16 GB.

## Scale

Real splits are about 12.5M records each (train S1/S2/S3 = 2.21M/5.03M/5.29M). `prepare()` runs one country at a time (0 cross-country true pairs), so only one partition's matrices are in memory. TF-IDF is hashed (`HASH_FEATURES`) and built in `CHUNK_ROWS` chunks across processes, the kNN transposes the target side a chunk at a time, and normalisation is parallel. The kNN cost grows with the number of S1 queries, so `char_max_df` is the speed knob (0.02: India S2 kNN 24 s per 8.7k queries). Train blocking uses `train_s1_sample`; every S2/S3 record stays in the pool.

## Team process

- Repo is private (`kksahu444/tri-source-entity-resolution`) and must stay private until results are announced. Team of four; the role table in `docs/PLAN.md` section 5 has no names assigned yet.
- Every leaderboard upload gets a commit, a tag `sub/d<day>-<n>` and a row in `docs/EXPERIMENTS.md` (OOF, public LB, config). Record who actually ran or built each entry.
- Code freeze 16:00 IST on 27 Sep; `docs/METHODOLOGY.md` is the filled-in organisers' template that `package` ships.
- `dataset/`, `artifacts/`, `output/`, `data_synthetic/` are git-ignored. The organisers' zip under `student_resource/` is tracked via Git LFS.
