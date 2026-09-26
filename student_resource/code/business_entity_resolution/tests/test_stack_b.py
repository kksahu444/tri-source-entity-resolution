import os
import pickle
import sys
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import stack_b  # noqa: E402
from src.stack_b import consistency_pairs, context_features, record_keys  # noqa: E402


def test_context_features_siblings_and_competition():
    r = pd.DataFrame({"core": ["acme", "acme", "beta", ""], "addr_core": ["1 main", "1 main", "2 elm", ""],
                      "house": ["1", "1", "2", ""]})
    keys = record_keys(r)
    qi = np.array([0, 0, 0, 0, 1])
    ri = np.array([0, 1, 2, 3, 2])
    pa = np.array([0.9, 0.4, 0.3, 0.2, 0.8])
    f = context_features(qi, ri, pa, keys)
    assert f["pa_rank"].tolist() == [1, 2, 3, 4, 1]
    assert np.allclose(f["pa_sib_name"], [0.4, 0.9, 0.0, 0.0, 0.0])      # other "acme" candidate of the same S1
    assert np.allclose(f["pa_rev_gap"], [0, 0, 0.5, 0, 0])               # record 2 is contested by S1 1 (0.8)
    assert np.allclose(f["pa_others_sum"][:4], [0.9, 1.4, 1.5, 1.6])
    assert np.isfinite(f.to_numpy()).all()


def test_consistency_features():
    r = pd.DataFrame({"core": ["indo inds", "indo inds", "other co", "zzz"],
                      "addr_core": ["34 main road haliyal", "34 main road haliyal", "9 elm", "34 main road haliyal"],
                      "house": ["34", "34", "9", "34"]})
    qi = np.array([0, 0, 0, 0])
    ri = np.array([0, 1, 2, 3])
    c1 = np.array([0.9, 0.8, 0.5, 0.4])
    pa = np.array([0.95, 0.9, 0.1, 0.3])
    cons = consistency_pairs(qi, ri, c1, r)
    assert len(cons["ii"]) == 4 * 4 - 4                                   # 4 candidates x 4 members, self excluded
    base = pd.DataFrame({"n_tok_ov": [2, 2, 0, 0], "a_tset": [1.0, 1.0, 0.1, 0.95]})
    f = context_features(qi, ri, pa, record_keys(r), cons, base)
    # record 3: new name, address equals the confident siblings 0 and 1 -> flag set; record 2 -> not
    assert f["cons_newname_addr_match"].tolist() == [0, 0, 0, 1]
    assert f["cons_addr_wmax"].iloc[3] > 0.9 and f["cons_name_wmax"].iloc[3] < 0.5
    assert f["cons_conf_addr_max"].iloc[2] < 0.5


def _fake_run_dir(consistency):
    tmp = tempfile.mkdtemp()
    for d in ("block", "feat", "pred"):
        os.makedirs(f"{tmp}/{d}")
    n = 240
    rng = np.random.default_rng(0)
    pd.DataFrame({"qi": [0]}).to_parquet(f"{tmp}/block/test_US_pairs.parquet")
    pd.DataFrame({"core": [f"name {i % 7}" for i in range(n)], "addr_core": [f"{i % 5} road" for i in range(n)],
                  "house": [str(i % 5) for i in range(n)]}).to_parquet(f"{tmp}/block/test_US_r.parquet")
    df = pd.DataFrame({"qi": np.arange(n) // 6, "ri": np.arange(n), "a": rng.random(n), "c1_score": rng.random(n),
                       "n_tok_ov": rng.integers(0, 2, n), "a_tset": rng.random(n)})
    df.to_parquet(f"{tmp}/feat/test_US.parquet")
    pa = rng.random(n)
    pd.DataFrame({"qi": df.qi, "ri": df.ri, "p": pa}).to_parquet(f"{tmp}/pred/test_US_stageA.parquet")
    pd.DataFrame({f"p{f}": np.clip(pa + 0.05 * f, 0, 1) for f in range(5)}).to_parquet(f"{tmp}/pred/test_US_stageA_folds.parquet")
    cfg = {"seed": 1, "paths": {"artifacts": tmp, "runs": tmp}, "features": {"consistency": consistency}}
    keys = record_keys(pd.read_parquet(f"{tmp}/block/test_US_r.parquet"))
    cons = consistency_pairs(df.qi.to_numpy(), df.ri.to_numpy(), df.c1_score.to_numpy(),
                             pd.read_parquet(f"{tmp}/block/test_US_r.parquet")) if consistency else None
    ctx = context_features(df.qi.to_numpy(), df.ri.to_numpy(), pa, keys, cons, df)
    cols = ["a", "c1_score", "n_tok_ov", "a_tset"] + list(ctx.columns)
    X = np.hstack([df[cols[:4]].to_numpy(np.float32), ctx.to_numpy(np.float32)])
    y = (df["a"] > 0.5).astype(int)
    ms = [lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(X, y), 5) for _ in range(5)]
    pickle.dump({"models": ms, "cols": cols, "full_model": None}, open(f"{tmp}/feat/stageB.pkl", "wb"))
    return cfg, tmp


def test_stackb_test_modes_end_to_end():
    for consistency in (False, True):
        for mode in ("average", "per_fold"):
            cfg, tmp = _fake_run_dir(consistency)
            cfg["stack_b_test_context"] = mode
            res = stack_b.run(cfg, "test")
            out = pd.read_parquet(f"{tmp}/pred/test_US.parquet")
            assert len(out) == 240 and out["p"].between(0, 1).all(), (consistency, mode)
            notes = res["notes"]
            assert ("pa_quantiles_test_fold0_US" in notes) == (mode == "per_fold")
