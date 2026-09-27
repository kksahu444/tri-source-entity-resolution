import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import gbm, stack  # noqa: E402


def _fake(tmp, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    for d in ("block", "feat", "pred"):
        os.makedirs(f"{tmp}/{d}", exist_ok=True)
    for split in ("dev", "test"):
        pd.DataFrame({"qi": [0]}).to_parquet(f"{tmp}/block/{split}_US_pairs.parquet")
        y = (rng.random(n) < 0.3).astype(np.int8)
        df = pd.DataFrame({"qi": np.arange(n) // 6, "ri": np.arange(n), "a": (y + rng.normal(0, 0.5, n)).astype(np.float32),
                           "b": rng.integers(0, 5, n).astype(np.int8), "c": rng.random(n)})
        if split == "dev":
            df.insert(2, "y", y)
            df.insert(3, "fold", (df["qi"] % 5).astype(np.int8))
        df.to_parquet(f"{tmp}/feat/{split}_US.parquet", row_group_size=700)
    return {"seed": 1, "paths": {"artifacts": tmp, "runs": tmp}}


def test_feature_matrix_matches_pandas(tmp_path):
    _fake(str(tmp_path))
    p = str(tmp_path / "feat" / "dev_US.parquet")
    X = stack.feature_matrix([p, p], ["c", "a", "b"], extra=2)
    ref = pd.read_parquet(p)[["c", "a", "b"]].to_numpy(np.float32)
    assert X.shape == (2 * len(ref), 5) and X.dtype == np.float32
    assert np.array_equal(X[:len(ref), :3], ref) and np.array_equal(X[len(ref):, :3], ref)


def test_chunked_predict_equals_one_pass():
    rng = np.random.default_rng(0)
    X = rng.random((5000, 4)).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    m = gbm.train_binary({}, X, y, 20, seed=0, lgb_params={"objective": "binary", "verbose": -1})
    assert np.allclose(gbm.predict(m, X, chunk=777), gbm.predict(m, X))


def test_stage_a_dev_and_test_end_to_end(tmp_path):
    cfg = _fake(str(tmp_path))
    res = stack.run(cfg, "dev")
    dev = pd.read_parquet(tmp_path / "pred" / "dev_US_stageA.parquet")
    assert len(dev) == 3000 and dev["p"].between(0, 1).all() and "stageA" in str(os.listdir(tmp_path / "feat"))
    assert res["stage"] == "stack"
    stack.run(cfg, "test")
    te = pd.read_parquet(tmp_path / "pred" / "test_US_stageA.parquet")
    folds = pd.read_parquet(tmp_path / "pred" / "test_US_stageA_folds.parquet")
    assert len(te) == 3000 and folds.shape == (3000, 5) and np.allclose(te["p"], folds.mean(axis=1), atol=1e-6)
