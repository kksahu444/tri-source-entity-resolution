"""Stage `stack`: stage-A LightGBM on D4 features, 5-fold out-of-fold on dev; fold-average on test."""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import gbm
from .featurize import countries, feat_dir
from .utils import Timer, get_logger

log = get_logger()
META = ["qi", "ri", "y", "fold"]
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
          "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 1.0,
          "max_bin": 63, "verbose": -1, "num_threads": 0}
MAX_ROUNDS, EARLY = 1500, 30


def train_full(cfg, X, y, iters, t=None):
    """Optional single model on all dev pairs for test inference (config stack.test_models: full):
    1 prediction pass instead of 5 over ~26M test pairs. Rounds = 1.1 x mean fold best iteration
    (more data than a fold). OOF scores still come from the 5 fold models."""
    if cfg.get("stack", {}).get("test_models", "folds") != "full":
        return None
    rounds = int(1.1 * np.mean(iters))
    m = gbm.train_binary(cfg, X, y, rounds, seed=cfg["seed"], lgb_params=PARAMS)
    if t is not None:
        log.info("full model: %d rounds (%.0fs)", rounds, t())
    return m


def predict_test(st: dict, X: np.ndarray) -> np.ndarray:
    if st.get("full_model") is not None:
        return gbm.predict(st["full_model"], X)
    return np.mean([gbm.predict(m, X) for m in st["models"]], axis=0)


def feature_matrix(paths: list[str], cols: list[str], extra: int = 0) -> np.ndarray:
    """float32 matrix of ``cols`` stacked over the parquet files, filled one column at a time from Arrow
    (no full pandas copy: that doubled the peak and the iteration-5 GPU runs died of memory). ``extra``
    trailing columns are left for the caller."""
    n = [pq.ParquetFile(p).metadata.num_rows for p in paths]
    X = np.empty((sum(n), len(cols) + extra), np.float32)
    row = 0
    for p, k in zip(paths, n):
        for j in range(0, len(cols), 32):
            t = pq.read_table(p, columns=cols[j:j + 32])
            for i, col in enumerate(cols[j:j + 32]):
                X[row:row + k, j + i] = t.column(col).to_numpy()
            del t
        row += k
    return X


def pred_dir(cfg):
    d = os.path.join(cfg["paths"]["artifacts"], "pred")
    os.makedirs(d, exist_ok=True)
    return d


def run(cfg: dict, split: str = "dev") -> dict:
    t = Timer()
    fd, pd_ = feat_dir(cfg), pred_dir(cfg)
    model_path = os.path.join(fd, "stageA.pkl")
    cs = countries(cfg, split)
    if split == "dev":
        paths = [os.path.join(fd, f"dev_{c}.parquet") for c in cs]
        cols = [k for k in pq.read_schema(paths[0]).names if k not in META]
        df = pd.concat([pd.read_parquet(p, columns=META).assign(_c=c) for p, c in zip(paths, cs)], ignore_index=True)
        X = feature_matrix(paths, cols)
        y = df["y"].to_numpy()
        fold = df["fold"].to_numpy()
        # early-stopping holdout: S1s drawn inside the training folds (never the scored fold)
        rng = np.random.default_rng(cfg["seed"])
        s1_key = df["_c"].map({c: i for i, c in enumerate(cs)}).to_numpy(np.int64) * 10_000_000 + df["qi"].to_numpy()
        uniq = np.unique(s1_key)
        es_s1 = set(uniq[rng.random(len(uniq)) < 0.05].tolist())
        es_mask = np.isin(s1_key, list(es_s1))
        oof = np.zeros(len(df), np.float32)
        models, iters = [], []
        for f in range(5):
            tr = (fold != f) & ~es_mask
            es = (fold != f) & es_mask
            m = gbm.train_binary(cfg, X[tr], y[tr], MAX_ROUNDS, X[es], y[es], EARLY, cfg["seed"], PARAMS)
            va = fold == f
            oof[va] = m.predict(X[va])
            models.append(m)
            iters.append(m.best_iteration)
            log.info("stage A fold %d: best_iter %d (%.0fs)", f, m.best_iteration, t())
        imp = pd.Series(np.mean([m.importance(len(cols)) for m in models], axis=0), index=cols)
        imp = (imp / imp.sum()).sort_values(ascending=False)
        full = train_full(cfg, X, y, iters, t)
        pickle.dump({"models": models, "cols": cols, "iters": iters, "full_model": full}, open(model_path, "wb"))
        for c in cs:
            m = df["_c"].to_numpy() == c
            out = pd.DataFrame({"qi": df["qi"][m], "ri": df["ri"][m], "y": df["y"][m], "p": oof[m]})
            out.to_parquet(os.path.join(pd_, f"dev_{c}.parquet"), index=False)
            out.to_parquet(os.path.join(pd_, f"dev_{c}_stageA.parquet"), index=False)
        info = {"iters": iters, "n_pairs": len(df), "pos_rate": round(float(y.mean()), 4),
                "top_features": {k: round(v, 4) for k, v in imp.head(25).items()}}
        json.dump({"importance": imp.round(5).to_dict(), **info}, open(os.path.join(cfg["paths"]["runs"], "stageA_dev.json"), "w"), indent=1)
    else:
        st = pickle.load(open(model_path, "rb"))
        info = {}
        for c in cs:
            path = os.path.join(fd, f"test_{c}.parquet")
            ids = pd.read_parquet(path, columns=["qi", "ri"])
            qi_t, ri_t = ids["qi"].to_numpy(), ids["ri"].to_numpy()
            del ids
            X = feature_matrix([path], st["cols"])
            if st.get("full_model") is not None:
                p, per_fold = predict_test(st, X), None
            else:
                # keep every fold model's prediction: stage B's test context is built per fold (T2)
                per_fold = np.stack([gbm.predict(m, X) for m in st["models"]], axis=1).astype(np.float32)
                p = per_fold.mean(axis=1)
            out = pd.DataFrame({"qi": qi_t, "ri": ri_t, "p": p.astype(np.float32)})
            out.to_parquet(os.path.join(pd_, f"test_{c}.parquet"), index=False)
            out.to_parquet(os.path.join(pd_, f"test_{c}_stageA.parquet"), index=False)
            if per_fold is not None:
                pd.DataFrame(per_fold, columns=[f"p{f}" for f in range(per_fold.shape[1])]).to_parquet(
                    os.path.join(pd_, f"test_{c}_stageA_folds.parquet"), index=False)
            del X
            info[c] = len(qi_t)
            log.info("stage A test %s: %s pairs (%.0fs)", c, f"{len(qi_t):,}", t())
    return {"stage": "stack", "runtime_s": t(), "notes": json.dumps(info)}
