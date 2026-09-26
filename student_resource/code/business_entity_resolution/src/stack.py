"""Stage `stack`: stage-A LightGBM on D4 features, 5-fold out-of-fold on dev; fold-average on test."""
from __future__ import annotations

import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from .featurize import countries, feat_dir
from .utils import Timer, get_logger

log = get_logger()
META = ["qi", "ri", "y", "fold"]
PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
          "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 1.0,
          "max_bin": 63, "verbose": -1, "num_threads": 0}
MAX_ROUNDS, EARLY = 1500, 30


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
        frames = [pd.read_parquet(os.path.join(fd, f"dev_{c}.parquet")).assign(_c=c) for c in cs]
        df = pd.concat(frames, ignore_index=True)
        del frames
        cols = [c for c in df.columns if c not in META and c != "_c"]
        X = df[cols].to_numpy(np.float32)
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
        params = {**PARAMS, "seed": cfg["seed"]}
        for f in range(5):
            tr = (fold != f) & ~es_mask
            es = (fold != f) & es_mask
            m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), MAX_ROUNDS,
                          valid_sets=[lgb.Dataset(X[es], y[es])],
                          callbacks=[lgb.early_stopping(EARLY, verbose=False)])
            va = fold == f
            oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
            models.append(m)
            iters.append(m.best_iteration)
            log.info("stage A fold %d: best_iter %d (%.0fs)", f, m.best_iteration, t())
        imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], axis=0), index=cols)
        imp = (imp / imp.sum()).sort_values(ascending=False)
        pickle.dump({"models": models, "cols": cols, "iters": iters}, open(model_path, "wb"))
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
            df = pd.read_parquet(os.path.join(fd, f"test_{c}.parquet"))
            X = df[st["cols"]].to_numpy(np.float32)
            p = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in st["models"]], axis=0)
            out = pd.DataFrame({"qi": df["qi"], "ri": df["ri"], "p": p.astype(np.float32)})
            out.to_parquet(os.path.join(pd_, f"test_{c}.parquet"), index=False)
            out.to_parquet(os.path.join(pd_, f"test_{c}_stageA.parquet"), index=False)
            info[c] = len(df)
            log.info("stage A test %s: %s pairs (%.0fs)", c, f"{len(df):,}", t())
    return {"stage": "stack", "runtime_s": t(), "notes": json.dumps(info)}
