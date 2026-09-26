"""Stage `stack`: stage-A LightGBM on D4 features, 5-fold out-of-fold on dev; fold-average on test."""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

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
            df = pd.read_parquet(os.path.join(fd, f"test_{c}.parquet"))
            qi_t, ri_t = df["qi"].to_numpy(), df["ri"].to_numpy()
            X = df[st["cols"]].to_numpy(np.float32)
            del df
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
