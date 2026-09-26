"""Stage `stackb`: stage-B LightGBM on the D4 features + context features derived from stage-A
probabilities (CPU variant of D6 stage B while the cross-encoder is unavailable).

Dev: stage-A probabilities are out-of-fold (cross-fitted), stage B is trained with the same 5 folds,
so its out-of-fold output never saw its own labels. Test: stage-A fold-average probabilities feed the
stage-B fold models. The result replaces the stage-A probability file consumed by `decide`.
"""
from __future__ import annotations

import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking.union import block_dir
from .featurize import countries, feat_dir
from .features import _group_stats
from .stack import MAX_ROUNDS, EARLY, META, PARAMS, pred_dir
from .utils import Timer, get_logger

log = get_logger()
F32 = np.float32


def context_features(qi, ri, pa, r_keys: dict) -> pd.DataFrame:
    f = pd.DataFrame({"pa": pa.astype(F32)})
    gmax, gsize, grank = _group_stats(qi, pa)
    f["pa_rank"] = grank.astype(F32)
    f["pa_gap"] = (gmax - pa).astype(F32)
    tot = np.bincount(qi, weights=pa)
    f["pa_others_sum"] = (tot[qi] - pa).astype(F32)
    f["pa_q_n05"] = np.bincount(qi, weights=(pa > 0.5).astype(float))[qi].astype(F32)
    rmax, rsize, rrank = _group_stats(ri, pa)
    f["pa_rev_rank"] = rrank.astype(F32)
    f["pa_rev_gap"] = (rmax - pa).astype(F32)
    rtot = np.bincount(ri, weights=pa)
    f["pa_rev_others_sum"] = (rtot[ri] - pa).astype(F32)
    # sibling agreement: best stage-A probability among OTHER candidates of the same S1 sharing a key
    for name, codes in r_keys.items():
        c = codes[ri]
        key = qi.astype(np.int64) * (int(codes.max()) + 2) + c + 1
        order = np.lexsort((-pa, key))
        ks = key[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
        sizes = np.diff(np.r_[starts, len(ks)])
        first = np.repeat(pa[order][starts], sizes)
        second = np.zeros(len(ks))
        has2 = sizes >= 2
        second_vals = np.zeros(len(starts))
        second_vals[has2] = pa[order][starts[has2] + 1]
        second = np.repeat(second_vals, sizes)
        is_first = np.zeros(len(ks), bool)
        is_first[starts] = True
        other_best = np.where(is_first, second, first)
        out = np.empty(len(pa))
        out[order] = other_best
        f[f"pa_sib_{name}"] = np.where(c >= 0, out, 0).astype(F32)
    return f


def record_keys(r: pd.DataFrame) -> dict:
    keys = {}
    sort_tok = lambda x: " ".join(sorted(set(x.split())))
    for name, s in (("name", r["core"].map(sort_tok)), ("addr", r["addr_core"].map(sort_tok)), ("house", r["house"])):
        codes, _ = pd.factorize(s)
        keys[name] = np.where(s.to_numpy() == "", -1, codes)
    return keys


def _load(cfg, split, c):
    fd, pdir = feat_dir(cfg), pred_dir(cfg)
    df = pd.read_parquet(os.path.join(fd, f"{split}_{c}.parquet"))
    pa = pd.read_parquet(os.path.join(pdir, f"{split}_{c}_stageA.parquet"))["p"].to_numpy()
    r = pd.read_parquet(os.path.join(block_dir(cfg), f"{split}_{c}_r.parquet"), columns=["core", "addr_core", "house"])
    ctx = context_features(df["qi"].to_numpy(), df["ri"].to_numpy(), pa, record_keys(r))
    return df, ctx


def run(cfg: dict, split: str = "dev") -> dict:
    t = Timer()
    fd, pdir = feat_dir(cfg), pred_dir(cfg)
    model_path = os.path.join(fd, "stageB.pkl")
    cs = countries(cfg, split)
    if split == "dev":
        frames = []
        for c in cs:
            df, ctx = _load(cfg, split, c)
            frames.append(pd.concat([df, ctx], axis=1).assign(_c=c))
        df = pd.concat(frames, ignore_index=True)
        del frames
        cols = [c for c in df.columns if c not in META and c != "_c"]
        X, y, fold = df[cols].to_numpy(np.float32), df["y"].to_numpy(), df["fold"].to_numpy()
        rng = np.random.default_rng(cfg["seed"] + 1)
        s1_key = df["_c"].map({c: i for i, c in enumerate(cs)}).to_numpy(np.int64) * 10_000_000 + df["qi"].to_numpy()
        uniq = np.unique(s1_key)
        es_mask = np.isin(s1_key, uniq[rng.random(len(uniq)) < 0.05])
        oof = np.zeros(len(df), np.float32)
        models = []
        for f in range(5):
            tr, es = (fold != f) & ~es_mask, (fold != f) & es_mask
            m = lgb.train({**PARAMS, "seed": cfg["seed"]}, lgb.Dataset(X[tr], y[tr]), MAX_ROUNDS,
                          valid_sets=[lgb.Dataset(X[es], y[es])], callbacks=[lgb.early_stopping(EARLY, verbose=False)])
            oof[fold == f] = m.predict(X[fold == f], num_iteration=m.best_iteration)
            models.append(m)
            log.info("stage B fold %d: best_iter %d (%.0fs)", f, m.best_iteration, t())
        pickle.dump({"models": models, "cols": cols}, open(model_path, "wb"))
        for c in cs:
            m = df["_c"].to_numpy() == c
            pd.DataFrame({"qi": df["qi"][m], "ri": df["ri"][m], "y": df["y"][m], "p": oof[m]}).to_parquet(
                os.path.join(pdir, f"dev_{c}.parquet"), index=False)
        imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], axis=0), index=cols)
        info = {"iters": [m.best_iteration for m in models],
                "top_features": (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()}
    else:
        st = pickle.load(open(model_path, "rb"))
        info = {}
        for c in cs:
            df, ctx = _load(cfg, split, c)
            X = pd.concat([df, ctx], axis=1)[st["cols"]].to_numpy(np.float32)
            p = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in st["models"]], axis=0)
            pd.DataFrame({"qi": df["qi"], "ri": df["ri"], "p": p.astype(F32)}).to_parquet(
                os.path.join(pdir, f"test_{c}.parquet"), index=False)
            info[c] = len(df)
    return {"stage": "stackb", "runtime_s": t(), "notes": json.dumps(info)}
