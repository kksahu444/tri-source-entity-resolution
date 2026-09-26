"""Stage `stackb`: stage-B boosting on the D4 features + context features derived from stage-A
probabilities (CPU variant of D6 stage B while the cross-encoder is unavailable).

Dev: stage-A probabilities are out-of-fold (cross-fitted), stage B is trained with the same 5 folds,
so its out-of-fold output never saw its own labels. Test (config stack_b_test_context):
  "average"  - context from the 5-fold-average stage-A probability, stage-B fold models averaged;
  "per_fold" - for each fold f, context from stage-A fold model f and scoring by stage-B fold model f,
               then the 5 outputs are averaged, so test inputs are distributed like the OOF inputs
               stage B was trained on (work order T2).
Optional consistency features (config features.consistency): a candidate's probability-weighted max and
mean name/address similarity to the S1's other top candidates, and a flag for "no name overlap but the
address matches the S1 and a confident sibling". They are features only: nothing is chained.
Cross-encoder (work order T6): when stage `ce` produced out-of-fold logits, stage B is trained twice on
the same folds, without and with the logit as one extra feature (NaN outside the band). `decide` compares
the two on the same dev entities (paired bootstrap) and records the variant; the test pass uses it.
The result replaces the probability file consumed by `decide`.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd
from numba import njit
from rapidfuzz import fuzz

from . import gbm
from .blocking.union import block_dir
from .cross_encoder import ce_logits, stack_variant
from .featurize import countries, feat_dir
from .features import _col, _group_stats, _sim
from .stack import MAX_ROUNDS, EARLY, META, PARAMS, pred_dir, predict_test
from .utils import Timer, get_logger

log = get_logger()
F32 = np.float32
MEMBERS = 5          # top candidates per S1 (by blocking score) used as comparison members
CONF_P = 0.8         # "confident sibling"
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def context_features(qi, ri, pa, r_keys: dict, cons: dict | None = None, base: pd.DataFrame | None = None) -> pd.DataFrame:
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
    if cons is not None:
        _aggregate_consistency(f, pa, cons, base)
    return f


# ------------------------------------------------------------------ consistency (features only)
@njit(cache=True)
def _member_pairs(order, starts, sizes, m_max):
    total = 0
    for g in range(len(starts)):
        n = sizes[g]
        m = min(m_max, n)
        total += n * m - m
    ii = np.empty(total, np.int64)
    jj = np.empty(total, np.int64)
    k = 0
    for g in range(len(starts)):
        a, n = starts[g], sizes[g]
        m = min(m_max, n)
        for x in range(n):
            for y in range(m):
                if y != x:
                    ii[k] = order[a + x]
                    jj[k] = order[a + y]
                    k += 1
    return ii, jj


def consistency_pairs(qi, ri, c1, r) -> dict:
    """Candidate-member pairs within each S1 and their name/address similarities (computed once)."""
    order = np.lexsort((-c1, qi))
    qs = qi[order]
    starts = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
    sizes = np.diff(np.r_[starts, len(qs)])
    ii, jj = _member_pairs(order.astype(np.int64), starts.astype(np.int64), sizes.astype(np.int64), MEMBERS)
    name_sim = np.empty(len(ii), F32)
    addr_sim = np.empty(len(ii), F32)
    step = 2_000_000
    for s in range(0, len(ii), step):
        a, b = ri[ii[s:s + step]], ri[jj[s:s + step]]
        name_sim[s:s + step] = _sim(_col(r, "core", a), _col(r, "core", b), fuzz.token_set_ratio)
        addr_sim[s:s + step] = _sim(_col(r, "addr_core", a), _col(r, "addr_core", b), fuzz.token_set_ratio)
    return {"ii": ii, "jj": jj, "name_sim": name_sim, "addr_sim": addr_sim, "n": len(qi)}


def _aggregate_consistency(f: pd.DataFrame, pa: np.ndarray, cons: dict, base: pd.DataFrame | None) -> None:
    ii, jj, n = cons["ii"], cons["jj"], cons["n"]
    w = pa[jj].astype(np.float64)
    sw = np.bincount(ii, weights=w, minlength=n)
    for name in ("name", "addr"):
        sim = cons[f"{name}_sim"].astype(np.float64)
        ws = w * sim
        mx = np.zeros(n)
        np.maximum.at(mx, ii, ws)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(sw > 0, np.bincount(ii, weights=ws, minlength=n) / sw, 0)
        f[f"cons_{name}_wmax"] = mx.astype(F32)
        f[f"cons_{name}_wmean"] = mean.astype(F32)
    conf = pa[jj] >= CONF_P
    conf_addr = np.zeros(n)
    np.maximum.at(conf_addr, ii[conf], cons["addr_sim"][conf].astype(np.float64))
    f["cons_conf_addr_max"] = conf_addr.astype(F32)
    if base is not None and {"n_tok_ov", "a_tset"} <= set(base.columns):
        f["cons_newname_addr_match"] = ((base["n_tok_ov"].to_numpy() == 0) & (base["a_tset"].to_numpy() >= 0.9)
                                        & (conf_addr >= 0.9)).astype(F32)


def record_keys(r: pd.DataFrame) -> dict:
    keys = {}
    sort_tok = lambda x: " ".join(sorted(set(x.split())))
    for name, s in (("name", pd.Series(r["core"], dtype=object).map(sort_tok)),
                    ("addr", pd.Series(r["addr_core"], dtype=object).map(sort_tok)),
                    ("house", pd.Series(r["house"], dtype=object))):
        codes, _ = pd.factorize(s)
        keys[name] = np.where(s.to_numpy() == "", -1, codes)
    return keys


def _load(cfg, split, c):
    """Returns (features, stage-A probabilities, record keys, consistency pairs or None)."""
    fd, pdir = feat_dir(cfg), pred_dir(cfg)
    df = pd.read_parquet(os.path.join(fd, f"{split}_{c}.parquet"))
    pa = pd.read_parquet(os.path.join(pdir, f"{split}_{c}_stageA.parquet"))["p"].to_numpy()
    r = pd.read_parquet(os.path.join(block_dir(cfg), f"{split}_{c}_r.parquet"), columns=["core", "addr_core", "house"],
                        dtype_backend="pyarrow")
    keys = record_keys(r)
    cons = None
    if (cfg.get("features") or {}).get("consistency", False):
        cons = consistency_pairs(df["qi"].to_numpy(), df["ri"].to_numpy(), df["c1_score"].to_numpy(), r)
    return df, pa, keys, cons


CE_COL = "ce_logit"


def ce_feature(cfg, split, c, qi, ri) -> np.ndarray:
    """Cross-encoder logit per (qi, ri) pair; NaN for pairs outside the band."""
    lg = ce_logits(cfg, split, c)
    if lg is None:
        raise FileNotFoundError(f"no cross-encoder logits for {split} {c}")
    out = np.full(len(qi), np.nan, np.float32)
    if len(lg) == 0 or len(qi) == 0:
        return out
    n_r = np.int64(max(int(ri.max()), int(lg["ri"].max())) + 1)
    key = qi.astype(np.int64) * n_r + ri
    lk = lg["qi"].to_numpy(np.int64) * n_r + lg["ri"].to_numpy(np.int64)
    order = np.argsort(key, kind="stable")
    pos = np.minimum(np.searchsorted(key[order], lk), len(key) - 1)
    hit = key[order][pos] == lk
    out[order[pos[hit]]] = lg["logit"].to_numpy(np.float32)[hit]
    return out


def _model_path(fd, variant):
    return os.path.join(fd, "stageB.pkl" if variant == "base" else f"stageB_{variant}.pkl")


def _quantiles(x: np.ndarray) -> list:
    return [round(float(v), 4) for v in np.quantile(x, QUANTILES)] + [round(float(x.mean()), 4)]


def run(cfg: dict, split: str = "dev") -> dict:
    t = Timer()
    fd, pdir = feat_dir(cfg), pred_dir(cfg)
    cs = countries(cfg, split)
    info = {}
    if split == "dev":
        Xs, ys, folds, qis, ris, cc = [], [], [], [], [], []
        use_ce = all(ce_logits(cfg, "dev", c) is not None for c in cs)
        for i, c in enumerate(cs):
            df, pa, keys, cons = _load(cfg, split, c)
            ctx = context_features(df["qi"].to_numpy(), df["ri"].to_numpy(), pa, keys, cons, df)
            info[f"pa_quantiles_dev_{c}"] = _quantiles(pa)
            fcols = [k for k in df.columns if k not in META]
            cols = fcols + list(ctx.columns)
            parts = [df[fcols].to_numpy(np.float32), ctx.to_numpy(np.float32)]
            if use_ce:
                parts.append(ce_feature(cfg, "dev", c, df["qi"].to_numpy(), df["ri"].to_numpy())[:, None])
            Xs.append(np.hstack(parts))
            ys.append(df["y"].to_numpy())
            folds.append(df["fold"].to_numpy())
            qis.append(df["qi"].to_numpy())
            ris.append(df["ri"].to_numpy())
            cc.append(np.full(len(df), i))
            del df, ctx, cons
            log.info("stage B inputs %s (%.0fs)", c, t())
        X, y, fold = np.vstack(Xs), np.concatenate(ys), np.concatenate(folds)
        del Xs
        qi_all, ri_all, c_all = np.concatenate(qis), np.concatenate(ris), np.concatenate(cc)
        df = pd.DataFrame({"qi": qi_all, "ri": ri_all, "y": y, "_c": np.array(cs)[c_all]})
        rng = np.random.default_rng(cfg["seed"] + 1)
        s1_key = c_all.astype(np.int64) * 10_000_000 + qi_all
        uniq = np.unique(s1_key)
        es_mask = np.isin(s1_key, uniq[rng.random(len(uniq)) < 0.05])
        variants = {"base": cols}
        if use_ce:
            variants["ce"] = cols + [CE_COL]
            info["ce_band_share_dev"] = round(float(np.isfinite(X[:, -1]).mean()), 4)
        for v, vcols in variants.items():
            Xv = X[:, :len(vcols)]
            oof = np.zeros(len(df), np.float32)
            models = []
            for f in range(5):
                tr, es = (fold != f) & ~es_mask, (fold != f) & es_mask
                m = gbm.train_binary(cfg, Xv[tr], y[tr], MAX_ROUNDS, Xv[es], y[es], EARLY, cfg["seed"], PARAMS)
                oof[fold == f] = m.predict(Xv[fold == f])
                models.append(m)
                log.info("stage B %s fold %d: best_iter %d (%.0fs)", v, f, m.best_iteration, t())
            pickle.dump({"models": models, "cols": vcols, "full_model": None}, open(_model_path(fd, v), "wb"))
            for c in cs:
                m = df["_c"].to_numpy() == c
                out = pd.DataFrame({"qi": df["qi"][m], "ri": df["ri"][m], "y": df["y"][m], "p": oof[m]})
                out.to_parquet(os.path.join(pdir, f"dev_{c}_{v}.parquet"), index=False)
                if v == "base":
                    out.to_parquet(os.path.join(pdir, f"dev_{c}.parquet"), index=False)
            imp = pd.Series(np.mean([m.importance(len(vcols)) for m in models], axis=0), index=vcols)
            key = "" if v == "base" else f"_{v}"
            info.update({f"iters{key}": [m.best_iteration for m in models],
                         f"top_features{key}": (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()})
        info["variants"] = list(variants)
    else:
        variant = stack_variant(cfg)
        st = pickle.load(open(_model_path(fd, variant), "rb"))
        info["variant"] = variant
        mode = cfg.get("stack_b_test_context", "average")
        for c in cs:
            df, pa, keys, cons = _load(cfg, split, c)
            qi, ri = df["qi"].to_numpy(), df["ri"].to_numpy()
            folds_path = os.path.join(pdir, f"{split}_{c}_stageA_folds.parquet")
            info[f"pa_quantiles_test_avg_{c}"] = _quantiles(pa)
            fcols = [k for k in st["cols"] if k in df.columns]
            X = np.empty((len(df), len(st["cols"])), np.float32)
            pos = {k: i for i, k in enumerate(st["cols"])}
            X[:, [pos[k] for k in fcols]] = df[fcols].to_numpy(np.float32)
            if CE_COL in pos:
                X[:, pos[CE_COL]] = ce_feature(cfg, split, c, qi, ri)
                info[f"ce_band_share_test_{c}"] = round(float(np.isfinite(X[:, pos[CE_COL]]).mean()), 4)

            def fill_ctx(pa_x):
                ctx = context_features(qi, ri, pa_x, keys, cons, df)
                X[:, [pos[k] for k in ctx.columns if k in pos]] = ctx[[k for k in ctx.columns if k in pos]].to_numpy(np.float32)

            if mode == "per_fold" and os.path.exists(folds_path) and st.get("full_model") is None:
                pf = pd.read_parquet(folds_path).to_numpy()
                info[f"pa_quantiles_test_fold0_{c}"] = _quantiles(pf[:, 0])
                p = np.zeros(len(df))
                for f, m in enumerate(st["models"]):
                    fill_ctx(pf[:, f])
                    p += gbm.predict(m, X) / len(st["models"])
            else:
                fill_ctx(pa)
                p = predict_test(st, X)
            pd.DataFrame({"qi": qi, "ri": ri, "p": p.astype(F32)}).to_parquet(
                os.path.join(pdir, f"test_{c}.parquet"), index=False)
            info[c] = len(df)
            log.info("stage B test %s (%s): %s pairs (%.0fs)", c, mode, f"{len(df):,}", t())
            del df, X, cons
    return {"stage": "stackb", "runtime_s": t(), "notes": json.dumps(info)}
