"""D9 default: leave-one-country-out offset prior for France (ALLOW_TEST_PRED_FIT = false).

For each direction (US -> India, India -> US): stage-A LightGBM trained on the source country only
(5-fold OOF there for calibration and thresholds), then applied to the held-out country. The oracle
offset b* maximises the held-out F0.5 of p' = sigmoid(logit(p) + b) under the source thresholds.
b_FR = mean of the two b*. The offset + temperature variant sigmoid(a * logit(p) + b) is also fitted.
"""
from __future__ import annotations

import json
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import calibrate
from .assign import hard_assign
from .blocking.union import block_dir
from .cv import build_dev_universe
from .decision import two_threshold
from .featurize import feat_dir
from .metrics import per_entity
from .stack import META, PARAMS
from .utils import Timer, get_logger

log = get_logger()
B_GRID = np.round(np.arange(-3.0, 3.01, 0.1), 2)
A_GRID = np.array([0.6, 0.8, 1.0, 1.25, 1.5])


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _load(cfg, c, dev):
    df = pd.read_parquet(os.path.join(feat_dir(cfg), f"dev_{c}.parquet"))
    q = pd.read_parquet(os.path.join(block_dir(cfg), f"dev_{c}_q.parquet"), columns=["entity_id"])
    n_true = dev.set_index("entity_id")["n_true"].reindex(q["entity_id"]).to_numpy()
    return df, n_true


def _f(qi, ri, p, y, n_true, th):
    keep = hard_assign(ri, p)
    sel = two_threshold.select(qi[keep], p[keep], th["tau1"], th["tau2"])
    return per_entity(qi[keep], sel, y[keep], n_true).mean()


def direction(cfg, src, tgt, rounds, dev):
    t = Timer()
    ds, nts = _load(cfg, src, dev)
    dt_, ntt = _load(cfg, tgt, dev)
    cols = [c for c in ds.columns if c not in META]
    Xs, ys, fold = ds[cols].to_numpy(np.float32), ds["y"].to_numpy(), ds["fold"].to_numpy()
    Xt = dt_[cols].to_numpy(np.float32)
    oof, pt = np.zeros(len(ds), np.float32), np.zeros(len(dt_))
    params = {**PARAMS, "seed": cfg["seed"]}
    for f in range(5):
        m = lgb.train(params, lgb.Dataset(Xs[fold != f], ys[fold != f]), rounds)
        oof[fold == f] = m.predict(Xs[fold == f])
        pt += m.predict(Xt) / 5
    iso = calibrate.fit(oof, ys, seed=cfg["seed"])
    ps = iso.predict(oof)
    qs, rs = ds["qi"].to_numpy(), ds["ri"].to_numpy()
    keep = hard_assign(rs, ps)
    th = two_threshold.tune(qs[keep], ps[keep], ys[keep], nts)
    pcal = iso.predict(pt)
    qt, rt, yt = dt_["qi"].to_numpy(), dt_["ri"].to_numpy(), dt_["y"].to_numpy()
    lg = _logit(pcal)
    curve = {float(b): _f(qt, rt, 1 / (1 + np.exp(-(lg + b))), yt, ntt, th) for b in B_GRID}
    b_star = max(curve, key=curve.get)
    peak = curve[b_star]
    flat = [b for b, v in curve.items() if v >= peak - 0.005]
    temp = {}
    for a in A_GRID:
        for b in B_GRID[::2]:
            temp[(float(a), float(b))] = _f(qt, rt, 1 / (1 + np.exp(-(a * lg + b))), yt, ntt, th)
    best_ab = max(temp, key=temp.get)
    out = {"src": src, "tgt": tgt, "thresholds_src": th, "src_oof_f05": th["score"],
           "tgt_f05_b0": round(curve[0.0], 5), "b_star": b_star, "tgt_f05_bstar": round(peak, 5),
           "flat_region": [min(flat), max(flat)], "temp_best_ab": best_ab,
           "temp_best_f05": round(temp[best_ab], 5), "temp_gain_over_offset": round(temp[best_ab] - peak, 5),
           "runtime_s": t()}
    log.info("LOCO %s -> %s: %s", src, tgt, json.dumps(out))
    return out


def run(cfg: dict, split: str = "dev") -> dict:
    t = Timer()
    dev = build_dev_universe(cfg)
    st = pickle.load(open(os.path.join(feat_dir(cfg), "stageA.pkl"), "rb"))
    rounds = int(np.mean(st["iters"]))
    res = [direction(cfg, "US", "India", rounds, dev), direction(cfg, "India", "US", rounds, dev)]
    b_fr = float(np.mean([r["b_star"] for r in res]))
    path = os.path.join(cfg["paths"]["artifacts"], "decision.pkl")
    dec = pickle.load(open(path, "rb"))
    dec["b_fr"] = b_fr
    pickle.dump(dec, open(path, "wb"))
    incomplete = all(r["temp_gain_over_offset"] > 0.005 for r in res)
    json.dump({"directions": res, "b_fr": b_fr, "offset_alone_incomplete": incomplete},
              open(os.path.join(cfg["paths"]["runs"], "loco.json"), "w"), indent=1, default=str)
    return {"stage": "loco", "loco_US_to_India": res[0]["tgt_f05_b0"], "loco_India_to_US": res[1]["tgt_f05_b0"],
            "runtime_s": t(), "notes": json.dumps({"b_fr": b_fr, "b_star": [r["b_star"] for r in res],
                                                   "offset_alone_incomplete": incomplete})}
