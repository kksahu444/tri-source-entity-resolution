"""Stage `decide`: calibration (D6) -> one S1 per record (D7) -> per-entity rule (D8) -> scores.

France gets the pooled US+India thresholds and the offset b_FR (D9; 0 until the
leave-one-country-out prior has been measured).
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from . import calibrate
from .assign import hard_assign, soft_assign
from .cv import build_dev_universe
from .decision import two_threshold
from .featurize import countries
from .metrics import per_entity, summary
from .stack import pred_dir
from .blocking.union import block_dir
from .utils import Timer, get_logger

log = get_logger()


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_offset(p, b):
    return 1 / (1 + np.exp(-(_logit(p) + b))) if b else p


def _assign(cfg, ri, p):
    if cfg["flags"]["HARD_ASSIGNMENT"]:
        return hard_assign(ri, p)
    return soft_assign(ri, p)


def run(cfg: dict, split: str = "dev") -> dict:
    t = Timer()
    pdir = pred_dir(cfg)
    path = os.path.join(cfg["paths"]["artifacts"], "decision.pkl")
    cs = countries(cfg, split)
    res = {"stage": "decide"}
    if split == "dev":
        dev = build_dev_universe(cfg)
        preds = {c: pd.read_parquet(os.path.join(pdir, f"dev_{c}.parquet")) for c in cs}
        p_all = np.concatenate([preds[c]["p"].to_numpy() for c in cs])
        y_all = np.concatenate([preds[c]["y"].to_numpy() for c in cs])
        iso = calibrate.fit(p_all, y_all, seed=cfg["seed"])
        info = {"ece_raw": round(calibrate.ece(p_all, y_all), 5)}
        per_c, th = {}, {}
        for c in cs:
            q = pd.read_parquet(os.path.join(block_dir(cfg), f"dev_{c}_q.parquet"), columns=["entity_id"])
            n_true = dev.set_index("entity_id")["n_true"].reindex(q["entity_id"]).to_numpy()
            d = preds[c]
            p = iso.predict(d["p"].to_numpy())
            info[f"ece_cal_{c}"] = round(calibrate.ece(p, d["y"].to_numpy()), 5)
            keep = _assign(cfg, d["ri"].to_numpy(), p)
            per_c[c] = (d["qi"].to_numpy()[keep], p[keep], d["y"].to_numpy()[keep], n_true, d["qi"].to_numpy(), d["y"].to_numpy())
            th[c] = two_threshold.tune(per_c[c][0], per_c[c][1], per_c[c][2], n_true)
            log.info("thresholds %s: %s (%.0fs)", c, th[c], t())
        # pooled thresholds (for France)
        off, s_all, p_pool, y_pool, nt_pool = 0, [], [], [], []
        for c in cs:
            s, p, y, nt = per_c[c][:4]
            s_all.append(s + off); p_pool.append(p); y_pool.append(y); nt_pool.append(nt)
            off += len(nt)
        th["pooled"] = two_threshold.tune(np.concatenate(s_all), np.concatenate(p_pool), np.concatenate(y_pool),
                                          np.concatenate(nt_pool))
        # final OOF score with per-country thresholds, plus oracles
        fs, nts, ctry, f_ceiling, f_pooled = [], [], [], [], []
        for c in cs:
            s, p, y, nt, s_raw, y_raw = per_c[c]
            sel = two_threshold.select(s, p, th[c]["tau1"], th[c]["tau2"])
            fs.append(per_entity(s, sel, y, nt))
            selp = two_threshold.select(s, p, th["pooled"]["tau1"], th["pooled"]["tau2"])
            f_pooled.append(per_entity(s, selp, y, nt))
            f_ceiling.append(per_entity(s_raw, y_raw == 1, y_raw, nt))   # predict exactly the retrieved true pairs
            nts.append(nt)
            ctry.append(np.full(len(nt), c))
        f, nt, ctry = np.concatenate(fs), np.concatenate(nts), np.concatenate(ctry)
        res.update(summary(f, nt, ctry))
        res["notes"] = json.dumps({
            **info, "thresholds": th,
            "oof_pooled_thresholds": round(float(np.concatenate(f_pooled).mean()), 5),
            "oracle_candidate_ceiling": round(float(np.concatenate(f_ceiling).mean()), 5),
            "oracle_perfect_singleton": round(float(np.where(nt == 0, 1.0, f).mean()), 5),
            "all_empty_score": round(float((nt == 0).mean()), 5)})
        pickle.dump({"iso": iso, "thresholds": th, "b_fr": 0.0}, open(path, "wb"))
    else:
        dec = pickle.load(open(path, "rb"))
        out = []
        for c in cs:
            d = pd.read_parquet(os.path.join(pdir, f"test_{c}.parquet"))
            p = dec["iso"].predict(d["p"].to_numpy())
            if c not in dec["thresholds"]:
                p = apply_offset(p, dec.get("b_fr", 0.0))
            keep = _assign(cfg, d["ri"].to_numpy(), p)
            th = dec["thresholds"].get(c, dec["thresholds"]["pooled"])
            s, pk = d["qi"].to_numpy()[keep], p[keep]
            sel = two_threshold.select(s, pk, th["tau1"], th["tau2"])
            out.append(pd.DataFrame({"country": c, "qi": s[sel], "ri": d["ri"].to_numpy()[keep][sel], "p": pk[sel]}))
            res[f"n_matches_{c}"] = int(sel.sum())
            log.info("decide test %s: %d matches, thresholds %s", c, sel.sum(), th)
        pd.concat(out, ignore_index=True).to_parquet(os.path.join(pdir, "test_selected.parquet"), index=False)
    res["runtime_s"] = t()
    return res
