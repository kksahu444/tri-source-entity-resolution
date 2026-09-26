"""Stage `decide`: calibration (D6) -> one S1 per record (D7) -> per-entity rule (D8) -> scores.

Rules compared out-of-fold: two thresholds (per country), GFM with independence, GFM with the count
model. The simpler rule is kept unless the paired bootstrap interval of the gain excludes zero.
France gets the pooled thresholds / count model and the offset b_FR (D9) applied to its calibrated
probabilities before count features, assignment and the rule.
"""
from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from . import calibrate
from .assign import hard_assign, soft_assign
from .blocking.union import block_dir
from .cv import build_dev_universe
from .decision import count_model, gfm_rule, two_threshold
from .featurize import countries, feat_dir
from .metrics import per_entity, summary
from .stack import pred_dir
from .utils import Timer, get_logger

log = get_logger()
AUX = ["src3", "addr_r_empty", "n_parts_q", "q_name_city_crowd"]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def apply_offset(p, b):
    return 1 / (1 + np.exp(-(_logit(p) + b))) if b else p


def _assign(cfg, ri, p):
    if cfg["flags"]["HARD_ASSIGNMENT"]:
        return hard_assign(ri, p)
    return np.ones(len(ri), bool)


def _soft(cfg, ri, p):
    return p if cfg["flags"]["HARD_ASSIGNMENT"] else soft_assign(ri, p)


def _entity_frame(qi, p, aux, n_s1):
    first = np.full(n_s1, -1)
    first[qi[::-1]] = np.arange(len(qi))[::-1]
    has = first >= 0
    parts = np.where(has, aux["n_parts_q"].to_numpy()[np.maximum(first, 0)], 0)
    crowd = np.where(has, aux["q_name_city_crowd"].to_numpy()[np.maximum(first, 0)], 0)
    return count_model.entity_features(qi, p, aux["src3"].to_numpy(), aux["addr_r_empty"].to_numpy(),
                                       n_s1, parts, crowd)


def _paired_gain_ci(fa, fb, n_boot=1000, seed=0):
    d = fb - fa
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(cfg: dict, split: str = "dev") -> dict:
    t = Timer()
    pdir, fd = pred_dir(cfg), feat_dir(cfg)
    path = os.path.join(cfg["paths"]["artifacts"], "decision.pkl")
    cs = countries(cfg, split)
    res = {"stage": "decide"}
    if split == "dev":
        dev = build_dev_universe(cfg).set_index("entity_id")
        preds = {c: pd.read_parquet(os.path.join(pdir, f"dev_{c}.parquet")) for c in cs}
        aux = {c: pd.read_parquet(os.path.join(fd, f"dev_{c}.parquet"), columns=AUX) for c in cs}
        p_all = np.concatenate([preds[c]["p"].to_numpy() for c in cs])
        y_all = np.concatenate([preds[c]["y"].to_numpy() for c in cs])
        iso = calibrate.fit(p_all, y_all, seed=cfg["seed"])
        info = {"ece_raw": round(calibrate.ece(p_all, y_all), 5)}
        E, th = {}, {}
        for c in cs:
            q = pd.read_parquet(os.path.join(block_dir(cfg), f"dev_{c}_q.parquet"), columns=["entity_id"])
            n_true = dev["n_true"].reindex(q["entity_id"]).to_numpy()
            fold_s1 = dev["fold"].reindex(q["entity_id"]).to_numpy()
            d = preds[c]
            p = iso.predict(d["p"].to_numpy())
            info[f"ece_cal_{c}"] = round(calibrate.ece(p, d["y"].to_numpy()), 5)
            keep = _assign(cfg, d["ri"].to_numpy(), p)
            p = _soft(cfg, d["ri"].to_numpy(), p)
            E[c] = {"qi": d["qi"].to_numpy()[keep], "p": p[keep], "y": d["y"].to_numpy()[keep],
                    "n_true": n_true, "fold": fold_s1, "aux": aux[c][keep].reset_index(drop=True),
                    "qi_all": d["qi"].to_numpy(), "y_all": d["y"].to_numpy()}
            th[c] = two_threshold.tune(E[c]["qi"], E[c]["p"], E[c]["y"], n_true)
            log.info("thresholds %s: %s (%.0fs)", c, th[c], t())
        off, parts = 0, []
        for c in cs:
            parts.append((E[c]["qi"] + off, E[c]["p"], E[c]["y"], E[c]["n_true"]))
            off += len(E[c]["n_true"])
        th["pooled"] = two_threshold.tune(*(np.concatenate([x[i] for x in parts]) for i in range(4)))
        # count models (pooled, no country), cross-fitted on the dev folds: K = 4 and the K whose
        # open-ended class holds <= 5% of dev entities (Appendix A guidance)
        X = pd.concat([_entity_frame(E[c]["qi"], E[c]["p"], E[c]["aux"], len(E[c]["n_true"])) for c in cs], ignore_index=True)
        counts = np.concatenate([count_model.counts_in_list(E[c]["qi"], E[c]["y"], len(E[c]["n_true"])) for c in cs])
        fold = np.concatenate([E[c]["fold"] for c in cs])
        k_auto = count_model.choose_k(counts)
        info["count_k_auto"] = k_auto
        cm = {}
        for k in sorted({4, k_auto}):
            yk = np.minimum(counts, k - 1)
            cprob, cmodels, temp = count_model.fit_oof(X, yk, fold, cfg["seed"], k)
            cm[k] = (cprob, cmodels, temp)
            info[f"count_K{k}_temp"] = round(temp, 4)
            info[f"count_K{k}_acc"] = round(float((cprob.argmax(1) == yk).mean()), 4)
            log.info("count model K=%d: temp %.3f acc %.4f (%.0fs)", k, temp, info[f"count_K{k}_acc"], t())
        # per-entity F for every rule
        rules = ["two_threshold", "gfm_indep"] + [f"gfm_count_K{k}" for k in cm]
        F = {r: [] for r in rules + ["ceiling"]}
        nts, ctry, off = [], [], 0
        for c in cs:
            e = E[c]
            n = len(e["n_true"])
            sel = two_threshold.select(e["qi"], e["p"], th[c]["tau1"], th[c]["tau2"])
            F["two_threshold"].append(per_entity(e["qi"], sel, e["y"], e["n_true"]))
            sel, _ = gfm_rule.select(e["qi"], e["p"], None)
            F["gfm_indep"].append(per_entity(e["qi"], sel, e["y"], e["n_true"]))
            for k, (cprob, _, _) in cm.items():
                sel, _ = gfm_rule.select(e["qi"], e["p"], cprob[off:off + n])
                F[f"gfm_count_K{k}"].append(per_entity(e["qi"], sel, e["y"], e["n_true"]))
            F["ceiling"].append(per_entity(e["qi_all"], e["y_all"] == 1, e["y_all"], e["n_true"]))
            nts.append(e["n_true"])
            ctry.append(np.full(n, c))
            off += n
            log.info("rules scored %s (%.0fs)", c, t())
        F = {k: np.concatenate(v) for k, v in F.items()}
        nt, ctry = np.concatenate(nts), np.concatenate(ctry)
        comp = {}
        for rule in rules[1:]:
            g, lo, hi = _paired_gain_ci(F["two_threshold"], F[rule])
            comp[rule] = {"f05": round(float(F[rule].mean()), 5), "gain_vs_tt": round(g, 5),
                          "gain_ci": [round(lo, 5), round(hi, 5)],
                          **{f"f05_{c}": round(float(F[rule][ctry == c].mean()), 5) for c in cs},
                          "f05_singleton": round(float(F[rule][nt == 0].mean()), 5)}
        chosen = "two_threshold"
        best_gain = 0.0
        for rule in rules[1:]:
            if comp[rule]["gain_ci"][0] > 0 and comp[rule]["gain_vs_tt"] > best_gain:
                chosen, best_gain = rule, comp[rule]["gain_vs_tt"]
        if len(cm) > 1:
            g, lo, hi = _paired_gain_ci(F["gfm_count_K4"], F[f"gfm_count_K{k_auto}"])
            info["gain_Kauto_vs_K4"] = {"gain": round(g, 5), "ci": [round(lo, 5), round(hi, 5)]}
        res.update(summary(F[chosen], nt, ctry))
        res["notes"] = json.dumps({
            **info, "rule": chosen, "thresholds": th, "rule_comparison": comp,
            "oof_f05_two_threshold": round(float(F["two_threshold"].mean()), 5),
            "oracle_candidate_ceiling": round(float(F["ceiling"].mean()), 5),
            "oracle_perfect_singleton": round(float(np.where(nt == 0, 1.0, F[chosen]).mean()), 5),
            "all_empty_score": round(float((nt == 0).mean()), 5)})
        prev = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
        k_used = int(chosen.split("_K")[1]) if chosen.startswith("gfm_count") else 4
        pickle.dump({"iso": iso, "thresholds": th, "rule": "gfm_count" if chosen.startswith("gfm_count") else chosen,
                     "count_K": k_used, "count_models": cm[k_used][1], "count_temp": cm[k_used][2],
                     "b_fr": prev.get("b_fr", 0.0)}, open(path, "wb"))
    else:
        dec = pickle.load(open(path, "rb"))
        out = []
        for c in cs:
            d = pd.read_parquet(os.path.join(pdir, f"test_{c}.parquet"))
            aux = pd.read_parquet(os.path.join(fd, f"test_{c}.parquet"), columns=AUX)
            p = dec["iso"].predict(d["p"].to_numpy())
            if c not in dec["thresholds"]:
                p = apply_offset(p, dec.get("b_fr", 0.0))
            keep = _assign(cfg, d["ri"].to_numpy(), p)
            p = _soft(cfg, d["ri"].to_numpy(), p)
            s, pk, rk = d["qi"].to_numpy()[keep], p[keep], d["ri"].to_numpy()[keep]
            n_s1 = int(pd.read_parquet(os.path.join(block_dir(cfg), f"test_{c}_q.parquet"), columns=["entity_id"]).shape[0])
            if dec["rule"] == "two_threshold":
                th = dec["thresholds"].get(c, dec["thresholds"]["pooled"])
                sel = two_threshold.select(s, pk, th["tau1"], th["tau2"])
            else:
                cp = None
                if dec["rule"] == "gfm_count":
                    X = _entity_frame(s, pk, aux[keep].reset_index(drop=True), n_s1)
                    cp = count_model.predict(dec["count_models"], dec["count_temp"], X)
                sel, _ = gfm_rule.select(s, pk, cp)
            out.append(pd.DataFrame({"country": c, "qi": s[sel], "ri": rk[sel], "p": pk[sel]}))
            res[f"n_matches_{c}"] = int(sel.sum())
            log.info("decide test %s: %d matches (%s, b_fr %.2f) (%.0fs)", c, sel.sum(), dec["rule"],
                     dec.get("b_fr", 0.0) if c not in dec["thresholds"] else 0.0, t())
        pd.concat(out, ignore_index=True).to_parquet(os.path.join(pdir, "test_selected.parquet"), index=False)
    res["runtime_s"] = t()
    return res
