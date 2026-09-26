"""Decision-layer ablations on the dev out-of-fold predictions (spec section 9, items 6-7), each
against the chosen configuration with a paired bootstrap interval over S1 entities.

python -m src.ablate [--config config_v2.yaml]  ->  run log rows tagged ablation:<name>, runs/ablations_decision.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

from .assign import hard_assign, soft_assign
from .blocking.union import block_dir
from .cv import build_dev_universe
from .decision import gfm_rule, two_threshold
from .decide import AUX, _entity_frame, _paired_gain_ci
from .decision import count_model
from .featurize import countries, feat_dir
from .metrics import per_entity
from .stack import pred_dir
from .utils import append_run_log, load_config, new_run_id


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    dec = pickle.load(open(os.path.join(cfg["paths"]["artifacts"], "decision.pkl"), "rb"))
    dev = build_dev_universe(cfg).set_index("entity_id")
    variants = {k: [] for k in ("chosen", "no_assignment", "soft_assignment", "tt_per_country", "tt_pooled",
                                "global_0.9", "gfm_indep")}
    for c in countries(cfg, "dev"):
        q = pd.read_parquet(os.path.join(block_dir(cfg), f"dev_{c}_q.parquet"), columns=["entity_id"])
        nt = dev["n_true"].reindex(q["entity_id"]).to_numpy()
        d = pd.read_parquet(os.path.join(pred_dir(cfg), f"dev_{c}.parquet"))
        aux = pd.read_parquet(os.path.join(feat_dir(cfg), f"dev_{c}.parquet"), columns=AUX)
        p = dec["iso"].predict(d["p"].to_numpy())
        qi, ri, y = d["qi"].to_numpy(), d["ri"].to_numpy(), d["y"].to_numpy()
        keep = hard_assign(ri, p)

        def rule(s, pp, yy, a, name=dec["rule"]):
            if name == "two_threshold":
                th = dec["thresholds"][c]
                return two_threshold.select(s, pp, th["tau1"], th["tau2"])
            cp = None
            if name == "gfm_count":
                cp = count_model.predict(dec["count_models"], dec["count_temp"], _entity_frame(s, pp, a, len(nt)))
            return gfm_rule.select(s, pp, cp)[0]

        a_k = aux[keep].reset_index(drop=True)
        variants["chosen"].append(per_entity(qi[keep], rule(qi[keep], p[keep], y[keep], a_k), y[keep], nt))
        variants["no_assignment"].append(per_entity(qi, rule(qi, p, y, aux), y, nt))
        ps = soft_assign(ri, p)
        variants["soft_assignment"].append(per_entity(qi, rule(qi, ps, y, aux), y, nt))
        th = dec["thresholds"][c]
        variants["tt_per_country"].append(per_entity(qi[keep], two_threshold.select(qi[keep], p[keep], th["tau1"], th["tau2"]), y[keep], nt))
        tp = dec["thresholds"]["pooled"]
        variants["tt_pooled"].append(per_entity(qi[keep], two_threshold.select(qi[keep], p[keep], tp["tau1"], tp["tau2"]), y[keep], nt))
        variants["global_0.9"].append(per_entity(qi[keep], two_threshold.select(qi[keep], p[keep], 0.9, 0.9), y[keep], nt))
        variants["gfm_indep"].append(per_entity(qi[keep], gfm_rule.select(qi[keep], p[keep], None)[0], y[keep], nt))
    F = {k: np.concatenate(v) for k, v in variants.items()}
    out = {"chosen_rule": dec["rule"], "chosen_f05": round(float(F["chosen"].mean()), 5)}
    for k, f in F.items():
        if k == "chosen":
            continue
        g, lo, hi = _paired_gain_ci(F["chosen"], f)
        out[k] = {"f05": round(float(f.mean()), 5), "delta_vs_chosen": round(g, 5), "ci": [round(lo, 5), round(hi, 5)]}
        append_run_log(cfg, {"run_id": new_run_id("ablate"), "stage": "ablate", "data_slice": f"dev:ablation:{k}",
                             "oof_f05": round(float(f.mean()), 5), "notes": json.dumps(out[k])})
    json.dump(out, open(os.path.join(cfg["paths"]["runs"], "ablations_decision.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
