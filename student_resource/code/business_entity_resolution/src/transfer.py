"""Work order T2: transfer checks without labels. Compares dev (out-of-fold) with test per country and
flags any metric whose test value differs from dev by more than 10% relative.

python -m src.transfer --config <config> --tag <run tag>  ->  runs/transfer_checks_<tag>.md
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .assign import hard_assign
from .blocking.union import block_dir
from .decide import AUX, _entity_frame
from .decision import count_model, gfm_rule, two_threshold
from .featurize import countries, feat_dir
from .stack import pred_dir
from .utils import load_config

Q = (0.05, 0.25, 0.5, 0.75, 0.95)


def _select(dec, c, qi, ri, p_raw, aux, n_s1):
    p = dec["iso"].predict(p_raw)
    keep = hard_assign(ri, p)
    s, pk = qi[keep], p[keep]
    if dec["rule"] == "two_threshold":
        th = dec["thresholds"].get(c, dec["thresholds"]["pooled"])
        sel = two_threshold.select(s, pk, th["tau1"], th["tau2"])
    else:
        cp = None
        if dec["rule"] == "gfm_count":
            cp = count_model.predict(dec["count_models"], dec["count_temp"], _entity_frame(s, pk, aux[keep].reset_index(drop=True), n_s1))
        sel, _ = gfm_rule.select(s, pk, cp)
    return p, s, sel


def metrics(dec, split, c, cfg) -> dict:
    pdir, fd, bd = pred_dir(cfg), feat_dir(cfg), block_dir(cfg)
    d = pd.read_parquet(os.path.join(pdir, f"{split}_{c}.parquet"))
    qi, ri, p_raw = d["qi"].to_numpy(), d["ri"].to_numpy(), d["p"].to_numpy()
    n_s1 = pq.read_metadata(os.path.join(bd, f"{split}_{c}_q.parquet")).num_rows
    n_r = pq.read_metadata(os.path.join(bd, f"{split}_{c}_r.parquet")).num_rows
    aux = pd.read_parquet(os.path.join(fd, f"{split}_{c}.parquet"), columns=AUX)
    p, s, sel = _select(dec, c, qi, ri, p_raw, aux, n_s1)
    best = np.zeros(n_r)
    np.maximum.at(best, ri, p)
    n_hi = np.bincount(qi[p >= 0.5], minlength=n_s1)
    n_kept = np.bincount(s[sel], minlength=n_s1)
    out = {
        "a_share_records_bestp_ge_0.5": float((best >= 0.5).mean()),
        "b_cands_p_ge_0.5_per_s1": float(n_hi.mean()),
        "b_links_kept_per_s1": float(n_kept.mean()),
        "b_kept_over_p_ge_0.5": float(n_kept.sum() / max(n_hi.sum(), 1)),
        "c_empty_list_rate": float((n_kept == 0).mean()),
        "e_cands_per_s1": float(len(qi) / n_s1),
        "e_share_records_in_a_list": float(len(np.unique(ri)) / n_r),
        "d_final_p_mean": float(p.mean()),
    }
    for k, v in zip(Q, np.quantile(p, Q)):
        out[f"d_final_p_q{int(k * 100):02d}"] = float(v)
    sa = os.path.join(pdir, f"{split}_{c}_stageA.parquet")
    if os.path.exists(sa):
        pa = pd.read_parquet(sa)["p"].to_numpy()
        out["d_stageA_p_mean"] = float(pa.mean())
        for k, v in zip(Q, np.quantile(pa, Q)):
            out[f"d_stageA_p_q{int(k * 100):02d}"] = float(v)
    if split == "dev":
        out["a_dev_true_matched_share"] = float(len(np.unique(ri[d["y"].to_numpy() == 1])) / n_r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    tag = args.tag or cfg["_hash"]
    dec = pickle.load(open(os.path.join(cfg["paths"]["artifacts"], "decision.pkl"), "rb"))
    lines = [f"# Transfer checks (work order T2), tag `{tag}`, rule `{dec['rule']}`\n",
             "Flag = test differs from dev by more than 10% relative (US/India). France has no dev counterpart; "
             "it is compared with the dev average of US and India.\n"]
    dev = {c: metrics(dec, "dev", c, cfg) for c in countries(cfg, "dev")}
    test = {c: metrics(dec, "test", c, cfg) for c in countries(cfg, "test")}
    dev_avg = {k: float(np.mean([dev[c][k] for c in dev if k in dev[c]])) for k in next(iter(dev.values()))}
    flags = []
    for c in test:
        ref = dev.get(c, dev_avg)
        lines += [f"\n## {c} (dev reference: {c if c in dev else 'mean of US and India'})\n",
                  "| Metric | Dev | Test | Rel. diff | Flag |", "|---|---|---|---|---|"]
        for k in sorted(test[c]):
            if k not in ref:
                continue
            dv, tv = ref[k], test[c][k]
            rel = (tv - dv) / abs(dv) if abs(dv) > 1e-9 else 0.0
            flag = abs(rel) > 0.10
            if flag:
                flags.append(f"{c}:{k}")
            lines.append(f"| {k} | {dv:.4f} | {tv:.4f} | {rel:+.1%} | {'FLAG' if flag else ''} |")
        if c in dev and "a_dev_true_matched_share" in dev[c]:
            lines.append(f"\nDev true matched share of records: {dev[c]['a_dev_true_matched_share']:.4f}")
    lines.insert(2, f"Flags: {', '.join(flags) if flags else 'none'}\n")
    path = os.path.join(cfg["paths"]["runs"], f"transfer_checks_{tag}.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump({"dev": dev, "test": test, "flags": flags}, open(path.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
