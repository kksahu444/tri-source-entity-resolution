"""Error analysis on the dev universe: where macro F0.5 is lost (blocking vs model), with examples.

python -m src.error_analysis  ->  runs/error_analysis.md
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

from . import io_utils
from .assign import hard_assign
from .blocking.union import block_dir
from .cv import build_dev_universe
from .decision import two_threshold
from .featurize import countries
from .stack import pred_dir
from .utils import load_config


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    dec = pickle.load(open(os.path.join(cfg["paths"]["artifacts"], "decision.pkl"), "rb"))
    dev = build_dev_universe(cfg).set_index("entity_id")
    gt = io_utils.load_ground_truth(cfg["paths"]["dataset"])
    lines = ["# Error analysis (dev universe, out-of-fold)\n"]
    rng = np.random.default_rng(0)
    for c in countries(cfg, "dev"):
        cols = ["entity_id", "name_a", "addr_a", "f_nonlatin"]
        q = pd.read_parquet(os.path.join(block_dir(cfg), f"dev_{c}_q.parquet"), columns=cols)
        r = pd.read_parquet(os.path.join(block_dir(cfg), f"dev_{c}_r.parquet"), columns=cols)
        d = pd.read_parquet(os.path.join(pred_dir(cfg), f"dev_{c}.parquet"))
        p = dec["iso"].predict(d["p"].to_numpy())
        keep = hard_assign(d["ri"].to_numpy(), p)
        th = dec["thresholds"][c]
        s, pk, yk, rk = d["qi"].to_numpy()[keep], p[keep], d["y"].to_numpy()[keep], d["ri"].to_numpy()[keep]
        sel = two_threshold.select(s, pk, th["tau1"], th["tau2"])
        g = gt[gt["s1_id"].isin(set(q["entity_id"]))]
        rpos = pd.Series(np.arange(len(r)), index=r["entity_id"])
        qpos = pd.Series(np.arange(len(q)), index=q["entity_id"])
        n_r = np.int64(len(r))
        true_keys = qpos.reindex(g["s1_id"]).to_numpy(np.int64) * n_r + rpos.reindex(g["cand_id"]).to_numpy(np.int64)
        cand_keys = d["qi"].to_numpy(np.int64) * n_r + d["ri"].to_numpy(np.int64)
        kept_keys = s.astype(np.int64) * n_r + rk
        sel_keys = kept_keys[sel]
        n_true = len(true_keys)
        missed_block = ~np.isin(true_keys, cand_keys)
        lost_assign = np.isin(true_keys, cand_keys) & ~np.isin(true_keys, kept_keys)
        fn_model = np.isin(true_keys, kept_keys) & ~np.isin(true_keys, sel_keys)
        fp = sel & (yk == 0)
        lines.append(f"\n## {c}\n")
        lines.append(f"- true pairs {n_true:,}: missed by blocking/pruning {missed_block.mean():.4f}, "
                     f"removed by assignment {lost_assign.mean():.4f}, not selected by the rule {fn_model.mean():.4f}")
        lines.append(f"- selected pairs {sel.sum():,}: false positives {fp.sum():,} ({fp.mean() / max(sel.mean(), 1e-9):.4f} of selected)")
        tr = rpos.reindex(g["cand_id"]).to_numpy()
        nl = r["f_nonlatin"].to_numpy()[tr] == 1
        empty = r["addr_a"].to_numpy()[tr] == ""
        lines.append(f"- blocking misses: non-Latin record {nl[missed_block].mean():.3f} (base {nl.mean():.3f}), "
                     f"empty record address {empty[missed_block].mean():.3f} (base {empty.mean():.3f})")
        s1_of_fp = s[fp]
        singleton_fp = dev["n_true"].reindex(q["entity_id"].to_numpy()[s1_of_fp]).to_numpy() == 0
        lines.append(f"- false positives on true singletons: {singleton_fp.mean():.3f} of FPs")

        def show(title, qi_arr, ri_arr, n=12):
            lines.append(f"\n### {c}: {title}\n")
            if len(qi_arr) == 0:
                return
            pick = rng.choice(len(qi_arr), min(n, len(qi_arr)), replace=False)
            for i in pick:
                a, b = q.iloc[qi_arr[i]], r.iloc[ri_arr[i]]
                lines.append(f"- S1 `{a['name_a']} | {a['addr_a']}`  vs  `{b['name_a']} | {b['addr_a']}`")

        mt = true_keys[missed_block]
        show("true pairs missed by blocking", (mt // n_r).astype(int), (mt % n_r).astype(int))
        show("false positives", s[fp], rk[fp])
        ft = true_keys[fn_model]
        show("true pairs not selected", (ft // n_r).astype(int), (ft % n_r).astype(int))
    out = os.path.join(cfg["paths"]["runs"], "error_analysis.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
