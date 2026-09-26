"""Where is macro F0.5 lost? Exact per-entity loss decomposition on the dev universe (OOF predictions,
the chosen decision rule), plus examples.

For every S1: F = actual score; F_noFP = score with the false positives removed; F_ceil = score with
every retrieved true record selected and no false positive. Then
    loss_false_links    = F_noFP - F
    loss_model_misses   = F_ceil - F_noFP
    loss_blocking       = 1 - F_ceil
sum exactly to 1 - F per entity; their means decompose 1 - macro F0.5.

python -m src.error_analysis --config config_v2.yaml  ->  runs/error_analysis.md (memory-light: Arrow)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .assign import hard_assign
from .blocking.union import block_dir
from .cv import build_dev_universe
from .decide import AUX, _entity_frame
from .decision import count_model, gfm_rule, two_threshold
from .featurize import countries, feat_dir
from .metrics import entity_f
from .stack import pred_dir
from .utils import load_config


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    dec = pickle.load(open(os.path.join(cfg["paths"]["artifacts"], "decision.pkl"), "rb"))
    dev = build_dev_universe(cfg).set_index("entity_id")
    gt = pd.read_csv(os.path.join(cfg["paths"]["dataset"], "train", "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    rng = np.random.default_rng(0)
    lines = [f"# Error analysis: where macro F0.5 is lost (dev OOF, rule `{dec['rule']}`)\n"]
    rows = []
    for c in countries(cfg, "dev"):
        qt = pq.read_table(os.path.join(block_dir(cfg), f"dev_{c}_q.parquet"), columns=["entity_id", "name_a", "addr_a"])
        rt = pq.read_table(os.path.join(block_dir(cfg), f"dev_{c}_r.parquet"),
                           columns=["entity_id", "name_a", "addr_a", "f_nonlatin", "addr_core"])
        q_ids = qt["entity_id"].to_numpy(zero_copy_only=False)
        r_ids = rt["entity_id"].to_numpy(zero_copy_only=False)
        r_nonlatin = rt["f_nonlatin"].to_numpy()
        r_empty = pc.equal(rt["addr_core"], "").to_numpy(zero_copy_only=False)
        d = pd.read_parquet(os.path.join(pred_dir(cfg), f"dev_{c}.parquet"))
        aux = pd.read_parquet(os.path.join(feat_dir(cfg), f"dev_{c}.parquet"), columns=AUX)
        p = dec["iso"].predict(d["p"].to_numpy())
        keep = hard_assign(d["ri"].to_numpy(), p)
        s, pk, yk, rk = d["qi"].to_numpy()[keep], p[keep], d["y"].to_numpy()[keep], d["ri"].to_numpy()[keep]
        n_q = len(q_ids)
        if dec["rule"] == "gfm_count":
            X = _entity_frame(s, pk, aux[keep].reset_index(drop=True), n_q)
            sel, _ = gfm_rule.select(s, pk, count_model.predict(dec["count_models"], dec["count_temp"], X))
        elif dec["rule"] == "gfm_indep":
            sel, _ = gfm_rule.select(s, pk, None)
        else:
            th = dec["thresholds"][c]
            sel = two_threshold.select(s, pk, th["tau1"], th["tau2"])
        # truth
        g = gt[gt["source1_entity_id"].isin(set(q_ids.tolist()))]
        g = g.assign(m=g["matched_entity_ids"].str.split(",")).explode("m")
        g = g[g["m"].fillna("") != ""]
        qpos = pd.Series(np.arange(n_q), index=q_ids)
        rpos = pd.Series(np.arange(len(r_ids)), index=r_ids)
        tq = qpos.reindex(g["source1_entity_id"]).to_numpy(np.int64)
        tr = rpos.reindex(g["m"]).to_numpy(np.int64)
        n_r = np.int64(len(r_ids))
        tkey = tq * n_r + tr
        cand_all = d["qi"].to_numpy(np.int64) * n_r + d["ri"].to_numpy(np.int64)
        retrieved = np.isin(tkey, cand_all)
        n_true = np.bincount(tq, minlength=n_q)
        n_ret = np.bincount(tq[retrieved], minlength=n_q)
        n_pred = np.bincount(s[sel], minlength=n_q)
        tp = np.bincount(s[sel & (yk == 1)], minlength=n_q)
        f_act = entity_f(n_pred, tp, n_true)
        f_nofp = entity_f(tp, tp, n_true)
        f_ceil = entity_f(n_ret, n_ret, n_true)
        crowd = np.zeros(n_q)
        first = np.full(n_q, -1)
        first[d["qi"].to_numpy()[::-1]] = np.arange(len(d))[::-1]
        has = first >= 0
        crowd[has] = aux["q_name_city_crowd"].to_numpy()[first[has]]
        blk_nonlatin = np.bincount(tq[~retrieved], weights=r_nonlatin[tr[~retrieved]], minlength=n_q)
        blk_empty = np.bincount(tq[~retrieved], weights=r_empty[tr[~retrieved]].astype(float), minlength=n_q)
        rows.append(pd.DataFrame({"country": c, "n_true": n_true, "f": f_act, "loss_fp": f_nofp - f_act,
                                  "loss_model_fn": f_ceil - f_nofp, "loss_block": 1 - f_ceil,
                                  "crowd": crowd, "blocked_nonlatin": blk_nonlatin > 0, "blocked_empty_addr": blk_empty > 0}))
        # examples
        fp_idx = np.flatnonzero(sel & (yk == 0))
        fn_keys = tkey[retrieved & ~np.isin(tkey, s[sel].astype(np.int64) * n_r + rk[sel])]
        miss_keys = tkey[~retrieved]

        def show(title, qi_arr, ri_arr):
            lines.append(f"\n### {c}: {title}\n")
            if len(qi_arr) == 0:
                return
            pick = rng.choice(len(qi_arr), min(args.examples, len(qi_arr)), replace=False)
            qa = qt.take(qi_arr[pick]).to_pydict()
            ra = rt.take(ri_arr[pick]).to_pydict()
            for i in range(len(pick)):
                lines.append(f"- `{qa['name_a'][i]} | {qa['addr_a'][i]}`  vs  `{ra['name_a'][i]} | {ra['addr_a'][i]}`")

        show("false links (selected, not a match)", s[fp_idx], rk[fp_idx])
        show("true matches retrieved but not selected", (fn_keys // n_r).astype(np.int64), (fn_keys % n_r).astype(np.int64))
        show("true matches never retrieved", (miss_keys // n_r).astype(np.int64), (miss_keys % n_r).astype(np.int64))
        del qt, rt
    df = pd.concat(rows, ignore_index=True)
    total = 1 - df["f"].mean()
    dec_tab = {"macro_f05": round(df["f"].mean(), 5), "total_loss": round(total, 5),
               "loss_false_links": round(df["loss_fp"].mean(), 5), "loss_model_misses": round(df["loss_model_fn"].mean(), 5),
               "loss_blocking": round(df["loss_block"].mean(), 5)}
    head = ["\n## Loss decomposition (share of 1 - macro F0.5)\n", "| Component | Loss | Share |", "|---|---|---|"]
    for k in ("loss_false_links", "loss_model_misses", "loss_blocking"):
        head.append(f"| {k.replace('loss_', '').replace('_', ' ')} | {dec_tab[k]:.5f} | {dec_tab[k] / total:.1%} |")
    head += ["\n## By slice (mean loss per entity x share of entities = contribution)\n",
             "| Slice | Entities | FP loss | Model-miss loss | Blocking loss |", "|---|---|---|---|---|"]
    slices = {"all": np.ones(len(df), bool), "US": df["country"].eq("US").to_numpy(), "India": df["country"].eq("India").to_numpy(),
              "true singletons": (df["n_true"] == 0).to_numpy(), "1 true match": (df["n_true"] == 1).to_numpy(),
              "2-3 true matches": df["n_true"].between(2, 3).to_numpy(), "4+ true matches": (df["n_true"] >= 4).to_numpy(),
              "name-city crowd >= 3": (df["crowd"] >= 3).to_numpy(),
              "has a blocked non-Latin record": df["blocked_nonlatin"].to_numpy(),
              "has a blocked empty-address record": df["blocked_empty_addr"].to_numpy()}
    n = len(df)
    for name, m in slices.items():
        head.append(f"| {name} | {m.mean():.1%} | {df['loss_fp'][m].sum() / n:.5f} | "
                    f"{df['loss_model_fn'][m].sum() / n:.5f} | {df['loss_block'][m].sum() / n:.5f} |")
    lines = lines[:1] + head + lines[1:]
    out = os.path.join(cfg["paths"]["runs"], "error_analysis.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    json.dump(dec_tab, open(os.path.join(cfg["paths"]["runs"], "loss_decomposition.json"), "w"), indent=1)
    print("\n".join(head))


if __name__ == "__main__":
    main()
