"""Measure how C7's blank-address recall depends on the number of competing S1 queries.

Reverse C7 (each empty-address record -> its top-k S1s) spreads a fixed k over however many S1s are
in the run, so its recall falls as the pool grows; the forward pass (each S1 -> its top-k empty-address
records) searches the same record pool whatever the S1 count. Recall is measured on the same 3% of dev
queries while the S1 pool grows from that sample to the dev universe to all train S1 (test-sized).

python -m src.analysis_c7_scale --config config_v3.yaml  ->  runs/c7_scale.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import io_utils
from .blocking.tfidf import TfidfSpace, topk
from .blocking.union import BLOCK_COLS
from .cv import build_dev_universe
from .normalize import norm_path
from .utils import load_config


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v3.yaml")
    ap.add_argument("--sample", type=float, default=0.03)
    ap.add_argument("--k_rev", type=int, default=10)
    ap.add_argument("--k_fwd", type=int, default=5)
    ap.add_argument("--min_cos", type=float, default=0.3)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    dev = build_dev_universe(cfg)
    gt = io_utils.load_ground_truth(cfg["paths"]["dataset"])
    s1_all = pq.read_table(norm_path(cfg, "train", 1), columns=BLOCK_COLS + ["country"]).to_pandas()
    devq = set(dev.loc[dev["role"] == "query", "entity_id"])
    rng = np.random.default_rng(cfg["seed"])
    report = {}
    for country in sorted(s1_all["country"].unique()):
        s1 = s1_all[s1_all["country"] == country].reset_index(drop=True)
        r = pa.concat_tables([pq.read_table(norm_path(cfg, "train", k), filters=[("country", "=", country)],
                                            columns=BLOCK_COLS) for k in (2, 3)]).to_pandas()
        emp = np.flatnonzero(r["addr_core"].to_numpy() == "")
        rn, _ = (space := TfidfSpace()).fit_transform(r["core"].tolist(), r["addr_core"].tolist())
        qn_all, _ = space.transform(s1["core"].tolist(), s1["addr_core"].tolist())
        rpos = pd.Series(np.arange(len(r)), index=r["entity_id"])
        is_dev = s1["entity_id"].isin(devq).to_numpy()
        sample = is_dev & (rng.random(len(s1)) < args.sample)
        g = gt[gt["s1_id"].isin(set(s1.loc[sample, "entity_id"]))]
        g = g[g["cand_id"].isin(set(r["entity_id"].iloc[emp]))]
        rows = {"blank_records": int(len(emp)), "eval_s1": int(sample.sum()), "blank_true_pairs": int(len(g))}
        n_r = np.int64(len(r))
        for name, pool in (("sample", sample), ("dev_universe", is_dev), ("all_train_s1", np.ones(len(s1), bool))):
            idx = np.flatnonzero(pool)
            qn = qn_all[idx]
            qpos = pd.Series(np.arange(len(idx)), index=s1["entity_id"].iloc[idx])
            tk = qpos.reindex(g["s1_id"]).to_numpy(np.int64) * n_r + rpos.reindex(g["cand_id"]).to_numpy(np.int64)
            eq, ek, _ = topk(rn[emp], qn.T.tocsr(), args.k_rev, threshold=args.min_cos, chunk=100_000)
            rev = ek.astype(np.int64) * n_r + emp[eq]
            fq, fr, _ = topk(qn, rn[emp].T.tocsr(), args.k_fwd, threshold=args.min_cos, chunk=100_000)
            fwd = fq.astype(np.int64) * n_r + emp[fr]
            rows[name] = {"n_s1": int(len(idx)),
                          "reverse_pc": round(float(np.isin(tk, rev).mean()), 4), "reverse_per_s1": round(len(rev) / len(idx), 2),
                          "forward_pc": round(float(np.isin(tk, fwd).mean()), 4), "forward_per_s1": round(len(fwd) / len(idx), 2),
                          "both_pc": round(float(np.isin(tk, np.union1d(rev, fwd)).mean()), 4)}
            print(country, name, rows[name], flush=True)
        report[country] = rows
        del rn, qn_all, r, s1
    json.dump(report, open(os.path.join(cfg["paths"]["runs"], "c7_scale.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
