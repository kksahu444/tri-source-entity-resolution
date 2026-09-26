"""Measure whether second-hop retrieval (expanding from confident matches) can recover blocking misses.

For each dev S1: anchors = its pruned candidates with stage-A OOF p >= 0.8. A true record missed by
blocking/pruning is "recoverable by key K" if it shares key K with one of that S1's anchors. The cost of
key K is the number of new (not already candidate) records it would add per S1.

python -m src.analysis_second_hop --config config_v2.yaml  ->  runs/second_hop.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .blocking.union import block_dir
from .stack import pred_dir
from .utils import load_config


def keys_for(t: pa.Table) -> dict:
    """Exact-key codes per record (Arrow compute, low memory); -1 where a key part is empty."""
    core, addr, city, house = t["core"], t["addr_core"], t["city"], t["house"]
    first = pc.list_element(pc.split_pattern(core, " "), 0)
    join = lambda a, b: pc.binary_join_element_wise(a, b, "|")
    empty = lambda *cols: np.logical_or.reduce([pc.equal(x, "").to_numpy(zero_copy_only=False) for x in cols])
    out = {"name": (core, empty(core)), "name+city": (join(core, city), empty(core, city)),
           "name+house": (join(core, house), empty(core, house)), "addr": (addr, empty(addr)),
           "house+name1": (join(house, first), empty(house, core))}
    codes = {}
    for k, (arr, is_empty) in out.items():
        idx = pc.dictionary_encode(arr).combine_chunks().indices.to_numpy(zero_copy_only=False).astype(np.int64)
        codes[k] = np.where(is_empty, -1, idx)
    return codes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v2.yaml")
    ap.add_argument("--anchor_p", type=float, default=0.8)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    gt = pd.read_csv(os.path.join(cfg["paths"]["dataset"], "train", "train_ground_truth.tsv"), sep="\t", dtype=str,
                     keep_default_na=False)
    report = {}
    for c in ("India", "US"):
        q = pq.read_table(os.path.join(block_dir(cfg), f"dev_{c}_q.parquet"), columns=["entity_id"]).to_pandas()
        rt = pq.read_table(os.path.join(block_dir(cfg), f"dev_{c}_r.parquet"),
                           columns=["entity_id", "core", "addr_core", "city", "house"])
        r = pd.DataFrame({"entity_id": rt["entity_id"].to_numpy(zero_copy_only=False)})
        pred = pd.read_parquet(os.path.join(pred_dir(cfg), f"dev_{c}_stageA.parquet"))
        g = gt[gt["source1_entity_id"].isin(set(q["entity_id"]))]
        g = g.assign(m=g["matched_entity_ids"].str.split(",")).explode("m")
        g = g[g["m"].fillna("") != ""]
        qpos = pd.Series(np.arange(len(q)), index=q["entity_id"])
        rpos = pd.Series(np.arange(len(r)), index=r["entity_id"])
        tq = qpos.reindex(g["source1_entity_id"]).to_numpy(np.int64)
        tr = rpos.reindex(g["m"]).to_numpy(np.int64)
        n_r = np.int64(len(r))
        cand = pred["qi"].to_numpy(np.int64) * n_r + pred["ri"].to_numpy(np.int64)
        tkey = tq * n_r + tr
        missed = ~np.isin(tkey, cand)
        anc = pred[pred["p"].to_numpy() >= args.anchor_p]
        info = {"true_pairs": int(len(tkey)), "missed": int(missed.sum()),
                "missed_share": round(float(missed.mean()), 5), "anchors_per_s1": round(len(anc) / len(q), 3)}
        codes = keys_for(rt)
        del rt
        mq, mr = tq[missed], tr[missed]
        for k, code in codes.items():
            # anchor keys per S1
            ak = pd.DataFrame({"q": anc["qi"].to_numpy(np.int64), "k": code[anc["ri"].to_numpy()]})
            ak = ak[ak["k"] >= 0].drop_duplicates()
            akey = set((ak["q"].to_numpy() * (code.max() + 2) + ak["k"].to_numpy()).tolist())
            mk = code[mr]
            rec = np.array([(mk[i] >= 0) and (int(mq[i] * (code.max() + 2) + mk[i]) in akey) for i in range(len(mq))])
            # cost: records sharing an anchor key, not already candidates of that S1
            size = np.bincount(code[code >= 0])
            added = ak.assign(n=size[ak["k"].to_numpy()]).groupby("q")["n"].sum()
            info[f"recoverable_by_{k}"] = round(float(rec.mean()) if len(rec) else 0.0, 4)
            info[f"pool_records_per_s1_{k}"] = round(float(added.sum() / len(q)), 2)
        report[c] = info
        print(c, json.dumps(info), flush=True)
        del r, pred, codes
    json.dump(report, open(os.path.join(cfg["paths"]["runs"], "second_hop.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
