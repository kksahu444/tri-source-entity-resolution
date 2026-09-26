"""Export the pruned pairs of a run for the cross-encoder kernel (kaggle/ce/train_ce.py), and join its scores back.

export: artifacts/feat/{split}_{country}.parquet (qi, ri) -> artifacts/ce/score_{split}.parquet (s1_id, cand_id)
        plus texts_{split}.parquet (entity_id, "name | address" in view A), one row per record.
join:   artifacts/ce/ce_{split}.parquet (s1_id, cand_id, ce_p) -> artifacts/ce/ce_{split}_{country}.parquet
        with qi, ri, ce_p in the row order of the feature file, ready to add as a column.

python -m src.ce_export --config config_v3.yaml --split dev [--join]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .blocking.union import block_dir
from .ce_pairs import pair_text
from .featurize import countries, feat_dir
from .normalize import norm_path
from .utils import load_config


def ids(cfg, split, c):
    d = block_dir(cfg)
    q = pq.read_table(os.path.join(d, f"{split}_{c}_q.parquet"), columns=["entity_id"]).column(0).to_numpy()
    r = pq.read_table(os.path.join(d, f"{split}_{c}_r.parquet"), columns=["entity_id"]).column(0).to_numpy()
    return q, r


def export(cfg, split, ce_dir):
    pairs, need = [], set()
    for c in countries(cfg, split):
        f = pd.read_parquet(os.path.join(feat_dir(cfg), f"{split}_{c}.parquet"), columns=["qi", "ri"])
        q, r = ids(cfg, split, c)
        p = pd.DataFrame({"s1_id": q[f["qi"].to_numpy()], "cand_id": r[f["ri"].to_numpy()]})
        need.update(p["s1_id"].unique())
        need.update(p["cand_id"].unique())
        pairs.append(p)
        print(split, c, f"{len(p):,} pairs", flush=True)
    pairs = pd.concat(pairs, ignore_index=True)
    src = "train" if split == "dev" else split           # the dev universe is cut from train
    texts = pa.concat_tables([pq.read_table(norm_path(cfg, src, k), columns=["entity_id", "name_a", "addr_a"],
                                            filters=[("entity_id", "in", list(need))]) for k in (1, 2, 3)]).to_pandas()
    assert len(texts) == len(need), (len(texts), len(need))
    pd.DataFrame({"entity_id": texts["entity_id"], "text": pair_text(texts)}).to_parquet(
        os.path.join(ce_dir, f"texts_{split}.parquet"), index=False, compression="zstd")
    pairs.to_parquet(os.path.join(ce_dir, f"score_{split}.parquet"), index=False, compression="zstd")
    print(split, f"{len(pairs):,} pairs, {len(texts):,} texts", flush=True)


def join(cfg, split, ce_dir):
    ce = pd.read_parquet(os.path.join(ce_dir, f"ce_{split}.parquet"))
    ce = ce.set_index(ce["s1_id"] + "\t" + ce["cand_id"])["ce_p"]
    for c in countries(cfg, split):
        f = pd.read_parquet(os.path.join(feat_dir(cfg), f"{split}_{c}.parquet"), columns=["qi", "ri"])
        q, r = ids(cfg, split, c)
        key = pd.Series(q[f["qi"].to_numpy()]) + "\t" + pd.Series(r[f["ri"].to_numpy()])
        p = ce.reindex(key).to_numpy(np.float32)
        missing = int(np.isnan(p).sum())
        assert missing == 0, f"{split} {c}: {missing} pairs have no ce_p; the kernel scored a different run"
        f.assign(ce_p=p).to_parquet(os.path.join(ce_dir, f"ce_{split}_{c}.parquet"), index=False)
        print(split, c, f"{len(f):,} pairs joined", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v3.yaml")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--join", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ce_dir = os.path.join(cfg["paths"]["artifacts"], "ce")
    os.makedirs(ce_dir, exist_ok=True)
    (join if args.join else export)(cfg, args.split, ce_dir)


if __name__ == "__main__":
    main()
