"""Cross-encoder training pairs from train S1 outside the dev universe (role 'out').

The dev universe (dev.city_group_share of city groups) is where the stack's OOF lives, so a cross-encoder
trained on it would need cross-fitting. Training on the other ~70% of city groups gives one model that
scores dev and test alike, with no fold bookkeeping. Candidates come from the real blocking
(block_country), so the negatives are the same kind the stack sees; pairs are cut at union_rank <= --rank.
All true pairs of the sampled S1 are kept, including the ones blocking misses.

python -m src.ce_pairs --config config_v3.yaml --per_country 25000  ->  artifacts/ce/ce_train.parquet
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import io_utils
from .blocking.union import BLOCK_COLS, block_country
from .cv import build_dev_universe
from .normalize import norm_path
from .utils import load_config

TEXT_COLS = ["entity_id", "name_a", "addr_a"]


def pair_text(df: pd.DataFrame) -> pd.Series:
    return df["name_a"] + " | " + df["addr_a"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v3.yaml")
    ap.add_argument("--per_country", type=int, default=25000)
    ap.add_argument("--rank", type=int, default=15)
    ap.add_argument("--val_share", type=float, default=0.1)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    dev = build_dev_universe(cfg)
    out = dev[dev["role"] == "out"]
    gt = io_utils.load_ground_truth(cfg["paths"]["dataset"])
    rng = np.random.default_rng(cfg["seed"] + 7)
    parts = []
    for country, grp in out.groupby("country"):
        pick = grp.sample(n=min(args.per_country, len(grp)), random_state=cfg["seed"])
        keys = pick["city_key"].unique()
        val_keys = set(keys[rng.random(len(keys)) < args.val_share])
        s1 = pq.read_table(norm_path(cfg, "train", 1), columns=BLOCK_COLS + ["name_a", "addr_a"],
                           filters=[("entity_id", "in", pick["entity_id"].tolist())]).to_pandas()
        r = pa.concat_tables([pq.read_table(norm_path(cfg, "train", k), filters=[("country", "=", country)],
                                            columns=BLOCK_COLS + ["name_a", "addr_a"]) for k in (2, 3)]).to_pandas()
        pairs, info = block_country(cfg, s1, r)
        pairs = pairs[pairs["union_rank"] <= args.rank]
        cand = pd.DataFrame({"s1_id": s1["entity_id"].to_numpy()[pairs["qi"]],
                             "cand_id": r["entity_id"].to_numpy()[pairs["ri"]],
                             "union_rank": pairs["union_rank"].to_numpy()})
        g = gt[gt["s1_id"].isin(set(s1["entity_id"]))]
        cand = cand.merge(g.assign(label=np.int8(1)), on=["s1_id", "cand_id"], how="outer")
        cand["label"] = cand["label"].fillna(0).astype(np.int8)
        cand["union_rank"] = cand["union_rank"].fillna(99).astype(np.int16)
        ta = pd.Series(pair_text(s1).to_numpy(), index=s1["entity_id"])
        tb = pd.Series(pair_text(r).to_numpy(), index=r["entity_id"])
        cand["text_a"] = ta.reindex(cand["s1_id"]).to_numpy()
        cand["text_b"] = tb.reindex(cand["cand_id"]).to_numpy()
        ck = pick.set_index("entity_id")["city_key"]
        cand["val"] = ck.reindex(cand["s1_id"]).isin(val_keys).to_numpy().astype(np.int8)
        cand["country"] = country
        print(country, f"{len(s1):,} S1", f"{len(cand):,} pairs", f"pos {cand['label'].mean():.3f}",
              f"true pairs outside the cut {int((cand['union_rank'] == 99).sum()):,}", flush=True)
        parts.append(cand)
        del pairs, r
    ce_dir = os.path.join(cfg["paths"]["artifacts"], "ce")
    os.makedirs(ce_dir, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(ce_dir, "ce_train.parquet"), index=False)


if __name__ == "__main__":
    main()
