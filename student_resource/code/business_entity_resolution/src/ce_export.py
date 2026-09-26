"""Work order T6: export pairs for the cross-encoder (Kaggle GPU).

Dev: band pairs (stage-B OOF p in [lo, hi]) plus a sample of confident pairs, with view-A texts,
labels and 3 folds grouped by (country, sorted core name of the S1): a text model can memorise names,
so city grouping alone is not enough. Test: band pairs of the test predictions.
Text: ("name: {n1} | address: {a1}", "name: {n2} | address: {a2}"), view A, no country.

python -m src.ce_export --config config_v2.yaml  ->  artifacts/ce/{dev,test}_pairs.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .blocking.union import block_dir
from .featurize import countries
from .stack import pred_dir
from .utils import load_config


def _text(t: pa.Table, idx: np.ndarray) -> np.ndarray:
    sub = t.take(pa.array(idx)).to_pydict()
    return np.array([f"name: {n} | address: {a}" for n, a in zip(sub["name_a"], sub["addr_a"])], dtype=object)


def _fold(country: str, cores: list[str]) -> np.ndarray:
    key = [country + "|" + " ".join(sorted(set(c.split()))) for c in cores]
    return np.array([int(hashlib.md5(k.encode()).hexdigest()[:8], 16) % 3 for k in key], np.int8)


def export(cfg, split, lo, hi, confident_share, seed):
    rng = np.random.default_rng(seed)
    frames = []
    for c in countries(cfg, split):
        d = pd.read_parquet(os.path.join(pred_dir(cfg), f"{split}_{c}.parquet"))
        p = d["p"].to_numpy()
        band = (p >= lo) & (p <= hi)
        keep = band | ((rng.random(len(p)) < confident_share) if split == "dev" else False)
        d = d[keep].reset_index(drop=True)
        qt = pq.read_table(os.path.join(block_dir(cfg), f"{split}_{c}_q.parquet"), columns=["entity_id", "name_a", "addr_a", "core"])
        rt = pq.read_table(os.path.join(block_dir(cfg), f"{split}_{c}_r.parquet"), columns=["entity_id", "name_a", "addr_a"])
        qi, ri = d["qi"].to_numpy(), d["ri"].to_numpy()
        out = pd.DataFrame({"country": c, "qi": qi.astype(np.int32), "ri": ri.astype(np.int32),
                            "s1_id": qt["entity_id"].take(pa.array(qi)).to_numpy(zero_copy_only=False),
                            "cand_id": rt["entity_id"].take(pa.array(ri)).to_numpy(zero_copy_only=False),
                            "text_a": _text(qt, qi), "text_b": _text(rt, ri), "p_stage": p[keep].astype(np.float32),
                            "in_band": band[keep]})
        if split == "dev":
            out["y"] = d["y"].to_numpy().astype(np.int8)
            out["ce_fold"] = _fold(c, qt["core"].take(pa.array(qi)).to_pylist())
        frames.append(out)
        print(split, c, f"{len(out):,} pairs ({band.sum():,} in band)", flush=True)
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v2.yaml")
    ap.add_argument("--lo", type=float, default=0.005)
    ap.add_argument("--hi", type=float, default=0.995)
    ap.add_argument("--confident_share", type=float, default=0.03)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = os.path.join(cfg["paths"]["artifacts"], "ce")
    os.makedirs(out, exist_ok=True)
    info = {"band": [args.lo, args.hi], "confident_share": args.confident_share}
    for split in ("dev", "test"):
        df = export(cfg, split, args.lo, args.hi, args.confident_share, cfg["seed"])
        df.to_parquet(os.path.join(out, f"{split}_pairs.parquet"), index=False, compression="zstd")
        info[split] = {"pairs": len(df), "in_band": int(df["in_band"].sum())}
        if split == "dev":
            info["dev"]["pos_rate"] = round(float(df["y"].mean()), 4)
            info["dev"]["fold_sizes"] = df["ce_fold"].value_counts().sort_index().tolist()
    json.dump(info, open(os.path.join(out, "export_info.json"), "w"), indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
