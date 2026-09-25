"""Stage `features`: cheap features -> pruner (D3) -> full features (D4) on the pruned pairs."""
from __future__ import annotations

import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

from . import io_utils, prune
from .blocking.union import block_dir
from .cv import build_dev_universe
from .features import cheap_features, full_features
from .utils import Timer, get_logger

log = get_logger()


def feat_dir(cfg):
    d = os.path.join(cfg["paths"]["artifacts"], "feat")
    os.makedirs(d, exist_ok=True)
    return d


def countries(cfg, split):
    return sorted(os.path.basename(p).split("_")[1] for p in glob.glob(os.path.join(block_dir(cfg), f"{split}_*_pairs.parquet")))


FEAT_COLS = ["entity_id", "core", "name_b", "legal", "alt", "addr_b", "addr_core", "house", "house_sfx", "street",
             "city", "region", "unit", "landmark", "nums", "n_parts", "f_tag", "f_lead", "f_bang", "f_digits",
             "f_nonlatin", "src"]


def load_block(cfg, split, country):
    """Arrow-backed strings: only the values of candidate pairs are materialised as Python objects."""
    d = block_dir(cfg)
    q = pd.read_parquet(os.path.join(d, f"{split}_{country}_q.parquet"), columns=FEAT_COLS)
    r = pd.read_parquet(os.path.join(d, f"{split}_{country}_r.parquet"), columns=FEAT_COLS, dtype_backend="pyarrow")
    pairs = pd.read_parquet(os.path.join(d, f"{split}_{country}_pairs.parquet"))
    return q, r, pairs


def leak_share(q, r, pairs, gt, dev, fold_of_pair) -> float:
    """Section 8 leak check: share of pairs whose record's true S1 is a query in another fold."""
    owner = gt.set_index("cand_id")["s1_id"]
    rec_owner = owner.reindex(np.asarray(r["entity_id"].to_numpy(), dtype=object)[pairs["ri"].to_numpy()])
    q_fold = dev["fold"].reindex(rec_owner.to_numpy()).to_numpy()
    return float(np.mean((q_fold >= 0) & (q_fold != fold_of_pair)))


def labels(q, r, pairs, gt):
    g = gt[gt["s1_id"].isin(set(q["entity_id"]))]
    qpos = pd.Series(np.arange(len(q)), index=q["entity_id"])
    rpos = pd.Series(np.arange(len(r)), index=r["entity_id"])
    n_r = np.int64(len(r))
    tk = qpos.reindex(g["s1_id"]).to_numpy(np.int64) * n_r + rpos.reindex(g["cand_id"]).to_numpy(np.int64)
    pk = pairs["qi"].to_numpy(np.int64) * n_r + pairs["ri"].to_numpy(np.int64)
    return np.isin(pk, tk).astype(np.int8), len(tk)


def run(cfg: dict, split: str = "dev") -> dict:
    t = Timer()
    fd = feat_dir(cfg)
    pc = cfg["prune"]
    model_path = os.path.join(fd, "pruner.pkl")
    res = {"stage": "features"}
    info = {}
    if split == "dev":
        gt = io_utils.load_ground_truth(cfg["paths"]["dataset"])
        dev = build_dev_universe(cfg).set_index("entity_id")
        cheap, ys, folds, qis, n_true = {}, {}, {}, {}, {}
        for c in countries(cfg, split):
            q, r, pairs = load_block(cfg, split, c)
            cheap[c] = cheap_features(q, r, pairs)
            ys[c], n_true[c] = labels(q, r, pairs, gt)
            folds[c] = dev["fold"].reindex(q["entity_id"]).to_numpy()[pairs["qi"].to_numpy()]
            qis[c] = pairs["qi"].to_numpy()
            info[c] = {"leak_share": round(leak_share(q, r, pairs, gt, dev, folds[c]), 5)}
            log.info("cheap features %s: %s pairs (%.0fs)", c, f"{len(pairs):,}", t())
        cs = list(cheap)
        X = pd.concat([cheap[c] for c in cs], ignore_index=True)
        cols = list(X.columns)
        y = np.concatenate([ys[c] for c in cs])
        fold = np.concatenate([folds[c] for c in cs])
        oof, models = prune.train_oof(X.to_numpy(np.float32), y, fold, cfg["seed"])
        del X
        off, ns = 0, {}
        for c in cs:
            m = len(ys[c])
            s = oof[off:off + m]
            off += m
            n, pcu, curve = prune.choose_n(qis[c], s, ys[c], n_true[c], pc["pc_tolerance"])
            ns[c] = n
            info[c].update({"N": n, "PC_union": round(pcu, 5), "PC_at_N": curve})
            cheap[c]["prune_score"] = s
        ns["France"] = max(ns.values())
        pickle.dump({"models": models, "cols": cols, "N": ns}, open(model_path, "wb"))
        log.info("pruner: %s (%.0fs)", json.dumps(info), t())
    else:
        pr = pickle.load(open(model_path, "rb"))
        ns, cols = pr["N"], pr["cols"]
        cheap = {}
    for c in countries(cfg, split):
        q, r, pairs = load_block(cfg, split, c)
        if split == "dev":
            ch = cheap.pop(c)
        else:
            ch = cheap_features(q, r, pairs)
            ch["prune_score"] = prune.predict(pr["models"], ch[cols].to_numpy(np.float32))
        n = ns.get(c, ns["France"])
        keep = prune.keep_mask(pairs["qi"].to_numpy(), pairs["ri"].to_numpy(), ch["prune_score"].to_numpy(),
                               n, pc["reverse_keep"])
        pairs, ch = pairs[keep].reset_index(drop=True), ch[keep].reset_index(drop=True)
        f = full_features(q, r, pairs, ch, support=cfg.get("features", {}).get("support", False))
        f.insert(0, "qi", pairs["qi"].to_numpy())
        f.insert(1, "ri", pairs["ri"].to_numpy())
        if split == "dev":
            f["y"] = ys[c][keep]
            f["fold"] = folds[c][keep].astype(np.int8)
            info[c]["PC_pruned"] = round(float(f["y"].sum() / n_true[c]), 5)
        info.setdefault(c, {}).update({"pairs_pruned": len(f), "per_s1": round(len(f) / len(q), 2), "N": n})
        f.to_parquet(os.path.join(fd, f"{split}_{c}.parquet"), index=False)
        log.info("full features %s %s: %s (%.0fs)", split, c, json.dumps(info[c]), t())
        del q, r, pairs, ch, f
    res.update({"runtime_s": t(), "notes": json.dumps(info)})
    json.dump(info, open(os.path.join(cfg["paths"]["runs"], f"features_{split}.json"), "w"), indent=1)
    return res
