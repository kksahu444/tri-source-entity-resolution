"""Stage `block` (D2): union of C1 (TF-IDF), C2 (name crowd) and C5 (keys) per country.

Outputs per (split, country) in artifacts/block/:
  {split}_{country}_q.parquet      query S1 rows (normalised view), row order = qi
  {split}_{country}_r.parquet      S2 u S3 rows of the same country, row order = ri
  {split}_{country}_pairs.parquet  union pairs with every channel's score/rank
The dev split reports pair completeness per channel and for the union (spec D2).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import io_utils
from ..cv import build_dev_universe
from ..normalize import norm_path
from ..utils import Timer, get_logger
from .crowd import crowd_channel, key_channel
from .tfidf import TfidfSpace, combined, pair_cos, topk

log = get_logger()
REC_COLS = None  # all columns


def block_dir(cfg):
    d = os.path.join(cfg["paths"]["artifacts"], "block")
    os.makedirs(d, exist_ok=True)
    return d


def read_norm(cfg, split, k, country=None):
    filters = [("country", "=", country)] if country else None
    return pq.read_table(norm_path(cfg, split, k), filters=filters).to_pandas()


def sorted_key(s: pd.Series) -> pd.Series:
    return s.map(lambda x: " ".join(sorted(set(x.split()))))


def first_alpha(s: pd.Series, min_len=3) -> pd.Series:
    def f(x):
        for t in x.split():
            if len(t) >= min_len and not t[0].isdigit():
                return t
        return ""
    return s.map(f)


def house_key(df: pd.DataFrame) -> pd.Series:
    st = first_alpha(df["street"])
    ct = first_alpha(df["city"], 2)
    tail = st.where(st != "", ct)
    return (df["house"] + "|" + tail).where((df["house"] != "") & (tail != ""), "")


def name_city_key(df: pd.DataFrame) -> pd.Series:
    ft = df["core"].str.split(n=1).str[0].fillna("")
    return (ft + "|" + df["city"]).where((ft != "") & (df["city"] != ""), "")


def _rank_within(qi: np.ndarray, score: np.ndarray) -> np.ndarray:
    order = np.lexsort((-score, qi))
    rank = np.empty(len(qi), np.int16)
    q_sorted = qi[order]
    starts = np.r_[0, np.flatnonzero(np.diff(q_sorted)) + 1]
    pos = np.arange(len(qi)) - np.repeat(starts, np.diff(np.r_[starts, len(qi)]))
    rank[order] = (pos + 1).clip(max=32000)
    return rank


def _topn_per_q(qi, ri, score, n):
    r = _rank_within(qi, score)
    keep = r <= n
    return qi[keep], ri[keep], score[keep]


def block_country(cfg, q: pd.DataFrame, r: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    bc, t = cfg["blocking"], Timer()
    info = {}
    space = TfidfSpace().fit(r["core"].tolist(), r["addr_core"].tolist())
    rn, ra = space.transform(r["core"].tolist(), r["addr_core"].tolist())
    qn, qa = space.transform(q["core"].tolist(), q["addr_core"].tolist())
    rc_t = combined(rn, ra).T.tocsr()
    qc = combined(qn, qa)
    info["t_vectorize"] = t()
    c1q, c1r, c1s = topk(qc, rc_t, bc["k_tfidf"], chunk=bc["query_chunk"])
    del rc_t, qc
    info["t_c1"] = t()
    c2q, c2r, c2size = crowd_channel(sorted_key(q["core"]), sorted_key(r["core"]), qa, ra, bc["crowd_cap"])
    info["t_c2"] = t()
    keys = {}
    for name, fn in (("c5a", house_key), ("c5b", name_city_key)):
        kq, kr, skipped = key_channel(fn(q), fn(r), bc["key_block_cap"])
        s = 0.5 * (pair_cos(qn, rn, kq, kr) + pair_cos(qa, ra, kq, kr))
        keys[name] = _topn_per_q(kq, kr, s, bc.get("key_keep", 10))
        info[f"{name}_skipped_q"] = skipped
    info["t_c5"] = t()

    n_r = np.int64(len(r))
    chans = {"c1": (c1q, c1r), "c2": (c2q, c2r), **{k: v[:2] for k, v in keys.items()}}
    all_keys = np.unique(np.concatenate([a.astype(np.int64) * n_r + b for a, b in chans.values()]))
    pairs = pd.DataFrame({"qi": (all_keys // n_r).astype(np.int32), "ri": (all_keys % n_r).astype(np.int32)})
    for name, (a, b) in chans.items():
        pos = np.searchsorted(all_keys, a.astype(np.int64) * n_r + b)
        pairs[f"in_{name}"] = np.zeros(len(pairs), np.int8)
        pairs.loc[pos, f"in_{name}"] = 1
        if name == "c1":
            rank = _rank_within(a, c1s)
            pairs["c1_rank"] = np.int16(bc["k_tfidf"] + 1)
            pairs.loc[pos, "c1_rank"] = rank
        if name == "c2":
            pairs["crowd_size"] = np.int32(0)
            pairs.loc[pos, "crowd_size"] = c2size
    qi, ri = pairs["qi"].to_numpy(), pairs["ri"].to_numpy()
    pairs["cos_name"] = pair_cos(qn, rn, qi, ri)
    pairs["cos_addr"] = pair_cos(qa, ra, qi, ri)
    pairs["c1_score"] = 0.5 * (pairs["cos_name"] + pairs["cos_addr"])
    pairs["union_rank"] = _rank_within(qi, pairs["c1_score"].to_numpy())
    pairs["src"] = r["src"].to_numpy()[ri]
    info["t_union"] = t()
    info["pairs"] = len(pairs)
    return pairs, info


def completeness(pairs: pd.DataFrame, q: pd.DataFrame, r: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Pair completeness per channel and for the union, by source and match-count bucket."""
    qpos = pd.Series(np.arange(len(q)), index=q["entity_id"])
    rpos = pd.Series(np.arange(len(r)), index=r["entity_id"])
    g = gt[gt["s1_id"].isin(qpos.index)]
    tq = qpos.reindex(g["s1_id"]).to_numpy()
    tr = rpos.reindex(g["cand_id"]).to_numpy()
    assert not np.isnan(tr.astype(float)).any(), "true records missing from the index"
    n_r = np.int64(len(r))
    true_keys = tq.astype(np.int64) * n_r + tr.astype(np.int64)
    pk = pairs["qi"].to_numpy(np.int64) * n_r + pairs["ri"].to_numpy(np.int64)
    out = {"n_true": len(true_keys)}
    found = np.isin(true_keys, pk)
    out["PC_union"] = round(float(found.mean()), 5)
    for ch in [c for c in pairs.columns if c.startswith("in_")]:
        out[f"PC_{ch[3:]}"] = round(float(np.isin(true_keys, pk[pairs[ch].to_numpy() == 1]).mean()), 5)
    src = r["src"].to_numpy()[tr]
    for s in (2, 3):
        out[f"PC_union_S{s}"] = round(float(found[src == s].mean()), 5)
    n_true = g.groupby("s1_id").size()
    b = n_true.reindex(g["s1_id"]).clip(upper=3).to_numpy()
    for v in (1, 2, 3):
        out[f"PC_union_bucket{v}{'+' if v == 3 else ''}"] = round(float(found[b == v].mean()), 5)
    out["cands_per_s1"] = round(len(pairs) / len(q), 2)
    out["RR"] = round(1 - len(pairs) / (len(q) * len(r)), 7)
    # upper bound: share of queries whose true set is fully retrieved
    return out


def run(cfg: dict, split: str = "dev") -> dict:
    t = Timer()
    d = block_dir(cfg)
    src_split = "train" if split == "dev" else "test"
    s1 = read_norm(cfg, src_split, 1)
    if split == "dev":
        dev = build_dev_universe(cfg)
        s1 = s1[s1["entity_id"].isin(dev.loc[dev["role"] == "query", "entity_id"])]
        gt = io_utils.load_ground_truth(cfg["paths"]["dataset"])
    res = {"stage": "block", "notes": ""}
    stats = {}
    for country in sorted(s1["country"].unique()):
        q = s1[s1["country"] == country].reset_index(drop=True)
        r = pd.concat([read_norm(cfg, src_split, k, country) for k in (2, 3)], ignore_index=True)
        pairs, info = block_country(cfg, q, r)
        q.to_parquet(os.path.join(d, f"{split}_{country}_q.parquet"), index=False)
        r.to_parquet(os.path.join(d, f"{split}_{country}_r.parquet"), index=False)
        pairs.to_parquet(os.path.join(d, f"{split}_{country}_pairs.parquet"), index=False)
        if split == "dev":
            info.update(completeness(pairs, q, r, gt))
            res[f"PC_union_{country}"] = info["PC_union"]
            res[f"RR_{country}"] = info["RR"]
        info.update({"n_q": len(q), "n_r": len(r)})
        stats[country] = info
        log.info("block %s %s: %s", split, country, json.dumps(info))
        del q, r, pairs
    res["cands_per_s1"] = round(sum(s["pairs"] for s in stats.values()) / sum(s["n_q"] for s in stats.values()), 2)
    res["runtime_s"] = t()
    res["notes"] = json.dumps(stats)
    json.dump(stats, open(os.path.join(cfg["paths"]["runs"], f"block_{split}.json"), "w"), indent=1)
    return res
