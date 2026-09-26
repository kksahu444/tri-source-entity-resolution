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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

from .. import io_utils
from ..cv import build_dev_universe
from ..normalize import norm_path
from ..utils import Timer, get_logger
from .crowd import crowd_channel, key_channel, second_hop_channel
from .tfidf import TfidfSpace, combined, pair_cos, topk

log = get_logger()
BLOCK_COLS = ["entity_id", "core", "addr_core", "house", "street", "city", "src"]


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


def rarest_alpha(s: pd.Series, df_counts: dict, min_len=3) -> pd.Series:
    """Rarest alphabetic token (by document frequency in the index) of each string: skips generic
    street-type words ("rue", "road") without a language-specific list."""
    def f(x):
        best, bdf = "", None
        for t in set(x.split()):
            if len(t) >= min_len and not t[0].isdigit():
                d = df_counts.get(t, 0)
                if bdf is None or d < bdf:
                    best, bdf = t, d
        return best
    return s.map(f)


def token_df(s: pd.Series) -> dict:
    from collections import Counter
    c = Counter()
    for x in s:
        c.update(set(x.split()))
    return c


def house_key(df: pd.DataFrame, street_df: dict | None = None) -> pd.Series:
    st = rarest_alpha(df["street"], street_df) if street_df is not None else first_alpha(df["street"])
    ct = first_alpha(df["city"], 2)
    tail = st.where(st != "", ct)
    return (df["house"] + "|" + tail).where((df["house"] != "") & (tail != ""), "")


def reverse_name_channel(rec_idx: np.ndarray, rn, qn, k: int, min_cos: float, chunk: int):
    """C7: the given records (empty address) retrieve their top-k S1 queries by name cosine.
    Returns (qi, ri) int32 arrays."""
    eq, ek, _ = topk(rn[rec_idx], qn.T.tocsr(), k, threshold=min_cos, chunk=chunk)
    return ek.astype(np.int32), rec_idx[eq].astype(np.int32)


def record_key_codes(r: pd.DataFrame, names: list[str]) -> dict:
    """Exact record keys for C6: int code per record, -1 where a key part is empty."""
    core = r["core"].astype(str)
    parts = {"name": core, "house": r["house"].astype(str), "city": r["city"].astype(str),
             "addr": r["addr_core"].astype(str)}
    out = {}
    for name in names:
        cols = [parts[p] for p in name.split("+")]
        key = cols[0]
        for c in cols[1:]:
            key = key + "|" + c
        empty = np.logical_or.reduce([(c == "").to_numpy() for c in cols])
        codes, _ = pd.factorize(key)
        out[name] = np.where(empty, -1, codes).astype(np.int64)
    return out


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
    space = TfidfSpace()
    rn, ra = space.fit_transform(r["core"].tolist(), r["addr_core"].tolist())
    qn, qa = space.transform(q["core"].tolist(), q["addr_core"].tolist())
    rc_t = combined(rn, ra).T.tocsr()
    qc = combined(qn, qa)
    # candidate generation on the rarer features only (short posting lists), then exact rescoring
    cap = max(bc.get("rare_df_min", 20000), int(bc.get("rare_df_share", 0.005) * len(r)))
    col_df = np.diff(rc_t.indptr)
    rc_t = sp.csr_matrix(sp.diags((col_df <= cap).astype(np.float32)) @ rc_t)
    rc_t.eliminate_zeros()
    info["t_vectorize"] = t()
    info["rare_df_cap"] = cap
    log.info("  vectorised %s queries, %s records, rare cap %d (%.0fs)", f"{len(q):,}", f"{len(r):,}", cap, t())
    gq, gr, _ = topk(qc, rc_t, bc.get("c1_generate", 100), threshold=0.0, chunk=bc["query_chunk"])
    del rc_t, qc
    gs = 0.5 * (pair_cos(qn, rn, gq, gr) + pair_cos(qa, ra, gq, gr))
    c1q, c1r, c1s = _topn_per_q(gq, gr, gs, bc["k_tfidf"])
    del gq, gr, gs
    info["t_c1"] = t()
    log.info("  C1: %s pairs (%.0fs)", f"{len(c1q):,}", t())
    c2q, c2r, c2size = crowd_channel(sorted_key(q["core"]), sorted_key(r["core"]), qa, ra, bc["crowd_cap"])
    info["t_c2"] = t()
    log.info("  C2: %s pairs (%.0fs)", f"{len(c2q):,}", t())
    keys = {}
    sdf = token_df(r["street"]) if bc.get("key_rare_token", False) else None
    hk = lambda d: house_key(d, sdf)
    for name, fn in (("c5a", hk), ("c5b", name_city_key)):
        kq, kr, skipped = key_channel(fn(q), fn(r), bc["key_block_cap"])
        s = 0.5 * (pair_cos(qn, rn, kq, kr) + pair_cos(qa, ra, kq, kr))
        keys[name] = _topn_per_q(kq, kr, s, bc.get("key_keep", 10))
        info[f"{name}_skipped_q"] = skipped
    info["t_c5"] = t()

    n_r = np.int64(len(r))
    chans = {"c1": (c1q, c1r), "c2": (c2q, c2r), **{k: v[:2] for k, v in keys.items()}}
    rv = bc.get("reverse_empty") or {}
    if rv.get("enabled", False):
        # C7: records with an empty address retrieve their top-k S1s by name alone. In C1 their
        # combined score is capped at 0.5 and in large name crowds C2 ranks them last, so the
        # error analysis found 59% of the blocking loss in entities with such a missed record.
        emp = np.flatnonzero(r["addr_core"].to_numpy() == "")
        if len(emp):
            c7q, c7r = reverse_name_channel(emp, rn, qn, rv.get("k", 10), rv.get("min_cos", 0.3), bc["query_chunk"])
            if rv.get("forward_k", 0):
                # A record's top-k S1s depend on how many S1s compete: the dev universe holds ~30% of the
                # queries test has, so reverse PC on dev is optimistic (runs/c7_scale.md). The forward
                # direction searches the same record pool in dev and test.
                fq, fr, _ = topk(qn, rn[emp].T.tocsr(), rv["forward_k"], threshold=rv.get("min_cos", 0.3),
                                 chunk=bc["query_chunk"])
                c7q, c7r = np.concatenate([c7q, fq.astype(np.int32)]), np.concatenate([c7r, emp[fr].astype(np.int32)])
            chans["c7"] = (c7q, c7r)
        info["c7_records"] = int(len(emp))
        info["t_c7"] = t()
        log.info("  C7: %s pairs from %s empty-address records (%.0fs)", f"{len(chans.get('c7', ([],))[0]):,}",
                 f"{len(emp):,}", t())
    sh = bc.get("second_hop") or {}
    if sh.get("enabled", False):
        # C6: expand from confident first-pass pairs through exact record keys (query expansion)
        first = np.unique(np.concatenate([a.astype(np.int64) * n_r + b for a, b in chans.values()]))
        fq, fr = (first // n_r).astype(np.int32), (first % n_r).astype(np.int32)
        fs = 0.5 * (pair_cos(qn, rn, fq, fr) + pair_cos(qa, ra, fq, fr))
        anc = fs >= sh.get("anchor_c1", 0.8)
        c6q, c6r = second_hop_channel(fq[anc], fr[anc], record_key_codes(r, sh.get("keys", ["name+house"])),
                                      sh.get("cap", 20))
        chans["c6"] = (c6q, c6r)
        info["c6_anchors"] = int(anc.sum())
        info["t_c6"] = t()
        log.info("  C6: %s pairs from %s anchors (%.0fs)", f"{len(c6q):,}", f"{anc.sum():,}", t())
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


def run(cfg: dict, split: str = "dev", frac: float | None = None) -> dict:
    """frac: benchmark mode; only that share of the queries is blocked and nothing is saved."""
    t = Timer()
    d = block_dir(cfg)
    src_split = "train" if split == "dev" else "test"
    s1_tbl = pq.read_table(norm_path(cfg, src_split, 1))
    n_full = {}
    if split == "dev":
        dev = build_dev_universe(cfg)
        ids = pa.array(dev.loc[dev["role"] == "query", "entity_id"].tolist())
        s1_tbl = s1_tbl.filter(pc.is_in(s1_tbl["entity_id"], value_set=ids))
        gt = io_utils.load_ground_truth(cfg["paths"]["dataset"])
    if frac:
        n_full = pd.Series(s1_tbl["country"].to_pylist()).value_counts().to_dict()
        rng = np.random.default_rng(cfg["seed"])
        s1_tbl = s1_tbl.filter(pa.array(rng.random(s1_tbl.num_rows) < frac))
    res = {"stage": "block_bench" if frac else "block", "notes": ""}
    stats = {}
    for country in sorted(set(s1_tbl["country"].to_pylist())):
        q_tbl = s1_tbl.filter(pc.equal(s1_tbl["country"], country))
        q = q_tbl.select(BLOCK_COLS).to_pandas()
        tbl_r = pa.concat_tables([pq.read_table(norm_path(cfg, src_split, k), filters=[("country", "=", country)])
                                  for k in (2, 3)])
        r = tbl_r.select(BLOCK_COLS).to_pandas()
        log.info("block %s %s: loaded %s queries, %s records (%.0fs)", split, country, f"{len(q):,}", f"{len(r):,}", t())
        pairs, info = block_country(cfg, q, r)
        if frac:
            per_q = (info["t_c1"] - info["t_vectorize"] + info["t_union"] - info["t_c5"]) / max(len(q), 1)
            info["projected_full_s"] = round(info["t_union"] + per_q * (n_full[country] - len(q)), 0)
        else:
            pq.write_table(q_tbl, os.path.join(d, f"{split}_{country}_q.parquet"))
            pq.write_table(tbl_r, os.path.join(d, f"{split}_{country}_r.parquet"))
            pairs.to_parquet(os.path.join(d, f"{split}_{country}_pairs.parquet"), index=False)
        del tbl_r
        if split == "dev":
            info.update(completeness(pairs, q, r, gt))
            res[f"PC_union_{country}"] = info["PC_union"]
            res[f"RR_{country}"] = info["RR"]
        info.update({"n_q": len(q), "n_r": len(r)})
        stats[country] = info
        log.info("block %s %s: %s", split, country, json.dumps(info))
        del q, r, pairs, q_tbl
    res["cands_per_s1"] = round(sum(s["pairs"] for s in stats.values()) / sum(s["n_q"] for s in stats.values()), 2)
    res["runtime_s"] = t()
    res["notes"] = json.dumps(stats)
    name = f"block_{split}{'_bench' if frac else ''}.json"
    json.dump(stats, open(os.path.join(cfg["paths"]["runs"], name), "w"), indent=1)
    return res
