"""D4 pair features, computed per country on view B. Never country, ids or row order.

cheap_features(): blocking scores/ranks + a few character similarities (pruner input, D3).
full_features():  names, legal form, address components, context (stage-A input, D6).
String features run in chunks of pairs so only the gathered values are held as Python objects.
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import CountVectorizer

from .blocking.tfidf import pair_cos
from .normalize import skeleton

F32 = np.float32
CHUNK = 1_000_000
WORKERS = int(os.environ.get("ER_WORKERS", max(1, min(6, (os.cpu_count() or 2) - 2))))


def _col(df: pd.DataFrame, col: str, idx: np.ndarray) -> np.ndarray:
    arr = df[col].array
    if hasattr(arr, "_pa_array"):       # Arrow-backed: materialise only the gathered values
        return arr._pa_array.take(pa.array(idx)).to_numpy(zero_copy_only=False)
    return df[col].to_numpy(dtype=object)[idx]


def _num(df: pd.DataFrame, col: str, idx: np.ndarray) -> np.ndarray:
    return np.asarray(df[col].to_numpy())[idx].astype(F32)


def _sim(a, b, scorer, scale=100.0) -> np.ndarray:
    return (cpdist(a, b, scorer=scorer, workers=-1) / scale).astype(F32)


def _binary_overlap(q_texts, r_texts, qi, ri):
    """Token overlap count and containment both ways, via binary token vectors."""
    cv = CountVectorizer(binary=True, token_pattern=r"\S+", lowercase=False, dtype=np.float32)
    cv.fit(pd.concat([pd.Series(q_texts, dtype=object), pd.Series(r_texts, dtype=object)], ignore_index=True))
    xq = cv.transform(pd.Series(q_texts, dtype=object)).tocsr()
    xr = cv.transform(pd.Series(r_texts, dtype=object)).tocsr()
    xq.sort_indices()
    xr.sort_indices()
    ov = pair_cos(xq, xr, qi, ri)
    nq = np.diff(xq.indptr)[qi].astype(F32)
    nr = np.diff(xr.indptr)[ri].astype(F32)
    with np.errstate(divide="ignore", invalid="ignore"):
        cq = np.where(nq > 0, ov / nq, 0).astype(F32)
        cr = np.where(nr > 0, ov / nr, 0).astype(F32)
    return ov.astype(F32), cq, cr


def _group_stats(key: np.ndarray, val: np.ndarray):
    """Per-row group max, group size and descending rank of val within group."""
    order = np.lexsort((-val, key))
    ks = key[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    sizes = np.diff(np.r_[starts, len(ks)])
    gmax = np.repeat(val[order][starts], sizes)
    rank = np.arange(len(ks)) - np.repeat(starts, sizes) + 1
    out_max, out_size, out_rank = np.empty_like(val), np.empty(len(val), np.int32), np.empty(len(val), np.int32)
    out_max[order], out_size[order], out_rank[order] = gmax, np.repeat(sizes, sizes), rank
    return out_max, out_size, out_rank


BLOCK_FEATS = ["in_c1", "in_c2", "in_c5a", "in_c5b", "c1_rank", "crowd_size", "cos_name", "cos_addr",
               "c1_score", "union_rank"]


def cheap_features(q: pd.DataFrame, r: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    qi, ri = pairs["qi"].to_numpy(), pairs["ri"].to_numpy()
    f = pairs[BLOCK_FEATS + [c for c in ("in_c6", "in_c7") if c in pairs.columns]].astype(F32).copy()
    f["src3"] = (pairs["src"].to_numpy() == 3).astype(F32)
    cols = {k: [] for k in ("n_jw", "n_tset", "house_eq", "house_conflict", "addr_r_empty")}
    for s in range(0, len(qi), CHUNK):
        a, b = qi[s:s + CHUNK], ri[s:s + CHUNK]
        cq, cr = _col(q, "core", a), _col(r, "core", b)
        cols["n_jw"].append(_sim(cq, cr, JaroWinkler.normalized_similarity, 1.0))
        cols["n_tset"].append(_sim(cq, cr, fuzz.token_set_ratio))
        hq, hr = _col(q, "house", a), _col(r, "house", b)
        cols["house_eq"].append(((hq == hr) & (hq != "")).astype(F32))
        cols["house_conflict"].append(((hq != hr) & (hq != "") & (hr != "")).astype(F32))
        cols["addr_r_empty"].append((_col(r, "addr_core", b) == "").astype(F32))
    for k, v in cols.items():
        f[k] = np.concatenate(v) if v else np.zeros(0, F32)
    c1 = f["c1_score"].to_numpy()
    gmax, gsize, _ = _group_stats(qi, c1)
    f["c1_gap"] = (gmax - c1).astype(F32)
    f["q_ncand"] = gsize.astype(F32)
    rmax, rsize, rrank = _group_stats(ri, c1)
    f["rev_rank"] = rrank.astype(F32)
    f["rev_gap"] = (rmax - c1).astype(F32)
    f["r_nq"] = rsize.astype(F32)
    return f


# ------------------------------------------------------------------ token-level helpers
def _is_abbr(s: str, l: str) -> bool:
    """Layer (c): s is a subsequence-abbreviation of l (same first letter, letters in order)."""
    if len(s) >= len(l) or s[0] != l[0]:
        return False
    it = iter(l)
    return all(c in it for c in s)


def _abbr_chunk(args):
    a, b = args
    fa, fb, acr = np.zeros(len(a), F32), np.zeros(len(a), F32), np.zeros(len(a), F32)
    for i, (x, y) in enumerate(zip(a, b)):
        xs, ys = x.split(), y.split()
        if not xs or not ys:
            continue
        sx, sy = set(xs), set(ys)
        fa[i] = sum(1 for t in xs if t in sy or any(_is_abbr(t, u) or _is_abbr(u, t) for u in ys if u[0] == t[0])) / len(xs)
        fb[i] = sum(1 for t in ys if t in sx or any(_is_abbr(t, u) or _is_abbr(u, t) for u in xs if u[0] == t[0])) / len(ys)
        if len(xs) == 1 and len(ys) >= 2 and xs[0] == "".join(t[0] for t in ys):
            acr[i] = 1
        elif len(ys) == 1 and len(xs) >= 2 and ys[0] == "".join(t[0] for t in xs):
            acr[i] = 1
    return fa, fb, acr


def token_match(a: np.ndarray, b: np.ndarray, pool: Pool | None):
    """(share of a-tokens matched, share of b-tokens matched, acronym match) with layer-(c) matching."""
    step = 200_000
    jobs = [(a[i:i + step], b[i:i + step]) for i in range(0, len(a), step)]
    parts = pool.map(_abbr_chunk, jobs) if pool is not None and len(jobs) > 1 else [_abbr_chunk(j) for j in jobs]
    if not parts:
        return np.zeros(0, F32), np.zeros(0, F32), np.zeros(0, F32)
    return tuple(np.concatenate([p[k] for p in parts]) for k in range(3))


def _skel_text(s) -> pd.Series:
    return pd.Series(s, dtype=object).map(lambda x: " ".join(skeleton(t) for t in x.split()))


def _string_chunk(q, r, qi, ri, pool) -> dict:
    g = _col
    out = {}
    cq, cr = g(q, "core", qi), g(r, "core", ri)
    out["n_ratio"] = _sim(cq, cr, fuzz.ratio)
    out["n_tsort"] = _sim(cq, cr, fuzz.token_sort_ratio)
    out["n_partial"] = _sim(cq, cr, fuzz.partial_ratio)
    n_tset = _sim(cq, cr, fuzz.token_set_ratio)
    nq, nr = g(q, "name_b", qi), g(r, "name_b", ri)
    out["full_tset"] = _sim(nq, nr, fuzz.token_set_ratio)
    out["full_ratio"] = _sim(nq, nr, fuzz.ratio)
    aq, ar = g(q, "alt", qi), g(r, "alt", ri)
    alt1 = np.where(ar != "", _sim(cq, ar, fuzz.token_set_ratio), 0)
    alt2 = np.where(aq != "", _sim(aq, cr, fuzz.token_set_ratio), 0)
    out["alt_best"] = np.maximum(alt1, alt2).astype(F32)
    out["n_best"] = np.maximum(out["alt_best"], n_tset)
    out["n_abbr_q"], out["n_abbr_r"], out["acronym"] = token_match(cq, cr, pool)
    out["n_len_q"] = np.fromiter((len(x.split()) for x in cq), F32, len(cq))
    out["n_len_r"] = np.fromiter((len(x.split()) for x in cr), F32, len(cr))
    lq, lr = g(q, "legal", qi), g(r, "legal", ri)
    out["legal_eq"] = ((lq == lr) & (lq != "")).astype(F32)
    out["legal_q_empty"] = (lq == "").astype(F32)
    out["legal_r_empty"] = (lr == "").astype(F32)
    xq, xr = g(q, "addr_core", qi), g(r, "addr_core", ri)
    out["a_tset"] = _sim(xq, xr, fuzz.token_set_ratio)
    out["a_ratio"] = _sim(xq, xr, fuzz.ratio)
    out["a_partial"] = _sim(xq, xr, fuzz.partial_ratio)
    out["a_abbr_q"], out["a_abbr_r"], _ = token_match(xq, xr, pool)
    hq, hr = g(q, "house", qi), g(r, "house", ri)
    out["house_q_empty"] = (hq == "").astype(F32)
    out["house_r_empty"] = (hr == "").astype(F32)
    out["house_ratio"] = _sim(hq, hr, fuzz.ratio)
    sq, sr = g(q, "house_sfx", qi), g(r, "house_sfx", ri)
    out["sfx_eq"] = ((sq == sr) & (sq != "")).astype(F32)
    out["sfx_conflict"] = ((sq != sr) & (sq != "") & (sr != "")).astype(F32)
    stq, str_ = g(q, "street", qi), g(r, "street", ri)
    out["street_tset"] = _sim(stq, str_, fuzz.token_set_ratio)
    out["street_ratio"] = _sim(stq, str_, fuzz.ratio)
    ctq, ctr = g(q, "city", qi), g(r, "city", ri)
    out["city_eq"] = ((ctq == ctr) & (ctq != "")).astype(F32)
    out["city_ratio"] = _sim(ctq, ctr, fuzz.ratio)
    out["city_tset"] = _sim(ctq, ctr, fuzz.token_set_ratio)
    out["city_in_addr"] = _sim(ctq, g(r, "addr_b", ri), fuzz.partial_ratio)
    out["city_r_empty"] = (ctr == "").astype(F32)
    rq, rr = g(q, "region", qi), g(r, "region", ri)
    out["region_eq"] = ((rq == rr) & (rq != "")).astype(F32)
    out["region_conflict"] = ((rq != rr) & (rq != "") & (rr != "")).astype(F32)
    out["region_r_empty"] = (rr == "").astype(F32)
    uq, ur = g(q, "unit", qi), g(r, "unit", ri)
    out["unit_tset"] = _sim(uq, ur, fuzz.token_set_ratio)
    out["unit_r_empty"] = (ur == "").astype(F32)
    out["landmark_tset"] = _sim(g(q, "landmark", qi), g(r, "landmark", ri), fuzz.token_set_ratio)
    return out


def full_features(q: pd.DataFrame, r: pd.DataFrame, pairs: pd.DataFrame, cheap: pd.DataFrame,
                  support: bool = False, gen: bool = False) -> pd.DataFrame:
    qi, ri = pairs["qi"].to_numpy(), pairs["ri"].to_numpy()
    f = cheap.copy()
    chunks = []
    with Pool(WORKERS) as pool:
        for s in range(0, len(qi), CHUNK):
            chunks.append(_string_chunk(q, r, qi[s:s + CHUNK], ri[s:s + CHUNK], pool))
    for k in (chunks[0] if chunks else {}):
        f[k] = np.concatenate([c[k] for c in chunks])
    del chunks
    # token overlaps through binary sparse vectors (numeric, whole country at once)
    ov, c1, c2 = _binary_overlap(q["core"], r["core"], qi, ri)
    f["n_tok_ov"], f["n_cont_q"], f["n_cont_r"] = ov, c1, c2
    _, c1, c2 = _binary_overlap(_skel_text(q["core"]), _skel_text(r["core"]), qi, ri)
    f["n_skel_cont_q"], f["n_skel_cont_r"] = c1, c2
    ov, c1, _ = _binary_overlap(q["legal"], r["legal"], qi, ri)
    f["legal_cont_q"] = c1
    f["legal_conflict"] = ((ov == 0) & (f["legal_q_empty"].to_numpy() == 0) & (f["legal_r_empty"].to_numpy() == 0)).astype(F32)
    ov, c1, c2 = _binary_overlap(q["addr_core"], r["addr_core"], qi, ri)
    f["a_tok_ov"], f["a_cont_q"], f["a_cont_r"] = ov, c1, c2
    ov, c1, c2 = _binary_overlap(q["nums"], r["nums"], qi, ri)
    f["num_ov"], f["num_cont_q"], f["num_cont_r"] = ov, c1, c2
    f["n_parts_q"] = _num(q, "n_parts", qi)
    f["n_parts_r"] = _num(r, "n_parts", ri)
    for c in ("f_tag", "f_lead", "f_bang", "f_digits", "f_nonlatin"):
        f[f"{c}_r"] = _num(r, c, ri)
    f["f_tag_q"] = _num(q, "f_tag", qi)
    # context: crowds and competition
    key = pd.Series(q["core"], dtype=object).map(lambda x: " ".join(sorted(set(x.split()))))
    f["q_name_crowd"] = key.map(key.value_counts()).to_numpy()[qi].astype(F32)
    kc = key + "|" + pd.Series(q["city"], dtype=object)
    f["q_name_city_crowd"] = kc.map(kc.value_counts()).to_numpy()[qi].astype(F32)
    strong = (f["c1_score"].to_numpy() >= 0.6).astype(F32)
    f["q_n_strong"] = np.bincount(qi, weights=strong, minlength=len(q))[qi].astype(F32)
    nb = f["n_best"].to_numpy()
    gmax, _, grank = _group_stats(qi, nb)
    f["n_best_gap"] = (gmax - nb).astype(F32)
    f["n_best_rank"] = grank.astype(F32)
    rmax, _, rrank = _group_stats(ri, nb)
    f["rev_n_best_rank"] = rrank.astype(F32)
    f["rev_n_best_gap"] = (rmax - nb).astype(F32)
    if support:
        support_features(f, q, r, qi, ri)
    if gen:
        gen_features(f, q, r, qi, ri)
    return f


def _house_num(h: np.ndarray) -> np.ndarray:
    out = np.full(len(h), np.nan)
    for i, x in enumerate(h):
        if x and len(x) <= 9:
            out[i] = float(x)
    return out


def gen_features(f: pd.DataFrame, q, r, qi, ri) -> None:
    """Generic features aimed at observed noise mechanisms (no dataset-specific word lists):
    website-style names (space-free containment), house numbers with dropped/changed digits or ranges,
    and agreement with the S1's best other candidate (anchor), a non-chaining consistency check."""
    cols = {k: [] for k in ("cmp_sub", "cmp_partial", "house_sub", "house_reldiff", "house_in_rnums")}
    for s in range(0, len(qi), CHUNK):
        a, b = qi[s:s + CHUNK], ri[s:s + CHUNK]
        cq = np.array([x.replace(" ", "") for x in _col(q, "core", a)], dtype=object)
        nr = np.array([x.replace(" ", "") for x in _col(r, "name_b", b)], dtype=object)
        cr = np.array([x.replace(" ", "") for x in _col(r, "core", b)], dtype=object)
        cols["cmp_sub"].append(np.fromiter(((len(x) >= 4 and x in y) or (len(z) >= 4 and z in x)
                                            for x, y, z in zip(cq, nr, cr)), bool, len(cq)).astype(F32))
        cols["cmp_partial"].append(_sim(cq, nr, fuzz.partial_ratio))
        hq, hr = _col(q, "house", a), _col(r, "house", b)
        cols["house_sub"].append(np.fromiter(((x != y) and bool(x) and bool(y) and (x in y or y in x)
                                              for x, y in zip(hq, hr)), bool, len(hq)).astype(F32))
        nq, nr_ = _house_num(hq), _house_num(hr)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(nq - nr_) / np.maximum(np.maximum(nq, nr_), 1)
        cols["house_reldiff"].append(np.where(np.isnan(rel), -1, rel).astype(F32))
        rn = _col(r, "nums", b)
        cols["house_in_rnums"].append(np.fromiter((bool(x) and x in set(y.split()) for x, y in zip(hq, rn)),
                                                  bool, len(hq)).astype(F32))
    for k, v in cols.items():
        f[k] = np.concatenate(v) if v else np.zeros(0, F32)
    # anchor agreement: the S1's best other candidate by blocking score
    c1 = f["c1_score"].to_numpy()
    order = np.lexsort((-c1, qi))
    qs = qi[order]
    starts = np.r_[0, np.flatnonzero(np.diff(qs)) + 1]
    sizes = np.diff(np.r_[starts, len(qs)])
    best = np.repeat(ri[order][starts], sizes)
    second = np.repeat(np.where(sizes >= 2, ri[order][np.minimum(starts + 1, len(qs) - 1)], -1), sizes)
    is_top = np.zeros(len(qs), bool)
    is_top[starts] = True
    anc_sorted = np.where(is_top, second, best)
    anchor = np.empty(len(qi), np.int64)
    anchor[order] = anc_sorted
    has = anchor >= 0
    an = np.where(has, anchor, 0)
    name_sim = np.zeros(len(qi), F32)
    addr_sim = np.zeros(len(qi), F32)
    for s in range(0, len(qi), CHUNK):
        b, a2 = ri[s:s + CHUNK], an[s:s + CHUNK]
        name_sim[s:s + CHUNK] = _sim(_col(r, "core", b), _col(r, "core", a2), fuzz.token_set_ratio)
        addr_sim[s:s + CHUNK] = _sim(_col(r, "addr_core", b), _col(r, "addr_core", a2), fuzz.token_set_ratio)
    f["anc_name_sim"] = np.where(has, name_sim, -1).astype(F32)
    f["anc_addr_sim"] = np.where(has, addr_sim, -1).astype(F32)


def _sorted_tokens(x: str) -> str:
    return " ".join(sorted(set(x.split())))


def _within_q_count(qi: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """For each pair: how many OTHER candidates of the same S1 share this record-side code (-1 = none)."""
    key = qi.astype(np.int64) * (int(codes.max()) + 2) + codes + 1
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return np.where(codes >= 0, cnt[inv] - 1, 0).astype(F32)


def support_features(f: pd.DataFrame, q, r, qi, ri) -> None:
    """Candidate support: the true records of an S1 (3.5 on average) resemble each other; a distractor
    usually stands alone. Counts of other candidates of the same S1 agreeing with this record."""
    r_core = pd.Series(r["core"], dtype=object).map(_sorted_tokens)
    r_addr = pd.Series(r["addr_core"], dtype=object).map(_sorted_tokens)
    for name, series in (("name", r_core), ("addr", r_addr), ("house", pd.Series(r["house"], dtype=object))):
        codes, _ = pd.factorize(series)
        codes = np.where(series.to_numpy() == "", -1, codes)[ri]
        f[f"sup_{name}"] = _within_q_count(qi, codes)
    ncand = f["q_ncand"].to_numpy()
    f["sup_name_share"] = (f["sup_name"].to_numpy() / np.maximum(ncand - 1, 1)).astype(F32)
    # strength of the best-scored other candidate sharing the name key (sibling agreement)
    strong = (f["n_best"].to_numpy() >= 0.9) & (f["a_tset"].to_numpy() >= 0.7)
    f["q_n_name_addr_strong"] = np.bincount(qi, weights=strong.astype(float), minlength=len(q))[qi].astype(F32)
