"""D4 pair features, computed per country on view B. Never country, ids or row order.

cheap_features(): blocking scores/ranks + a few character similarities (pruner input, D3).
full_features():  names, legal form, address components, context (stage-A input, D6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.feature_extraction.text import CountVectorizer

from .blocking.tfidf import pair_cos
from .normalize import skeleton

F32 = np.float32


def _col(df: pd.DataFrame, col: str, idx: np.ndarray) -> np.ndarray:
    return df[col].to_numpy(dtype=object)[idx]


def _sim(a, b, scorer, scale=100.0) -> np.ndarray:
    return (cpdist(a, b, scorer=scorer, workers=-1) / scale).astype(F32)


def _binary_overlap(q_texts, r_texts, qi, ri, analyzer=None):
    """Token overlap count and containment both ways, via binary token vectors."""
    cv = CountVectorizer(binary=True, token_pattern=r"\S+", lowercase=False, dtype=np.float32,
                         analyzer=analyzer or "word")
    cv.fit(pd.concat([pd.Series(q_texts), pd.Series(r_texts)], ignore_index=True))
    xq = cv.transform(q_texts).tocsr()
    xr = cv.transform(r_texts).tocsr()
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
    f = pairs[BLOCK_FEATS].astype(F32).copy()
    f["src3"] = (pairs["src"].to_numpy() == 3).astype(F32)
    cq, cr = _col(q, "core", qi), _col(r, "core", ri)
    f["n_jw"] = _sim(cq, cr, JaroWinkler.normalized_similarity, 1.0)
    f["n_tset"] = _sim(cq, cr, fuzz.token_set_ratio)
    hq, hr = _col(q, "house", qi), _col(r, "house", ri)
    f["house_eq"] = ((hq == hr) & (hq != "")).astype(F32)
    f["house_conflict"] = ((hq != hr) & (hq != "") & (hr != "")).astype(F32)
    f["addr_r_empty"] = (_col(r, "addr_core", ri) == "").astype(F32)
    gmax, gsize, _ = _group_stats(qi, f["c1_score"].to_numpy())
    f["c1_gap"] = (gmax - f["c1_score"].to_numpy()).astype(F32)
    f["q_ncand"] = gsize.astype(F32)
    rmax, rsize, rrank = _group_stats(ri, f["c1_score"].to_numpy())
    f["rev_rank"] = rrank.astype(F32)
    f["rev_gap"] = (rmax - f["c1_score"].to_numpy()).astype(F32)
    f["r_nq"] = rsize.astype(F32)
    return f


def _acronym_match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(len(a), F32)
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            continue
        xs, ys = x.split(), y.split()
        if len(xs) == 1 and len(ys) >= 2 and xs[0] == "".join(t[0] for t in ys):
            out[i] = 1
        elif len(ys) == 1 and len(xs) >= 2 and ys[0] == "".join(t[0] for t in xs):
            out[i] = 1
    return out


def _skel_text(s: pd.Series) -> pd.Series:
    return s.map(lambda x: " ".join(skeleton(t) for t in x.split()))


def _abbrev_token_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Layer (c): share of tokens on each side matched exactly or as a subsequence-abbreviation
    (same first letter, letters in order) of a token on the other side."""
    def is_abbr(s, l):
        if len(s) >= len(l) or s[0] != l[0]:
            return False
        it = iter(l)
        return all(c in it for c in s)

    fa, fb = np.zeros(len(a), F32), np.zeros(len(a), F32)
    for i, (x, y) in enumerate(zip(a, b)):
        xs, ys = x.split(), y.split()
        if not xs or not ys:
            continue
        sy = set(ys)
        sx = set(xs)
        fa[i] = sum(1 for t in xs if t in sy or any(is_abbr(t, u) or is_abbr(u, t) for u in ys if u[0] == t[0])) / len(xs)
        fb[i] = sum(1 for t in ys if t in sx or any(is_abbr(t, u) or is_abbr(u, t) for u in xs if u[0] == t[0])) / len(ys)
    return fa, fb


def full_features(q: pd.DataFrame, r: pd.DataFrame, pairs: pd.DataFrame, cheap: pd.DataFrame) -> pd.DataFrame:
    qi, ri = pairs["qi"].to_numpy(), pairs["ri"].to_numpy()
    f = cheap.copy()
    g = lambda df, c, idx: _col(df, c, idx)
    # names
    cq, cr = g(q, "core", qi), g(r, "core", ri)
    f["n_ratio"] = _sim(cq, cr, fuzz.ratio)
    f["n_tsort"] = _sim(cq, cr, fuzz.token_sort_ratio)
    f["n_partial"] = _sim(cq, cr, fuzz.partial_ratio)
    nq, nr = g(q, "name_b", qi), g(r, "name_b", ri)
    f["full_tset"] = _sim(nq, nr, fuzz.token_set_ratio)
    f["full_ratio"] = _sim(nq, nr, fuzz.ratio)
    aq, ar = g(q, "alt", qi), g(r, "alt", ri)
    alt1 = np.where(ar != "", _sim(cq, ar, fuzz.token_set_ratio), 0)
    alt2 = np.where(aq != "", _sim(aq, cr, fuzz.token_set_ratio), 0)
    f["alt_best"] = np.maximum(alt1, alt2).astype(F32)
    f["n_best"] = np.maximum(f["alt_best"].to_numpy(), f["n_tset"].to_numpy())
    ov, c1, c2 = _binary_overlap(q["core"], r["core"], qi, ri)
    f["n_tok_ov"], f["n_cont_q"], f["n_cont_r"] = ov, c1, c2
    ov, c1, c2 = _binary_overlap(_skel_text(q["core"]), _skel_text(r["core"]), qi, ri)
    f["n_skel_cont_q"], f["n_skel_cont_r"] = c1, c2
    f["n_abbr_q"], f["n_abbr_r"] = _abbrev_token_match(cq, cr)
    f["acronym"] = _acronym_match(cq, cr)
    f["n_len_q"] = np.array([len(x.split()) for x in cq], F32)
    f["n_len_r"] = np.array([len(x.split()) for x in cr], F32)
    # legal form (soft only: flips are common, D0 block 11)
    lq, lr = g(q, "legal", qi), g(r, "legal", ri)
    f["legal_eq"] = ((lq == lr) & (lq != "")).astype(F32)
    f["legal_q_empty"] = (lq == "").astype(F32)
    f["legal_r_empty"] = (lr == "").astype(F32)
    ov, c1, c2 = _binary_overlap(q["legal"], r["legal"], qi, ri)
    f["legal_cont_q"] = c1
    f["legal_conflict"] = ((ov == 0) & (lq != "") & (lr != "")).astype(F32)
    # address
    xq, xr = g(q, "addr_core", qi), g(r, "addr_core", ri)
    f["a_tset"] = _sim(xq, xr, fuzz.token_set_ratio)
    f["a_ratio"] = _sim(xq, xr, fuzz.ratio)
    f["a_partial"] = _sim(xq, xr, fuzz.partial_ratio)
    ov, c1, c2 = _binary_overlap(q["addr_core"], r["addr_core"], qi, ri)
    f["a_tok_ov"], f["a_cont_q"], f["a_cont_r"] = ov, c1, c2
    f["a_abbr_q"], f["a_abbr_r"] = _abbrev_token_match(xq, xr)
    hq, hr = g(q, "house", qi), g(r, "house", ri)
    f["house_q_empty"] = (hq == "").astype(F32)
    f["house_r_empty"] = (hr == "").astype(F32)
    f["house_ratio"] = _sim(hq, hr, fuzz.ratio)
    sq, sr = g(q, "house_sfx", qi), g(r, "house_sfx", ri)
    f["sfx_eq"] = ((sq == sr) & (sq != "")).astype(F32)
    f["sfx_conflict"] = ((sq != sr) & (sq != "") & (sr != "")).astype(F32)
    ov, c1, c2 = _binary_overlap(q["nums"], r["nums"], qi, ri)
    f["num_ov"], f["num_cont_q"], f["num_cont_r"] = ov, c1, c2
    stq, str_ = g(q, "street", qi), g(r, "street", ri)
    f["street_tset"] = _sim(stq, str_, fuzz.token_set_ratio)
    f["street_ratio"] = _sim(stq, str_, fuzz.ratio)
    ctq, ctr = g(q, "city", qi), g(r, "city", ri)
    f["city_eq"] = ((ctq == ctr) & (ctq != "")).astype(F32)
    f["city_ratio"] = _sim(ctq, ctr, fuzz.ratio)
    f["city_tset"] = _sim(ctq, ctr, fuzz.token_set_ratio)
    f["city_in_addr"] = _sim(ctq, g(r, "addr_b", ri), fuzz.partial_ratio)
    f["city_r_empty"] = (ctr == "").astype(F32)
    rq, rr = g(q, "region", qi), g(r, "region", ri)
    f["region_eq"] = ((rq == rr) & (rq != "")).astype(F32)
    f["region_conflict"] = ((rq != rr) & (rq != "") & (rr != "")).astype(F32)
    f["region_r_empty"] = (rr == "").astype(F32)
    uq, ur = g(q, "unit", qi), g(r, "unit", ri)
    f["unit_tset"] = _sim(uq, ur, fuzz.token_set_ratio)
    f["unit_r_empty"] = (ur == "").astype(F32)
    f["landmark_tset"] = _sim(g(q, "landmark", qi), g(r, "landmark", ri), fuzz.token_set_ratio)
    f["n_parts_q"] = q["n_parts"].to_numpy()[qi].astype(F32)
    f["n_parts_r"] = r["n_parts"].to_numpy()[ri].astype(F32)
    # junk flags
    for c in ("f_tag", "f_lead", "f_bang", "f_digits", "f_nonlatin"):
        f[f"{c}_r"] = r[c].to_numpy()[ri].astype(F32)
    f["f_tag_q"] = q["f_tag"].to_numpy()[qi].astype(F32)
    # context: crowds and competition
    key = q["core"].map(lambda x: " ".join(sorted(set(x.split()))))
    f["q_name_crowd"] = key.map(key.value_counts()).to_numpy()[qi].astype(F32)
    kc = key + "|" + q["city"]
    f["q_name_city_crowd"] = kc.map(kc.value_counts()).to_numpy()[qi].astype(F32)
    strong = (f["c1_score"].to_numpy() >= 0.6).astype(F32)
    f["q_n_strong"] = pd.Series(strong).groupby(qi).transform("sum").to_numpy(F32)
    nb = f["n_best"].to_numpy()
    gmax, _, grank = _group_stats(qi, nb)
    f["n_best_gap"] = (gmax - nb).astype(F32)
    f["n_best_rank"] = grank.astype(F32)
    rmax, _, rrank = _group_stats(ri, nb)
    f["rev_n_best_rank"] = rrank.astype(F32)
    f["rev_n_best_gap"] = (rmax - nb).astype(F32)
    return f
