"""Pairwise features for (Source-1 record, candidate) pairs.

Feature families
----------------
* **name** - rapidfuzz edit/token similarities on normalised and legal-free names, TF-IDF cosines,
  IDF-weighted fuzzy token overlap (rare shared tokens are strong evidence; rare *unmatched*
  tokens such as "bakery" vs "robotics" are strong counter-evidence), acronym / legal-form /
  digit agreement, best score over DBA name variants.
* **address** - similarities on the landmark-free core, postcode and house-number agreement or
  conflict, landmark flags, coverage of one address by the other (partial addresses).
* **context** - how a pair ranks among all candidates of the same Source-1 record and among all
  Source-1 records competing for the same candidate. Source 1 is deduplicated, so a candidate
  that fits another Source-1 record better is probably not ours.
* **provenance** - which blockers proposed the pair and with what similarity.

No feature uses the country label except for "same country", so unseen countries (France) are
handled like any other.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .blocking import TextIndex
from .normalize import skeleton

log = logging.getLogger(__name__)

CONTEXT_COLUMNS = ["cos_full_char", "cos_name_char", "name_tset_core", "name_wjacc", "addr_tset", "cos_addr_word"]
ID_COLUMNS = ["s1_id", "cand_id"]


def _pairwise(left: list[str], right: list[str], scorer, scale: float = 100.0) -> np.ndarray:
    """Element-wise string similarity in [0, 1]; NaN when either side is empty."""
    scores = process.cpdist(left, right, scorer=scorer, workers=-1, dtype=np.float32) / scale
    empty = np.fromiter((not a or not b for a, b in zip(left, right)), dtype=bool, count=len(left))
    scores[empty] = np.nan
    return scores.astype(np.float32)


def _pair_cosine(mat, li: np.ndarray, ri: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Row-wise dot product of L2-normalised sparse rows ``mat[li[k]]`` and ``mat[ri[k]]``."""
    out = np.empty(len(li), dtype=np.float32)
    for s in range(0, len(li), chunk):
        a, b = mat[li[s:s + chunk]], mat[ri[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _token_overlap(left: list[str], right: list[str], idf: dict[str, float], jw_cut: float = 0.9) -> np.ndarray:
    """IDF-weighted token agreement with fuzzy token matching (typos, transliterations).

    Columns: weighted Jaccard, weighted coverage of left, of right, unmatched IDF mass left,
    right, and the IDF of the rarest exactly shared token.
    """
    default = max(idf.values(), default=1.0)
    out = np.full((len(left), 6), np.nan, dtype=np.float32)
    for i, (a, b) in enumerate(zip(left, right)):
        la, lb = set(a.split()), set(b.split())
        if not la or not lb:
            continue
        w = {t: idf.get(t, default) for t in la | lb}
        exact = la & lb
        ma, mb = set(exact), set(exact)
        for x in la - exact:
            for y in lb - exact:
                if skeleton(x) == skeleton(y) or (min(len(x), len(y)) >= 4 and JaroWinkler.similarity(x, y) >= jw_cut):
                    ma.add(x)
                    mb.add(y)
        wa, wb = sum(w[t] for t in la), sum(w[t] for t in lb)
        shared = sum(w[t] for t in exact)
        cov_a, cov_b = sum(w[t] for t in ma), sum(w[t] for t in mb)
        out[i] = (shared / (wa + wb - shared), cov_a / wa, cov_b / wb, wa - cov_a, wb - cov_b,
                  max((w[t] for t in exact), default=0.0))
    return out


def _set_agreement(left: list[str], right: list[str], prefix_len: int = 0) -> np.ndarray:
    """Agreement of space-separated token sets (postcodes, house numbers, digits, legal forms).

    Columns: both present, any shared, conflict (both present, none shared), Jaccard,
    first-token equal, any shared ``prefix_len`` prefix (0 = skip).
    """
    out = np.full((len(left), 6), np.nan, dtype=np.float32)
    out[:, :3] = 0.0
    for i, (a, b) in enumerate(zip(left, right)):
        if not a or not b:
            continue
        la, lb = a.split(), b.split()
        sa, sb = set(la), set(lb)
        inter = sa & sb
        out[i, 0] = 1.0
        out[i, 1] = float(bool(inter))
        out[i, 2] = float(not inter)
        out[i, 3] = len(inter) / len(sa | sb)
        out[i, 4] = float(la[0] == lb[0])
        if prefix_len:
            out[i, 5] = float(bool({t[:prefix_len] for t in sa} & {t[:prefix_len] for t in sb}))
    return out


def _variant_best(core_l, core_r, var_l, var_r, fallback: np.ndarray) -> np.ndarray:
    """Best token-set similarity over DBA / trade-name variants (only computed where variants exist)."""
    out = fallback.copy()
    for i in np.flatnonzero((var_l != "") | (var_r != "")):
        lv = var_l[i].split("|") if var_l[i] else [core_l[i]]
        rv = var_r[i].split("|") if var_r[i] else [core_r[i]]
        out[i] = max(fuzz.token_set_ratio(x, y) for x in lv + [core_l[i]] for y in rv + [core_r[i]]) / 100.0
    return out


def add_group_context(feats: pd.DataFrame, columns: list[str], counts: bool = True) -> pd.DataFrame:
    """Rank / gap of each score within its Source-1 group and within its candidate group."""
    new = {}
    by_s1 = feats.groupby("s1_id")
    by_s1_src = feats.groupby(["s1_id", "cand_src"])
    by_cand = feats.groupby("cand_id")
    for col in columns:
        name = col
        new[f"{name}_rk_s1"] = by_s1[col].rank(ascending=False, method="min")
        new[f"{name}_gap_s1"] = by_s1[col].transform("max") - feats[col]
        new[f"{name}_rk_s1src"] = by_s1_src[col].rank(ascending=False, method="min")
        new[f"{name}_rk_cand"] = by_cand[col].rank(ascending=False, method="min")
        new[f"{name}_gap_cand"] = by_cand[col].transform("max") - feats[col]
    if counts:
        new["n_cand_s1"] = by_s1["cand_id"].transform("size")
        new["n_cand_s1src"] = by_s1_src["cand_id"].transform("size")
        new["n_s1_cand"] = by_cand["s1_id"].transform("size")
    return pd.concat([feats, pd.DataFrame(new, index=feats.index).astype(np.float32)], axis=1)


def build_features(recs: pd.DataFrame, cands: pd.DataFrame, index: TextIndex) -> pd.DataFrame:
    """Compute the full feature matrix for every candidate pair (row order = ``cands`` order)."""
    by_id = recs.set_index("entity_id")
    L = by_id.loc[cands["s1_id"].to_numpy()].reset_index(drop=True)
    R = by_id.loc[cands["cand_id"].to_numpy()].reset_index(drop=True)
    li = index.pos.loc[cands["s1_id"].to_numpy()].to_numpy()
    ri = index.pos.loc[cands["cand_id"].to_numpy()].to_numpy()
    col = lambda df, c: df[c].tolist()  # noqa: E731

    f: dict[str, np.ndarray] = {"cand_src": cands["cand_src"].to_numpy()}
    for c in cands.columns:
        if c.startswith("blk_"):
            f[c] = cands[c].to_numpy()

    # ---- name ------------------------------------------------------------------------
    nn_l, nn_r = col(L, "name_norm"), col(R, "name_norm")
    nc_l, nc_r = col(L, "name_core"), col(R, "name_core")
    f["name_ratio_norm"] = _pairwise(nn_l, nn_r, fuzz.ratio)
    f["name_tset_norm"] = _pairwise(nn_l, nn_r, fuzz.token_set_ratio)
    f["name_ratio_core"] = _pairwise(nc_l, nc_r, fuzz.ratio)
    f["name_partial_core"] = _pairwise(nc_l, nc_r, fuzz.partial_ratio)
    f["name_tsort_core"] = _pairwise(nc_l, nc_r, fuzz.token_sort_ratio)
    f["name_tset_core"] = _pairwise(nc_l, nc_r, fuzz.token_set_ratio)
    f["name_jw_core"] = _pairwise(nc_l, nc_r, JaroWinkler.normalized_similarity, scale=1.0)
    f["name_lev_core"] = _pairwise(nc_l, nc_r, Levenshtein.normalized_similarity, scale=1.0)
    f["name_skel_ratio"] = _pairwise(col(L, "name_skel"), col(R, "name_skel"), fuzz.token_sort_ratio)
    tok = _token_overlap(nc_l, nc_r, index.name_idf)
    for j, name in enumerate(["wjacc", "cov_l", "cov_r", "unmatched_l", "unmatched_r", "max_shared_idf"]):
        f[f"name_{name}"] = tok[:, j]
    f["name_variant_best"] = _variant_best(
        np.array(nc_l, dtype=object), np.array(nc_r, dtype=object),
        L["name_variants"].to_numpy(dtype=object), R["name_variants"].to_numpy(dtype=object), f["name_tset_core"],
    )
    compact_l = [c.replace(" ", "") for c in nc_l]
    compact_r = [c.replace(" ", "") for c in nc_r]
    f["acronym_match"] = np.array(
        [float((bool(al) and al == cr) or (bool(ar) and ar == cl))
         for al, ar, cl, cr in zip(L["name_acronym"], R["name_acronym"], compact_l, compact_r)], dtype=np.float32)
    f["name_compact_eq"] = np.array([float(a == b and bool(a)) for a, b in zip(compact_l, compact_r)], dtype=np.float32)
    legal = _set_agreement(col(L, "name_legal"), col(R, "name_legal"))
    f["legal_both"], f["legal_shared"], f["legal_conflict"] = legal[:, 0], legal[:, 1], legal[:, 2]
    digits = _set_agreement(col(L, "name_digits"), col(R, "name_digits"))
    f["name_digit_conflict"] = digits[:, 2]
    f["name_len_l"] = np.array([len(x) for x in nc_l], dtype=np.float32)
    f["name_len_r"] = np.array([len(x) for x in nc_r], dtype=np.float32)
    f["name_ntok_l"] = np.array([len(x.split()) for x in nc_l], dtype=np.float32)
    f["name_ntok_r"] = np.array([len(x.split()) for x in nc_r], dtype=np.float32)
    f["name_first_tok_eq"] = np.array(
        [float(bool(a) and bool(b) and a.split()[0] == b.split()[0]) for a, b in zip(nc_l, nc_r)], dtype=np.float32)

    # ---- address ---------------------------------------------------------------------
    ac_l, ac_r = col(L, "addr_core"), col(R, "addr_core")
    an_l, an_r = col(L, "addr_norm"), col(R, "addr_norm")
    f["addr_ratio"] = _pairwise(ac_l, ac_r, fuzz.ratio)
    f["addr_partial"] = _pairwise(ac_l, ac_r, fuzz.partial_ratio)
    f["addr_tsort"] = _pairwise(ac_l, ac_r, fuzz.token_sort_ratio)
    f["addr_tset"] = _pairwise(ac_l, ac_r, fuzz.token_set_ratio)
    f["addr_tset_full"] = _pairwise(an_l, an_r, fuzz.token_set_ratio)
    atok = _token_overlap(ac_l, ac_r, index.addr_idf)
    for j, name in enumerate(["wjacc", "cov_l", "cov_r", "unmatched_l", "unmatched_r", "max_shared_idf"]):
        f[f"addr_{name}"] = atok[:, j]
    pcs = _set_agreement(col(L, "addr_postcodes"), col(R, "addr_postcodes"), prefix_len=3)
    f["pc_both"], f["pc_match"], f["pc_conflict"], f["pc_prefix3"] = pcs[:, 0], pcs[:, 1], pcs[:, 2], pcs[:, 5]
    nums = _set_agreement(col(L, "addr_numbers"), col(R, "addr_numbers"))
    f["num_both"], f["num_match"], f["num_conflict"] = nums[:, 0], nums[:, 1], nums[:, 2]
    f["num_jacc"], f["num_first_eq"] = nums[:, 3], nums[:, 4]
    f["landmark_l"] = L["has_landmark"].to_numpy(dtype=np.float32)
    f["landmark_r"] = R["has_landmark"].to_numpy(dtype=np.float32)
    f["addr_len_l"] = np.array([len(x) for x in ac_l], dtype=np.float32)
    f["addr_len_r"] = np.array([len(x) for x in ac_r], dtype=np.float32)

    # ---- vector-space cosines ----------------------------------------------------------
    for name in ("name_char", "name_word", "full_char", "addr_word", "addr_char"):
        f[f"cos_{name}"] = _pair_cosine(index.mats[name], li, ri)
    if "embed" in index.dense:
        emb = index.dense["embed"]
        f["cos_embed"] = np.einsum("ij,ij->i", emb[li], emb[ri]).astype(np.float32)

    # ---- country (open set: only "same label or not") ------------------------------------
    c_l, c_r = L["country_norm"].to_numpy(dtype=object), R["country_norm"].to_numpy(dtype=object)
    f["country_eq"] = np.where((c_l == "") | (c_r == ""), np.nan, (c_l == c_r).astype(np.float32))

    feats = pd.concat([cands[ID_COLUMNS].reset_index(drop=True), pd.DataFrame(f)], axis=1)
    feats = add_group_context(feats, CONTEXT_COLUMNS)
    log.info("features: %d pairs x %d features", len(feats), feats.shape[1] - len(ID_COLUMNS))
    return feats


def feature_columns(feats: pd.DataFrame) -> list[str]:
    """Model input columns (everything except ids and labels)."""
    return [c for c in feats.columns if c not in (*ID_COLUMNS, "label")]
