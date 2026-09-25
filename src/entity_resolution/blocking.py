"""Candidate generation (blocking).

Runs once per country partition (there are no cross-country matches in train, and TF-IDF fitted
per country keeps the vocabulary local). For every Source-1 record, against Source 2 and Source 3
separately:

==============  ===============================================  ===============================
channel         representation                                   catches
==============  ===============================================  ===============================
full_char       char 3-4gram TF-IDF of name + address            typos, generic look-alikes
crowd           same compact core name, top k by address cosine  big same-name crowds (a chain's
                                                                 branches, "Sai Traders")
addr_word       word 1-2gram TF-IDF of the address core          DBA / trade-name differences
key             exact keys (exact address; house number + street  exact duplicates, blank or
                + first name skeleton; first name skeleton +     reordered addresses the kNN
                city)                                            ranks low
name_word       word TF-IDF of name tokens + phonetic skeletons  name-only: off by default (k=0)
name_char       char 2-4gram TF-IDF of the legal-free name       name-only: off by default (k=0)
embed           optional multilingual sentence embedding         semantic / cross-lingual
==============  ===============================================  ===============================

Content-identical target records (same source, same normalised name and address) are blocked
once through a representative row and expanded back to every member id afterwards, so each
duplicate is listed and costs no extra kNN work.

The union is ranked by a fixed, hand-weighted blocking score (a sum of TF-IDF cosines) and cut to
``max_per_source`` candidates per Source-1 record and target source, plus each target's best
``reverse_k`` Source-1 records. The cut is a rule, not a
learned model, so the output of this module is exactly what the first scoring model sees
(``candidate_pairs.tsv``).

TF-IDF models are fitted on the records of the split being processed (unsupervised, no labels and
no external data), which also adapts the vocabulary to the unseen France records at test time.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from .config import BlockingConfig

log = logging.getLogger(__name__)

BLOCKER_BITS = {"name_char": 1, "name_word": 2, "full_char": 4, "addr_word": 8, "key": 16, "embed": 32,
                "crowd": 64}
KNN_BLOCKERS = ("full_char", "name_word", "addr_word", "name_char")
FEATURE_MATS = ("name_char", "name_word", "full_char", "addr_word", "addr_char")
N_THREADS = os.cpu_count() or 1


@dataclass
class TextIndex:
    """Vector representations of every record of a partition, aligned by row position."""

    ids: np.ndarray
    pos: pd.Series  # entity_id -> row position
    source: np.ndarray  # 1 / 2 / 3 per row
    mats: dict[str, sp.csr_matrix] = field(default_factory=dict)
    dense: dict[str, np.ndarray] = field(default_factory=dict)
    name_idf: dict[str, float] = field(default_factory=dict)
    addr_idf: dict[str, float] = field(default_factory=dict)


def _tfidf(texts: list[str], **kwargs) -> sp.csr_matrix:
    """L2-normalised TF-IDF matrix (rows are unit vectors, so dot product = cosine)."""
    vec = TfidfVectorizer(lowercase=False, sublinear_tf=True, dtype=np.float32, **kwargs)
    try:
        return vec.fit_transform(texts).tocsr()
    except ValueError:  # empty vocabulary, e.g. every address blank, or max_df pruned everything
        return sp.csr_matrix((len(texts), 1), dtype=np.float32)


def _idf(docs: list[str]) -> dict[str, float]:
    """Smoothed inverse document frequency of whitespace tokens."""
    df: dict[str, int] = {}
    for doc in docs:
        for tok in set(doc.split()):
            df[tok] = df.get(tok, 0) + 1
    n = len(docs)
    return {tok: float(np.log((n + 1) / (cnt + 1)) + 1.0) for tok, cnt in df.items()}


def build_text_index(recs: pd.DataFrame, cfg: BlockingConfig, for_features: bool = True) -> TextIndex:
    """Fit the text representations on the normalised records of one partition.

    Blocking alone only needs the kNN channels it runs plus the ones in the blocking score;
    ``for_features`` adds everything the pairwise features read.
    """
    ks = {"name_char": cfg.name_char_k, "name_word": cfg.name_word_k,
          "full_char": cfg.full_char_k, "addr_word": cfg.addr_word_k}
    wanted = {n for n, k in ks.items() if k > 0} | {n for n, w in cfg.score_weights.items() if w}
    if cfg.crowd_k > 0:
        wanted.add("addr_word")
    if for_features:
        wanted |= set(FEATURE_MATS)
    name_core = recs["name_core"].tolist()
    addr_core = recs["addr_core"].tolist()
    # max_df drops n-grams shared by a large share of the partition: they carry almost no signal
    # and their posting lists dominate the cost of the sparse products.
    char_df = cfg.char_max_df if len(recs) >= 1000 else 1.0
    builders = {
        "name_char": lambda: _tfidf(name_core, analyzer="char_wb", ngram_range=(2, 4), max_df=char_df),
        "name_word": lambda: _tfidf(
            [f"{c} " + " ".join(f"~{s}" for s in k.split()) for c, k in zip(name_core, recs["name_skel"])],
            analyzer=str.split),
        "full_char": lambda: _tfidf([f"{n} | {a}" for n, a in zip(name_core, addr_core)],
                                    analyzer="char_wb", ngram_range=(3, 4), max_df=char_df),
        "addr_word": lambda: _tfidf(addr_core, analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2)),
        "addr_char": lambda: _tfidf(addr_core, analyzer="char_wb", ngram_range=(3, 4), max_df=char_df),
    }
    ids = recs["entity_id"].to_numpy()
    index = TextIndex(ids=ids, pos=pd.Series(np.arange(len(ids)), index=ids), source=recs["source"].to_numpy(),
                      mats={name: build() for name, build in builders.items() if name in wanted})
    if for_features:
        index.name_idf, index.addr_idf = _idf(name_core), _idf(addr_core)
    if cfg.use_embeddings:
        index.dense["embed"] = _encode(recs, cfg.embed_model)
    return index


def _encode(recs: pd.DataFrame, model_name: str) -> np.ndarray:
    """Optional dense embeddings of "name, address" (lazy import; see requirements-embeddings.txt)."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(model_name)
    prefix = "query: " if "e5" in model_name.lower() else ""
    texts = [f"{prefix}{n}, {a}" for n, a in zip(recs["business_name"], recs["business_address"])]
    emb = model.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    return emb.astype(np.float32)


def sparse_topk(query: sp.csr_matrix, data: sp.csr_matrix, k: int, min_sim: float) -> tuple[np.ndarray, ...]:
    """Top-``k`` cosine neighbours in ``data`` for every row of ``query`` (multi-threaded, C++).

    Returns ``(query_row, data_row, similarity)`` arrays.
    """
    if query.shape[0] == 0 or data.shape[0] == 0 or k <= 0:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    top = sp_matmul_topn(query, data.T.tocsr(), top_n=k, threshold=min_sim, n_threads=N_THREADS).tocoo()
    return top.row.astype(np.int64), top.col.astype(np.int64), top.data.astype(np.float32)


def dense_topk(query: np.ndarray, data: np.ndarray, k: int, min_sim: float) -> tuple[np.ndarray, ...]:
    """Top-``k`` neighbours for L2-normalised dense vectors (chunked matrix product)."""
    if len(query) == 0 or len(data) == 0 or k <= 0:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    k = min(k, len(data))
    qs, ds, ss = [], [], []
    for start in range(0, len(query), 1024):
        sims = query[start:start + 1024] @ data.T
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(sims, top, axis=1)
        rows = np.repeat(np.arange(start, start + len(sims)), k)
        keep = vals.ravel() >= min_sim
        qs.append(rows[keep])
        ds.append(top.ravel()[keep])
        ss.append(vals.ravel()[keep])
    return np.concatenate(qs), np.concatenate(ds), np.concatenate(ss)


def pair_cosine(mat: sp.csr_matrix, li: np.ndarray, ri: np.ndarray, chunk: int = 200_000) -> np.ndarray:
    """Row-wise dot product of L2-normalised sparse rows ``mat[li[k]]`` and ``mat[ri[k]]``."""
    out = np.empty(len(li), dtype=np.float32)
    for s in range(0, len(li), chunk):
        a, b = mat[li[s:s + chunk]], mat[ri[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _record_keys(recs: pd.DataFrame) -> list[list[str]]:
    """Exact blocking keys per record. Every key includes the address or a piece of it: names
    repeat hundreds of times in one country, so a name-only key is either a stop-key or noise.
    None needs a postcode (the test addresses have none)."""
    keys_all = []
    for skel, addr, street, city in zip(recs["name_skel"], recs["addr_core"], recs["addr_street"],
                                        recs["addr_city"]):
        keys = []
        first = skel.split()[0] if skel else ""
        if len(addr) >= 12:
            keys.append(f"a:{addr}")
        if street:
            keys.append(f"hs:{street}|{first}")
        if city and len(first) >= 3:
            keys.append(f"nc:{first}|{city}")
        keys_all.append(keys)
    return keys_all


def _crowd_pairs(recs: pd.DataFrame, addr: sp.csr_matrix, s1_rows: np.ndarray, rep_rows: np.ndarray, k: int,
                 max_crowd: int, chunk_pairs: int = 20_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Name-crowd channel: targets with exactly the same compact core name, ranked by address
    cosine, top ``k`` per query. Crowds with more than ``max_crowd`` targets are skipped (the
    name+address kNN ranks those)."""
    compact = recs["name_core"].str.replace(" ", "", regex=False).to_numpy()
    codes, _ = pd.factorize(np.concatenate([compact[s1_rows], compact[rep_rows]]))
    q_code, t_code = codes[:len(s1_rows)], codes[len(s1_rows):]
    t_ok = np.array([len(c) >= 4 for c in compact[rep_rows]], dtype=bool)
    size = np.bincount(t_code[t_ok], minlength=codes.max() + 1)
    t_ok &= size[t_code] <= max_crowd
    size = np.bincount(t_code[t_ok], minlength=codes.max() + 1)
    right = pd.DataFrame({"code": t_code[t_ok], "t": np.flatnonzero(t_ok)})
    q = np.flatnonzero(size[q_code] > 0)
    q = q[np.argsort(q_code[q], kind="stable")]
    cum = np.cumsum(size[q_code[q]])
    qs, ts, ss = [], [], []
    start = 0
    while start < len(q):
        stop = max(int(np.searchsorted(cum, cum[start] - size[q_code[q[start]]] + chunk_pairs, "right")), start + 1)
        part = pd.DataFrame({"code": q_code[q[start:stop]], "q": q[start:stop]}).merge(right, on="code")
        start = stop
        pq, pt = part["q"].to_numpy(np.int64), part["t"].to_numpy(np.int64)
        sim = pair_cosine(addr, s1_rows[pq], rep_rows[pt])
        top = _rank_within(pq, sim) < k
        qs.append(pq[top])
        ts.append(pt[top])
        ss.append(sim[top])
    if not qs:
        return np.array([], np.int64), np.array([], np.int64), np.array([], np.float32)
    return np.concatenate(qs), np.concatenate(ts), np.concatenate(ss)


def _key_pairs(recs: pd.DataFrame, s1_rows: np.ndarray, tgt_rows: np.ndarray,
               max_block: int) -> tuple[np.ndarray, np.ndarray]:
    """``(query_pos, target_pos)`` pairs sharing an exact key; keys whose target side is larger
    than ``max_block`` are stop-keys and propose nothing."""
    rows = np.concatenate([s1_rows, tgt_rows])
    kdf = pd.DataFrame({"pos": np.arange(len(rows)), "key": _record_keys(recs.iloc[rows])}).explode("key")
    kdf = kdf.dropna(subset=["key"])
    if kdf.empty:
        return np.array([], np.int64), np.array([], np.int64)
    kdf["key"] = pd.factorize(kdf["key"])[0]
    is_q = kdf["pos"].to_numpy() < len(s1_rows)
    left, right = kdf[is_q], kdf[~is_q]
    right = right[right["key"].map(right.groupby("key").size()) <= max_block]
    pairs = left.merge(right, on="key", suffixes=("_q", "_t"))[["pos_q", "pos_t"]].drop_duplicates()
    return pairs["pos_q"].to_numpy(np.int64), pairs["pos_t"].to_numpy(np.int64) - len(s1_rows)


def _dedupe(recs: pd.DataFrame, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group target rows by normalised name + address.

    Returns ``(rep_rows, group)``: one representative row per group and, for every input row,
    the index of its group in ``rep_rows``.
    """
    content = recs["name_norm"].to_numpy()[rows] + "\t" + recs["addr_norm"].to_numpy()[rows]
    group, _ = pd.factorize(content)
    first = pd.Series(np.arange(len(rows))).groupby(group).first().to_numpy()
    return rows[first], group


def _rank_within(groups: np.ndarray, score: np.ndarray) -> np.ndarray:
    """0-based rank of each score inside its group, best first (ties broken by position)."""
    order = np.lexsort((-score, groups))
    g = groups[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    sizes = np.diff(np.r_[starts, len(g)])
    rank = np.empty(len(g), dtype=np.int32)
    rank[order] = np.arange(len(g)) - np.repeat(starts, sizes)
    return rank


def _expand_members(rep_idx: np.ndarray, group: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For pairs pointing at representatives ``rep_idx``, list every member of each group.

    Returns ``(pair_idx, member_pos)``: the input pair each output row came from and the member's
    position among the target rows.
    """
    by_group = np.argsort(group, kind="stable")
    counts = np.bincount(group)
    ptr = np.r_[0, np.cumsum(counts)[:-1]]
    n = counts[rep_idx]
    pair_idx = np.repeat(np.arange(len(rep_idx)), n)
    offset = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    return pair_idx, by_group[np.repeat(ptr[rep_idx], n) + offset]


def _block_target(recs: pd.DataFrame, index: TextIndex, cfg: BlockingConfig, s1_rows: np.ndarray,
                  tgt: int) -> pd.DataFrame:
    tgt_rows = np.flatnonzero(index.source == tgt)
    if len(s1_rows) == 0 or len(tgt_rows) == 0:
        return pd.DataFrame()
    rep_rows, group = _dedupe(recs, tgt_rows)
    n_rep = len(rep_rows)
    ks = {"name_char": cfg.name_char_k, "name_word": cfg.name_word_k,
          "full_char": cfg.full_char_k, "addr_word": cfg.addr_word_k}

    proposals = []  # (blocker, pair key = query_pos * n_rep + rep_pos, similarity)
    for name in KNN_BLOCKERS:
        if ks[name] > 0:
            mat = index.mats[name]
            q, d, s = sparse_topk(mat[s1_rows], mat[rep_rows], ks[name], cfg.min_sim)
            proposals.append((name, q * n_rep + d, s))
    if "embed" in index.dense:
        emb = index.dense["embed"]
        q, d, s = dense_topk(emb[s1_rows], emb[rep_rows], cfg.embed_k, cfg.min_sim)
        proposals.append(("embed", q * n_rep + d, s))
    if cfg.crowd_k > 0:
        q, d, s = _crowd_pairs(recs, index.mats["addr_word"], s1_rows, rep_rows, cfg.crowd_k, cfg.max_crowd)
        proposals.append(("crowd", q * n_rep + d, s))
    q, d = _key_pairs(recs, s1_rows, rep_rows, cfg.max_key_block)
    proposals.append(("key", q * n_rep + d, np.ones(len(q), np.float32)))

    pair, inv = np.unique(np.concatenate([p[1] for p in proposals]), return_inverse=True)
    mask = np.zeros(len(pair), dtype=np.int64)
    sims = {name: np.zeros(len(pair), dtype=np.float32) for name in BLOCKER_BITS}
    start = 0
    for name, keys, s in proposals:
        at = inv[start:start + len(keys)]
        start += len(keys)
        mask[at] |= BLOCKER_BITS[name]
        sims[name][at] = s

    q, d = pair // n_rep, pair % n_rep
    li, ri = s1_rows[q], rep_rows[d]
    score = np.zeros(len(pair), dtype=np.float32)
    for name, weight in cfg.score_weights.items():
        if weight:
            score += weight * pair_cosine(index.mats[name], li, ri)
    rank = _rank_within(q, score)
    keep = rank < cfg.max_per_source if cfg.max_per_source > 0 else np.ones(len(pair), dtype=bool)
    if cfg.max_per_source > 0 and cfg.reverse_k > 0:
        # also keep each target's best S1s, so the matcher and the one-S1-per-record assignment
        # see the Source-1 records competing for it; bounded so no S1 collects every stray target
        keep |= (_rank_within(d, score) < cfg.reverse_k) & (rank < 3 * cfg.max_per_source)

    kept = np.flatnonzero(keep)
    pair_idx, member = _expand_members(d[kept], group)
    src = kept[pair_idx]
    out = pd.DataFrame({"s1_id": index.ids[li[src]], "cand_id": index.ids[tgt_rows[member]]})
    for name in BLOCKER_BITS:
        out[f"blk_{name}"] = sims[name][src]
    out["blk_mask"] = mask[src]
    out["blk_count"] = np.bitwise_count(mask[src].astype(np.uint64)).astype(np.int8)
    out["blk_score"] = score[src]
    out["blk_rank"] = rank[src]
    out["blk_dup"] = np.bincount(group)[d[src]].astype(np.int32)
    out["cand_src"] = tgt
    return out


def generate_candidates(recs: pd.DataFrame, index: TextIndex, cfg: BlockingConfig) -> pd.DataFrame:
    """Candidate pairs of one partition: one row per (s1_id, cand_id).

    Columns: ``s1_id, cand_id, cand_src``, ``blk_<name>`` (similarity, 0 when that blocker did not
    propose the pair), ``blk_mask`` (bit set of proposing blockers), ``blk_count``, ``blk_score``
    (the rule-based ranking score), ``blk_rank`` (0-based rank of the representative among this
    S1's candidates from the same source) and ``blk_dup`` (size of the candidate's
    content-duplicate group).
    """
    s1_rows = np.flatnonzero(index.source == 1)
    frames = [f for tgt in (2, 3) if not (f := _block_target(recs, index, cfg, s1_rows, tgt)).empty]
    if not frames:
        return pd.DataFrame(columns=["s1_id", "cand_id", "cand_src"])
    cands = pd.concat(frames, ignore_index=True)
    log.info("blocking: %s candidate pairs for %s S1 records (%.1f per record)", f"{len(cands):,}",
             f"{len(s1_rows):,}", len(cands) / max(len(s1_rows), 1))
    return cands
