"""Candidate generation (blocking).

Several cheap, complementary blockers are run for every Source-1 record against Source 2 and
Source 3 separately, and their outputs are unioned:

==============  ===============================================  ===============================
blocker         representation                                   catches
==============  ===============================================  ===============================
name_char       char 2-4gram TF-IDF of the legal-free name       typos, suffix/abbrev. changes
name_word       word TF-IDF of name tokens + phonetic skeletons  transliterations, word reorder
full_char       char 3-4gram TF-IDF of name + address            generic look-alikes
addr_word       word 1-2gram TF-IDF of the address core          DBA / trade-name differences
key             exact keys (postcode+name skeleton, acronym,     abbreviations like IBM, exact
                full skeleton, exact address)                    duplicates the kNN may rank low
embed           optional multilingual sentence embedding         semantic / cross-lingual
==============  ===============================================  ===============================

TF-IDF models are fitted on the records of the split being processed (unsupervised, no labels and
no external data), which also adapts the vocabulary to the unseen France records at test time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import BlockingConfig

log = logging.getLogger(__name__)

BLOCKER_BITS = {"name_char": 1, "name_word": 2, "full_char": 4, "addr_word": 8, "key": 16, "embed": 32}
KNN_BLOCKERS = ("name_char", "name_word", "full_char", "addr_word")


@dataclass
class TextIndex:
    """Vector representations of every record of a split, aligned by row position."""

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
    except ValueError:  # empty vocabulary, e.g. every address blank
        return sp.csr_matrix((len(texts), 1), dtype=np.float32)


def _idf(docs: list[str]) -> dict[str, float]:
    """Smoothed inverse document frequency of whitespace tokens."""
    df: dict[str, int] = {}
    for doc in docs:
        for tok in set(doc.split()):
            df[tok] = df.get(tok, 0) + 1
    n = len(docs)
    return {tok: float(np.log((n + 1) / (cnt + 1)) + 1.0) for tok, cnt in df.items()}


def build_text_index(recs: pd.DataFrame, cfg: BlockingConfig | None = None) -> TextIndex:
    """Fit all text representations on the normalised records of one split."""
    name_core = recs["name_core"].tolist()
    addr_core = recs["addr_core"].tolist()
    name_words = [f"{c} " + " ".join(f"~{s}" for s in k.split()) for c, k in zip(name_core, recs["name_skel"])]
    full = [f"{n} | {a}" for n, a in zip(name_core, addr_core)]
    mats = {
        "name_char": _tfidf(name_core, analyzer="char_wb", ngram_range=(2, 4)),
        "name_word": _tfidf(name_words, analyzer=str.split),
        "full_char": _tfidf(full, analyzer="char_wb", ngram_range=(3, 4)),
        "addr_word": _tfidf(addr_core, analyzer="word", token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2)),
        "addr_char": _tfidf(addr_core, analyzer="char_wb", ngram_range=(3, 4)),
    }
    ids = recs["entity_id"].to_numpy()
    index = TextIndex(
        ids=ids,
        pos=pd.Series(np.arange(len(ids)), index=ids),
        source=recs["source"].to_numpy(),
        mats=mats,
        name_idf=_idf(name_core),
        addr_idf=_idf(addr_core),
    )
    if cfg is not None and cfg.use_embeddings:
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
    """Top-``k`` cosine neighbours in ``data`` for every row of ``query`` (chunked sparse product).

    Returns ``(query_row, data_row, similarity)`` arrays.
    """
    if query.shape[0] == 0 or data.shape[0] == 0 or k <= 0:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    data_t = data.T.tocsr()
    chunk = int(max(32, min(4096, 2e7 // max(data.shape[0], 1))))  # bound the product's memory
    qs, ds, ss = [], [], []
    for start in range(0, query.shape[0], chunk):
        prod = (query[start:start + chunk] @ data_t).tocsr()
        for i in range(prod.shape[0]):
            lo, hi = prod.indptr[i], prod.indptr[i + 1]
            if hi == lo:
                continue
            vals, cols = prod.data[lo:hi], prod.indices[lo:hi]
            if hi - lo > k:
                top = np.argpartition(-vals, k)[:k]
                vals, cols = vals[top], cols[top]
            keep = vals >= min_sim
            qs.append(np.full(int(keep.sum()), start + i))
            ds.append(cols[keep])
            ss.append(vals[keep])
    if not qs:
        return np.array([], int), np.array([], int), np.array([], np.float32)
    return np.concatenate(qs), np.concatenate(ds), np.concatenate(ss)


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


def _record_keys(recs: pd.DataFrame) -> list[list[str]]:
    """Exact blocking keys per record (empty-safe)."""
    keys_all = []
    for core, skel, acr, pcs, addr in zip(
        recs["name_core"], recs["name_skel"], recs["name_acronym"], recs["addr_postcodes"], recs["addr_core"]
    ):
        keys = []
        first_skel = skel.split()[0] if skel else ""
        compact = core.replace(" ", "")
        for pc in pcs.split():
            if first_skel:
                keys.append(f"pc:{pc}|{first_skel}")
            if acr:
                keys.append(f"acr:{acr}|{pc}")
            if 2 <= len(compact) <= 5 and " " not in core:  # the name itself looks like an acronym
                keys.append(f"acr:{compact}|{pc}")
        if len(skel) >= 6:
            keys.append(f"skel:{skel}")
        if len(addr) >= 12:
            keys.append(f"addr:{addr}")
        keys_all.append(keys)
    return keys_all


def _key_block(recs: pd.DataFrame, max_block: int) -> pd.DataFrame:
    kdf = pd.DataFrame({"entity_id": recs["entity_id"].to_numpy(), "source": recs["source"].to_numpy(),
                        "key": _record_keys(recs)}).explode("key").dropna(subset=["key"])
    if kdf.empty:
        return pd.DataFrame(columns=["s1_id", "cand_id", "sim"])
    sizes = kdf.groupby(["key", "source"]).size().groupby(level="key").max()
    kdf = kdf[kdf["key"].map(sizes) <= max_block]
    left = kdf[kdf["source"] == 1][["key", "entity_id"]].rename(columns={"entity_id": "s1_id"})
    right = kdf[kdf["source"] != 1][["key", "entity_id"]].rename(columns={"entity_id": "cand_id"})
    pairs = left.merge(right, on="key")[["s1_id", "cand_id"]].drop_duplicates()
    pairs["sim"] = 1.0
    return pairs


def _proposed_by(long: pd.DataFrame, wide: pd.DataFrame, blocker: str) -> np.ndarray:
    """Boolean per row of ``wide``: did ``blocker`` propose this pair (even with similarity 0)?"""
    sub = long.loc[long["blocker"] == blocker, ["s1_id", "cand_id"]].drop_duplicates()
    if sub.empty:
        return np.zeros(len(wide), dtype=bool)
    proposed = pd.MultiIndex.from_frame(sub)
    return wide.index.isin(proposed)


def generate_candidates(recs: pd.DataFrame, index: TextIndex, cfg: BlockingConfig) -> pd.DataFrame:
    """Union of all blockers -> one row per (s1_id, cand_id) with per-blocker similarities.

    Columns: ``s1_id, cand_id, cand_src, blk_<name>`` (similarity, 0 when that blocker did not
    propose the pair), ``blk_mask`` (bit set of proposing blockers) and ``blk_count``.
    """
    s1_rows = np.flatnonzero(index.source == 1)
    frames = []
    ks = {"name_char": cfg.name_char_k, "name_word": cfg.name_word_k,
          "full_char": cfg.full_char_k, "addr_word": cfg.addr_word_k}
    for tgt in (2, 3):
        tgt_rows = np.flatnonzero(index.source == tgt)
        for name in KNN_BLOCKERS:
            mat = index.mats[name]
            q, d, s = sparse_topk(mat[s1_rows], mat[tgt_rows], ks[name], cfg.min_sim)
            frames.append(pd.DataFrame({"s1_id": index.ids[s1_rows[q]], "cand_id": index.ids[tgt_rows[d]],
                                        "sim": s, "blocker": name}))
        if "embed" in index.dense:
            emb = index.dense["embed"]
            q, d, s = dense_topk(emb[s1_rows], emb[tgt_rows], cfg.embed_k, cfg.min_sim)
            frames.append(pd.DataFrame({"s1_id": index.ids[s1_rows[q]], "cand_id": index.ids[tgt_rows[d]],
                                        "sim": s, "blocker": "embed"}))
    key_pairs = _key_block(recs, cfg.max_key_block)
    key_pairs["blocker"] = "key"
    frames.append(key_pairs)

    long = pd.concat(frames, ignore_index=True)
    long["sim"] = long["sim"].astype(np.float32)
    wide = long.groupby(["s1_id", "cand_id", "blocker"])["sim"].max().unstack("blocker", fill_value=0.0)
    for name in BLOCKER_BITS:
        if name not in wide.columns:
            wide[name] = 0.0
    mask = np.zeros(len(wide), dtype=np.int64)
    for name, bit in BLOCKER_BITS.items():
        mask |= np.where(_proposed_by(long, wide, name), bit, 0)
    cands = wide[list(BLOCKER_BITS)].add_prefix("blk_").reset_index()
    cands.columns.name = None
    cands["blk_mask"] = mask
    cands["blk_count"] = [bin(m).count("1") for m in mask]
    cands["cand_src"] = cands["cand_id"].str[1].astype(int)

    if cfg.max_candidates and cfg.max_candidates > 0:
        score = cands[[f"blk_{b}" for b in ("name_char", "full_char", "name_word")]].max(axis=1)
        cands = (cands.assign(_score=score).sort_values(["s1_id", "_score"], ascending=[True, False])
                 .groupby("s1_id", sort=False).head(cfg.max_candidates).drop(columns="_score"))
    cands = cands.sort_values(["s1_id", "cand_id"]).reset_index(drop=True)
    log.info("blocking: %s candidate pairs for %s S1 records (%.1f per record)", f"{len(cands):,}",
             f"{len(s1_rows):,}", len(cands) / max(len(s1_rows), 1))
    return cands

