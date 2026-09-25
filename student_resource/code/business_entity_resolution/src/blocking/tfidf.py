"""C1: joint name + address TF-IDF with exact top-k (sparse_dot_topn, multi-threaded).

Record vector = [sqrt(.5) * name char-3-gram TF-IDF, sqrt(.5) * address word TF-IDF], each part
L2-normalised, so the retrieval score is 0.5 * cos_name + 0.5 * cos_addr. Vectorisers are fitted on
the index side (S2 u S3 of one country), which is allowed test-set text statistics (C-3).
"""
from __future__ import annotations

import os

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # pragma: no cover
    sp_matmul_topn = None

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = None

W = np.float32(np.sqrt(0.5))
N_THREADS = max(1, (os.cpu_count() or 2) - 1)


class TfidfSpace:
    def __init__(self):
        self.vn = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, min_df=2,
                                  max_df=0.05, dtype=np.float32)
        self.va = TfidfVectorizer(token_pattern=r"\S+", lowercase=False, sublinear_tf=True, min_df=2,
                                  max_df=0.02, dtype=np.float32)

    def fit(self, names, addrs):
        self.vn.fit(names)
        self.va.fit(addrs)
        return self

    def transform(self, names, addrs):
        n = self.vn.transform(names).tocsr()
        a = self.va.transform(addrs).tocsr()
        n.sort_indices()
        a.sort_indices()
        return n, a


def combined(n: sp.csr_matrix, a: sp.csr_matrix) -> sp.csr_matrix:
    c = sp.hstack([n * W, a * W], format="csr", dtype=np.float32)
    c.sort_indices()
    return c


def topk(q: sp.csr_matrix, index_t: sp.csr_matrix, k: int, threshold: float = 0.05, chunk: int = 50_000):
    """Exact top-k by sparse products in query chunks. ``index_t`` is the transposed index (d x m)."""
    qs, rs, ss = [], [], []
    for s in range(0, q.shape[0], chunk):
        c = sp_matmul_topn(q[s:s + chunk], index_t, top_n=k, threshold=threshold, n_threads=N_THREADS).tocoo()
        qs.append(c.row.astype(np.int32) + s)
        rs.append(c.col.astype(np.int32))
        ss.append(c.data.astype(np.float32))
    return np.concatenate(qs), np.concatenate(rs), np.concatenate(ss)


if njit is not None:
    @njit(parallel=True, cache=True)
    def _pair_dot(xp, xi, xd, yp, yi, yd, qi, ri, out):
        for t in prange(len(qi)):
            a0, a1 = xp[qi[t]], xp[qi[t] + 1]
            b0, b1 = yp[ri[t]], yp[ri[t] + 1]
            s = 0.0
            while a0 < a1 and b0 < b1:
                ia, ib = xi[a0], yi[b0]
                if ia == ib:
                    s += xd[a0] * yd[b0]
                    a0 += 1
                    b0 += 1
                elif ia < ib:
                    a0 += 1
                else:
                    b0 += 1
            out[t] = s


def pair_cos(x: sp.csr_matrix, y: sp.csr_matrix, qi: np.ndarray, ri: np.ndarray) -> np.ndarray:
    """Row-wise dot products x[qi[t]] . y[ri[t]] (rows are L2-normalised, so this is the cosine)."""
    out = np.zeros(len(qi), dtype=np.float32)
    if len(qi) == 0:
        return out
    if njit is not None:
        _pair_dot(x.indptr, x.indices, x.data, y.indptr, y.indices, y.data,
                  qi.astype(np.int64), ri.astype(np.int64), out)
        return out
    for s in range(0, len(qi), 1_000_000):
        e = s + 1_000_000
        out[s:e] = np.asarray(x[qi[s:e]].multiply(y[ri[s:e]]).sum(1)).ravel()
    return out
