"""numba port of Appendix A's GFM (P_independent / P_count_model / expand_count_classes / gfm),
batched over entities. Same arithmetic as f05_decision_rule.py; tests/test_gfm_count.py checks
the chosen sets against the reference."""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _pb(p):
    m = len(p)
    dist = np.zeros(m + 1)
    dist[0] = 1.0
    for j in range(m):
        for s in range(m, 0, -1):
            dist[s] = dist[s] * (1 - p[j]) + dist[s - 1] * p[j]
        dist[0] *= (1 - p[j])
    return dist


@njit(cache=True)
def _loo(p, i):
    out = np.empty(len(p) - 1)
    k = 0
    for j in range(len(p)):
        if j != i:
            out[k] = p[j]
            k += 1
    return out


@njit(cache=True)
def _p_independent(p):
    m = len(p)
    P = np.zeros((m, m))
    p0 = 1.0
    for i in range(m):
        d = _pb(_loo(p, i))
        for s in range(m):
            P[i, s] = p[i] * d[s]
        p0 *= (1 - p[i])
    return P, p0


@njit(cache=True)
def _expand(pc, p):
    """K-class count output [P0..P_{K-2}, P_{K-1}+] -> Pcount(s), s = 0..m (Appendix A semantics:
    open class spread over s >= K-1 by PB(s); mass on impossible counts s > m goes to s = m)."""
    m = len(p)
    top = len(pc) - 1
    out = np.zeros(m + 1)
    for s in range(min(top, m + 1)):
        out[s] = pc[s]
    if m >= top:
        pb = _pb(p)[top:]
        tot = pb.sum()
        if tot > 0:
            for s in range(len(pb)):
                out[top + s] += pc[top] * pb[s] / tot
        else:
            out[top] += pc[top]
    else:
        for s in range(m + 1, len(pc)):
            out[m] += pc[s]
    return out / out.sum()


@njit(cache=True)
def _p_count(p, cd_in):
    m = len(p)
    cd = cd_in[:m + 1].copy()
    cd = cd / cd.sum()
    pb = _pb(p)
    P = np.zeros((m, m))
    for i in range(m):
        loo = _pb(_loo(p, i))
        for s in range(1, m + 1):
            r = loo[s - 1] / pb[s] if pb[s] > 0 else 0.0
            P[i, s - 1] = cd[s] * p[i] * r
    return P, cd[0]


@njit(cache=True)
def _gfm(P, p0, beta2):
    m = P.shape[0]
    best_val, best_k = p0, 0
    best_top = np.zeros(0, np.int64)
    w = np.empty(m)
    for k in range(1, m + 1):
        for s in range(m):
            w[s] = (1 + beta2) / (beta2 * (s + 1) + k)
        delta = P @ w
        top = np.argsort(-delta, kind="mergesort")[:k]
        val = 0.0
        for j in top:
            val += delta[j]
        if val > best_val + 1e-12:
            best_val, best_k, best_top = val, k, top.copy()
    return best_top, best_val


@njit(parallel=True, cache=True)
def gfm_batch(starts, ends, p_sorted, count_probs, use_count, skip_below, beta2):
    """Entities e own p_sorted[starts[e]:ends[e]] (descending). Returns a 0/1 mask aligned with
    p_sorted and the expected F per entity."""
    sel = np.zeros(len(p_sorted), np.bool_)
    ef = np.zeros(len(starts))
    for e in prange(len(starts)):
        a, b = starts[e], ends[e]
        probs = p_sorted[a:b].copy()
        if probs[0] < skip_below:
            v = 1.0
            for x in probs:
                v *= (1 - x)
            ef[e] = v
            continue
        if use_count:
            pc = np.minimum(np.maximum(probs, 1e-6), 1 - 1e-6)
            P, p0 = _p_count(pc, _expand(count_probs[e], pc))
        else:
            P, p0 = _p_independent(probs)
        top, v = _gfm(P, p0, beta2)
        for j in top:
            sel[a + j] = True
        ef[e] = v
    return sel, ef
