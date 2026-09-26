"""D8 day-1 rule: keep the top candidate if p >= tau1, each further candidate if p >= tau2.

Tuned on a 0.01 grid out-of-fold; the chosen point is the centre of the region within `tol` of the
best score (not the single peak).
"""
from __future__ import annotations

import numpy as np

from ..metrics import entity_f

GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def rank_within(s1_idx: np.ndarray, p: np.ndarray) -> np.ndarray:
    order = np.lexsort((-p, s1_idx))
    ks = s1_idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ks)) + 1]
    sizes = np.diff(np.r_[starts, len(ks)])
    rank = np.empty(len(p), np.int32)
    rank[order] = np.arange(len(ks)) - np.repeat(starts, sizes) + 1
    return rank


def select(s1_idx: np.ndarray, p: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    rank = rank_within(s1_idx, p)
    top_p = np.zeros(s1_idx.max() + 1 if len(s1_idx) else 0)
    top_p[s1_idx[rank == 1]] = p[rank == 1]
    first_ok = top_p[s1_idx] >= tau1
    return first_ok & ((rank == 1) | (p >= tau2))


def score_grid(s1_idx, p, y, n_true):
    """Macro F0.5 for every (tau1, tau2) on GRID; returns matrix [len(GRID) tau1, len(GRID) tau2]."""
    n = len(n_true)
    rank = rank_within(s1_idx, p)
    top = rank == 1
    p1 = np.zeros(n)
    y1 = np.zeros(n)
    has = np.zeros(n, bool)
    p1[s1_idx[top]], y1[s1_idx[top]], has[s1_idx[top]] = p[top], y[top], True
    rest = ~top
    rs, rp, ry = s1_idx[rest], p[rest], y[rest]
    base_empty = (n_true == 0).astype(float)           # score when nothing is predicted
    order = np.argsort(-p1, kind="stable")              # S1s by top probability, descending
    p1_sorted = p1[order]
    out = np.zeros((len(GRID), len(GRID)))
    for j, t2 in enumerate(GRID):
        m = rp >= t2
        k2 = np.bincount(rs[m], minlength=n)
        tp2 = np.bincount(rs[m & (ry == 1)], minlength=n)
        f_on = entity_f(1 + k2, y1 + tp2, n_true)
        f_on = np.where(has, f_on, base_empty)
        gain = (f_on - base_empty)[order]
        cum = np.r_[0.0, np.cumsum(gain)]
        for i, t1 in enumerate(GRID):
            k = np.searchsorted(-p1_sorted, -t1, side="right")   # number of S1 with p1 >= t1
            out[i, j] = (base_empty.sum() + cum[k]) / n
    return out


def tune(s1_idx, p, y, n_true, tol=0.002):
    sc = score_grid(s1_idx, p, y, n_true)
    best = sc.max()
    ii, jj = np.nonzero(sc >= best - tol)
    t1, t2 = float(GRID[int(round(np.median(ii)))]), float(GRID[int(round(np.median(jj)))])
    i, j = np.searchsorted(GRID, t1), np.searchsorted(GRID, t2)
    if sc[i, j] < best - tol:
        i, j = np.unravel_index(sc.argmax(), sc.shape)
        t1, t2 = float(GRID[i]), float(GRID[j])
    pi, pj = np.unravel_index(sc.argmax(), sc.shape)
    return {"tau1": t1, "tau2": t2, "score": float(sc[i, j]), "peak": float(best),
            "peak_tau1": float(GRID[pi]), "peak_tau2": float(GRID[pj])}
