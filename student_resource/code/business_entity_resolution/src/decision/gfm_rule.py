"""D8 day-2 rule: GFM per S1 (Appendix A), with independence or the count model.

select() runs the numba batch port (gfm_fast); select_reference() loops over Appendix A's
functions directly and is kept for tests.
"""
from __future__ import annotations

import numpy as np

from .f05_decision_rule import P_count_model, P_independent, expand_count_classes, gfm


def _layout(s1_idx, p, max_m):
    order = np.lexsort((-p, s1_idx))
    s = s1_idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1] if len(s) else np.zeros(0, np.int64)
    ends = np.r_[starts[1:], len(s)] if len(s) else np.zeros(0, np.int64)
    ends = np.minimum(ends, starts + max_m)
    return order, s, starts.astype(np.int64), ends.astype(np.int64)


def select(s1_idx: np.ndarray, p: np.ndarray, count_probs: np.ndarray | None = None, max_m: int = 30,
           skip_below: float = 1e-3, beta: float = 0.5):
    """Returns (selected mask per pair, {s1: expected F})."""
    from .gfm_fast import gfm_batch
    sel = np.zeros(len(p), bool)
    if len(p) == 0:
        return sel, {}
    order, s, starts, ends = _layout(s1_idx, p, max_m)
    ents = s[starts]
    cp = count_probs[ents].astype(np.float64) if count_probs is not None else np.zeros((len(ents), 4))
    m, ef = gfm_batch(starts, ends, p[order].astype(np.float64), cp, count_probs is not None,
                      float(skip_below), beta * beta)
    sel[order[m]] = True
    return sel, dict(zip(ents.tolist(), ef.tolist()))


def select_reference(s1_idx, p, count_probs=None, max_m=30, skip_below=1e-3):
    sel = np.zeros(len(p), bool)
    order, s, starts, ends = _layout(s1_idx, p, max_m)
    exp_f = {}
    for a, b in zip(starts, ends):
        idx = order[a:b]
        probs = p[idx]
        e1 = s[a]
        if probs[0] < skip_below:
            exp_f[e1] = float(np.prod(1 - probs))
            continue
        if count_probs is None:
            P, p0 = P_independent(probs)
        else:
            P, p0 = P_count_model(probs, expand_count_classes(count_probs[e1], probs))
        chosen, v = gfm(list(range(len(idx))), P, p0)
        sel[idx[chosen]] = True
        exp_f[e1] = v
    return sel, exp_f
