"""D8 day-2 rule: GFM per S1 (Appendix A), with independence or the count model."""
from __future__ import annotations

import numpy as np

from .f05_decision_rule import P_count_model, P_independent, expand_count_classes, gfm


def select(s1_idx: np.ndarray, p: np.ndarray, count_probs: np.ndarray | None = None, max_m: int = 30,
           skip_below: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Returns (selected mask per pair, expected F per S1 for S1s that have candidates).

    count_probs: [n_s1, 4] class probabilities, or None for the independence model.
    S1s whose best candidate is below skip_below are predicted empty without running GFM
    (their expected F of the empty set is >= 1 - m * skip_below).
    """
    sel = np.zeros(len(p), bool)
    order = np.lexsort((-p, s1_idx))
    s = s1_idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    ends = np.r_[starts[1:], len(s)]
    exp_f = {}
    for a, b in zip(starts, ends):
        idx = order[a:b][:max_m]
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
