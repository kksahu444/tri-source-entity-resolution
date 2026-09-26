"""D7: one S1 per S2/S3 record. No clustering, no transitive links."""
from __future__ import annotations

import numpy as np


def hard_assign(ri: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Mask keeping, for every record, only its highest-probability S1 (ties: first)."""
    order = np.lexsort((-p, ri))
    first = np.r_[True, np.diff(ri[order]) != 0]
    keep = np.zeros(len(ri), bool)
    keep[order[first]] = True
    return keep


def soft_assign(ri: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Rescale each record's probabilities across its S1s so they sum to at most 1."""
    tot = np.bincount(ri, weights=p)
    return np.where(tot[ri] > 1, p / tot[ri], p)
