"""D6 calibration: isotonic regression on out-of-fold stage probabilities (US + India pooled)."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit(p: np.ndarray, y: np.ndarray, max_n: int = 3_000_000, seed: int = 0) -> IsotonicRegression:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(p), min(len(p), max_n), replace=False)
    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
    iso.fit(p[idx], y[idx])
    return iso


def ece(p: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    b = np.minimum((p * bins).astype(int), bins - 1)
    n = np.bincount(b, minlength=bins)
    conf = np.bincount(b, weights=p, minlength=bins)
    acc = np.bincount(b, weights=y, minlength=bins)
    m = n > 0
    return float(np.sum(np.abs(conf[m] - acc[m])) / len(p))
