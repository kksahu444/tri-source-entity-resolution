"""Vectorised macro F0.5 (identical to f05() of Appendix A, per S1 entity) with bootstrap intervals."""
from __future__ import annotations

import numpy as np


def entity_f(n_pred: np.ndarray, tp: np.ndarray, n_true: np.ndarray, beta: float = 0.5) -> np.ndarray:
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tp > 0, (1 + b2) * tp / (b2 * n_true + n_pred), 0.0)
    return np.where((n_pred == 0) & (n_true == 0), 1.0, f)


def per_entity(s1_idx: np.ndarray, selected: np.ndarray, y: np.ndarray, n_true: np.ndarray) -> np.ndarray:
    """s1_idx/selected/y per candidate pair; n_true per S1 (all true matches, retrieved or not)."""
    n = len(n_true)
    n_pred = np.bincount(s1_idx[selected], minlength=n)
    tp = np.bincount(s1_idx[selected & (y == 1)], minlength=n)
    return entity_f(n_pred, tp, n_true)


def bootstrap_ci(f: np.ndarray, n_boot: int = 1000, seed: int = 0, batch: int = 50) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    for b in range(0, n_boot, batch):
        idx = rng.integers(0, len(f), size=(min(batch, n_boot - b), len(f)))
        means.append(f[idx].mean(axis=1))
    m = np.concatenate(means)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summary(f: np.ndarray, n_true: np.ndarray, country: np.ndarray, n_boot: int = 1000) -> dict:
    lo, hi = bootstrap_ci(f, n_boot)
    out = {"oof_f05": round(float(f.mean()), 5), "oof_f05_ci": f"[{lo:.5f}, {hi:.5f}]",
           "oof_f05_singleton": round(float(f[n_true == 0].mean()), 5),
           "oof_f05_nonsingleton": round(float(f[n_true > 0].mean()), 5)}
    for c in np.unique(country):
        m = country == c
        lo, hi = bootstrap_ci(f[m], n_boot // 4)
        out[f"oof_f05_{c}"] = round(float(f[m].mean()), 5)
        out[f"oof_f05_{c}_ci"] = f"[{lo:.5f}, {hi:.5f}]"
    return out
