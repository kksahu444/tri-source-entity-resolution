"""D8 count model: how many true matches an S1 has inside its pruned candidate list.

K classes [0, 1, ..., K-2, K-1+]; K = 4 is the original (0, 1, 2, 3+). choose_k() picks the smallest K
whose open-ended class holds <= 5% of dev entities (Appendix A guidance). Features come from calibrated
out-of-fold pair probabilities (cross-fitting), never the country. K-class LightGBM, 5-fold out-of-fold
by the same folds, temperature-scaled on the OOF logits.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

PARAMS = {"objective": "multiclass", "learning_rate": 0.05, "num_leaves": 63,
          "min_data_in_leaf": 200, "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1,
          "verbose": -1, "num_threads": 0}
ROUNDS = 400


def counts_in_list(s1_idx: np.ndarray, y: np.ndarray, n_s1: int) -> np.ndarray:
    return np.bincount(s1_idx[y == 1], minlength=n_s1)


def choose_k(counts: np.ndarray, open_share: float = 0.05, k_min: int = 4, k_max: int = 12) -> int:
    """Smallest K (>= k_min) with share(count >= K-1) <= open_share."""
    for k in range(k_min, k_max + 1):
        if np.mean(counts >= k - 1) <= open_share:
            return k
    return k_max


def entity_features(s1_idx: np.ndarray, p: np.ndarray, src3: np.ndarray, addr_empty: np.ndarray,
                    n_s1: int, s1_parts: np.ndarray, s1_crowd: np.ndarray) -> pd.DataFrame:
    """One row per S1 (index 0..n_s1-1) from its candidates' calibrated probabilities."""
    order = np.lexsort((-p, s1_idx))
    s, pp = s1_idx[order], p[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    sizes = np.diff(np.r_[starts, len(s)])
    rank = np.arange(len(s)) - np.repeat(starts, sizes)
    top = np.zeros((n_s1, 5), np.float32)
    m = rank < 5
    top[s[m], rank[m]] = pp[m]
    f = pd.DataFrame(top, columns=[f"p{i + 1}" for i in range(5)])
    bc = lambda w: np.bincount(s1_idx, weights=w, minlength=n_s1).astype(np.float32)
    f["gap12"] = f["p1"] - f["p2"]
    f["sum"] = bc(p)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    f["entropy"] = bc(-(pc * np.log(pc) + (1 - pc) * np.log(1 - pc)))
    f["n_05"] = bc((p > 0.5).astype(float))
    f["n_02"] = bc((p > 0.2).astype(float))
    f["n_cand"] = bc(np.ones_like(p))
    for k, name in ((0, "s2"), (1, "s3")):
        msk = src3 == k
        f[f"sum_{name}"] = np.bincount(s1_idx[msk], weights=p[msk], minlength=n_s1)
        mx = np.zeros(n_s1)
        np.maximum.at(mx, s1_idx[msk], p[msk])
        f[f"max_{name}"] = mx
    f["share_addr_empty"] = bc(addr_empty.astype(float)) / np.maximum(f["n_cand"], 1)
    f["s1_parts"] = s1_parts.astype(np.float32)
    f["s1_crowd"] = s1_crowd.astype(np.float32)
    return f


def target(s1_idx: np.ndarray, y: np.ndarray, n_s1: int, k: int = 4) -> np.ndarray:
    return np.minimum(counts_in_list(s1_idx, y, n_s1), k - 1)


def _softmax(z, t=1.0):
    z = z / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_oof(X: pd.DataFrame, y: np.ndarray, fold: np.ndarray, seed: int, k: int = 4):
    """Returns (OOF class probabilities after temperature scaling, models, temperature)."""
    logits = np.zeros((len(y), k))
    models = []
    for f in sorted(set(fold.tolist()) - {-1}):
        tr, va = fold != f, fold == f
        m = lgb.train({**PARAMS, "num_class": k, "seed": seed}, lgb.Dataset(X[tr], y[tr]), ROUNDS)
        logits[va] = m.predict(X[va], raw_score=True)
        models.append(m)
    nll = lambda t: -np.mean(np.log(_softmax(logits, t)[np.arange(len(y)), y] + 1e-12))
    temp = float(minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded").x)
    return _softmax(logits, temp), models, temp


def predict(models, temp: float, X: pd.DataFrame) -> np.ndarray:
    z = np.mean([m.predict(X, raw_score=True) for m in models], axis=0)
    return _softmax(z, temp)
