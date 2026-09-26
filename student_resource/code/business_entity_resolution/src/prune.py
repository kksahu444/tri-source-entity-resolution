"""D3 pruner: LightGBM on cheap features, out-of-fold on the dev universe.

Per S1 keep the top N, N = smallest value with PC(N) >= PC(union) - pc_tolerance (per country;
France uses the larger of the US and India values), plus every record's top `reverse_keep` S1s.
The pruned set is exactly what candidate_pairs.tsv holds and what the final model scores.
"""
from __future__ import annotations

import numpy as np

from . import gbm
from .features import _group_stats

PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "max_bin": 63, "verbose": -1, "num_threads": 0}
ROUNDS = 300


def train_oof(X, y, fold, seed, neg_keep=0.3, cfg=None):
    """Returns (oof scores, list of fold models). Negatives subsampled for training only."""
    rng = np.random.default_rng(seed)
    oof = np.zeros(len(y), np.float32)
    models = []
    for f in sorted(set(fold.tolist()) - {-1}):
        tr = (fold != f) & ((y == 1) | (rng.random(len(y)) < neg_keep))
        va = fold == f
        m = gbm.train_binary(cfg or {}, X[tr], y[tr], ROUNDS, seed=seed, lgb_params=PARAMS)
        oof[va] = m.predict(X[va])
        models.append(m)
    return oof, models


def predict(models, X):
    return np.mean([gbm.predict(m, X) for m in models], axis=0).astype(np.float32)


def choose_n(qi, score, y, n_true_total, pc_tol, n_max=60):
    """Smallest N with PC(N) >= PC(union) - tol; PC relative to all true pairs of the queries."""
    _, _, rank = _group_stats(qi, score)
    pos_rank = rank[y == 1]
    pc_union = len(pos_rank) / n_true_total
    counts = np.bincount(np.minimum(pos_rank, n_max + 1), minlength=n_max + 2)
    cum = np.cumsum(counts[1:n_max + 1]) / n_true_total
    ok = np.flatnonzero(cum >= pc_union - pc_tol)
    n = int(ok[0] + 1) if len(ok) else n_max
    return n, pc_union, {int(k): round(float(cum[k - 1]), 5) for k in (5, 10, 15, 20, 25, 30, 40, 50, 60) if k <= n_max}


def keep_mask(qi, ri, score, n, reverse_keep):
    _, _, rank = _group_stats(qi, score)
    _, _, rrank = _group_stats(ri, score)
    return (rank <= n) | (rrank <= reverse_keep)
