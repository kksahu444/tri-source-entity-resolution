"""From pair probabilities to per-entity match lists, optimised for macro F0.5.

Two ideas matter for this metric:

1. **Exclusivity** - Source 1 is deduplicated, so a Source-2/3 record belongs to at most one
   Source-1 entity. Keeping only the best-scoring Source-1 entity per candidate removes many
   false merges (look-alikes) at almost no recall cost.
2. **Per-entity decision** - a flat probability ``threshold``, or ``expected_f``: for each
   entity pick the top-k candidates (k = 0 allowed) that maximise the expected F0.5 under the
   model's probabilities. k = 0 is chosen when "this is a singleton" is the likelier story,
   which is exactly where F0.5 pays a full 1.0.

Both are tuned on out-of-fold probabilities (never on the test set).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import BETA, macro_f_from_selection

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)
FLOOR_GRID = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])


def enforce_exclusive(scored: pd.DataFrame) -> pd.DataFrame:
    """Keep, for every candidate id, only the Source-1 row(s) with the highest probability."""
    best = scored.groupby("cand_id")["p"].transform("max")
    return scored[scored["p"] >= best]


def _expected_f_select(scored: pd.DataFrame, floor: float, beta: float) -> pd.DataFrame:
    """Per Source-1 entity choose the top-k maximising an expected-F approximation.

    E[F | top-k] ~= (1+b2) * sum_{i<=k} p_i / (k + b2 * sum_all p_i)   (ratio of expectations)
    E[F | k=0]   =  prod_i (1 - p_i)                                     (probability of a singleton)
    """
    if scored.empty:
        return scored
    b2 = beta * beta
    df = scored.sort_values(["s1_id", "p"], ascending=[True, False], kind="mergesort")
    s1 = df["s1_id"].to_numpy()
    p = df["p"].to_numpy()
    grp = df.groupby("s1_id", sort=False)["p"]
    k = grp.cumcount().to_numpy() + 1
    e_k = (1 + b2) * grp.cumsum().to_numpy() / (k + b2 * grp.transform("sum").to_numpy())
    log_q = pd.Series(np.log1p(-np.clip(p, 0.0, 1 - 1e-9)))
    e_0 = np.exp(log_q.groupby(s1).transform("sum").to_numpy())

    tab = pd.DataFrame({"s1_id": s1, "k": k, "e_k": e_k, "e_0": e_0})
    best = tab.loc[tab.groupby("s1_id", sort=False)["e_k"].idxmax()].set_index("s1_id")
    k_star = np.where(best["e_k"] > best["e_0"], best["k"], 0)
    k_star = pd.Series(k_star, index=best.index)
    keep = (k <= k_star.reindex(s1).to_numpy()) & (p >= floor)
    return df[keep]


def select_matches(
    scored: pd.DataFrame, strategy: str, threshold: float, exclusive: bool, beta: float = BETA
) -> pd.DataFrame:
    """Rows of ``scored`` (columns ``s1_id, cand_id, p``) that are predicted matches."""
    df = enforce_exclusive(scored) if exclusive else scored
    if strategy == "threshold":
        return df[df["p"] >= threshold]
    if strategy == "expected_f":
        return _expected_f_select(df, threshold, beta)
    raise ValueError(f"unknown decision strategy: {strategy}")


def tune_decision(scored: pd.DataFrame, n_true: pd.Series, beta: float = BETA) -> tuple[dict, pd.DataFrame]:
    """Grid-search strategy / threshold / exclusivity on labelled OOF scores.

    ``scored`` needs ``s1_id, cand_id, p, label``; ``n_true`` = true-match count for every
    Source-1 id being evaluated (entities without candidates included).
    """
    rows = []
    for exclusive in (True, False):
        for strategy, grid in (("threshold", THRESHOLD_GRID), ("expected_f", FLOOR_GRID)):
            for t in grid:
                sel = select_matches(scored, strategy, float(t), exclusive, beta)
                rows.append({"strategy": strategy, "threshold": float(t), "exclusive": exclusive,
                             "macro_f": macro_f_from_selection(sel, n_true, beta)})
    grid = pd.DataFrame(rows).sort_values("macro_f", ascending=False).reset_index(drop=True)
    best = grid.iloc[0].to_dict()
    return {"strategy": best["strategy"], "threshold": best["threshold"], "exclusive": bool(best["exclusive"]),
            "oof_macro_f": best["macro_f"]}, grid


def to_mapping(selected: pd.DataFrame) -> dict[str, list[str]]:
    """``{s1_id: [cand ids sorted by probability desc]}``."""
    if selected.empty:
        return {}
    df = selected.sort_values(["s1_id", "p"], ascending=[True, False])
    return df.groupby("s1_id", sort=False)["cand_id"].agg(list).to_dict()
