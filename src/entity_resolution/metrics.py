"""Official metric (macro F0.5 over Source-1 entities) and blocking diagnostics."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

BETA = 0.5


def f_beta(pred: Iterable[str], true: Iterable[str], beta: float = BETA) -> float:
    """Per-entity F-beta exactly as the challenge defines it.

    * no true matches and empty prediction -> 1.0 (correct singleton)
    * no true matches but something predicted -> 0.0
    * true matches but empty prediction or zero overlap -> 0.0
    """
    pred, true = set(pred), set(true)
    if not true and not pred:
        return 1.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision, recall = tp / len(pred), tp / len(true)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def macro_f_beta(
    predictions: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str] | None = None,
    beta: float = BETA,
) -> float:
    """Average per-entity F-beta over ``s1_ids`` (default: all ids in ``truth``)."""
    ids = list(s1_ids) if s1_ids is not None else list(truth)
    if not ids:
        return float("nan")
    return float(np.mean([f_beta(predictions.get(i, ()), truth.get(i, ()), beta) for i in ids]))


def f_beta_from_counts(tp: np.ndarray, n_pred: np.ndarray, n_true: np.ndarray, beta: float = BETA) -> np.ndarray:
    """Vectorised per-entity F-beta from counts (same edge cases as :func:`f_beta`)."""
    tp, n_pred, n_true = (np.asarray(a, dtype=float) for a in (tp, n_pred, n_true))
    out = np.zeros_like(tp)
    out[(n_true == 0) & (n_pred == 0)] = 1.0
    m = tp > 0
    precision, recall = tp[m] / n_pred[m], tp[m] / n_true[m]
    b2 = beta * beta
    out[m] = (1 + b2) * precision * recall / (b2 * precision + recall)
    return out


def macro_f_from_selection(selected: pd.DataFrame, n_true: pd.Series, beta: float = BETA) -> float:
    """Macro F-beta for selected pairs.

    ``selected`` has columns ``s1_id`` and ``label`` (1 = true match); ``n_true`` is indexed by
    every Source-1 id being scored and holds its number of true matches (including matches the
    blocker missed, so recall is not flattered).
    """
    agg = selected.groupby("s1_id")["label"].agg(["size", "sum"])
    n_pred = agg["size"].reindex(n_true.index, fill_value=0).to_numpy()
    tp = agg["sum"].reindex(n_true.index, fill_value=0).to_numpy()
    return float(f_beta_from_counts(tp, n_pred, n_true.to_numpy(), beta).mean())


def truth_pairs(truth: Mapping[str, Iterable[str]]) -> pd.DataFrame:
    rows = [(s1, c) for s1, ids in truth.items() for c in ids]
    return pd.DataFrame(rows, columns=["s1_id", "cand_id"])


def blocking_report(
    cands: pd.DataFrame, truth: Mapping[str, Iterable[str]], n_s1: int, n_targets: int, beta: float = BETA,
    cuts: Iterable[int] = (1, 2, 3, 5, 10, 15, 20, 30),
) -> dict:
    """Recall ceiling and reduction ratio of a candidate set.

    ``upper_bound_macro_f`` is the score a perfect matcher would reach using only these candidates.
    With ``blk_rank`` present, ``at_cut`` gives the same numbers for a tighter per-source cut; with
    ``blk_mask`` present, ``channels`` gives each blocker's recall and the recall only it provides.
    """
    from .blocking import BLOCKER_BITS  # noqa: PLC0415 (metrics stays importable without blocking deps)

    gt = truth_pairs(truth)
    n_true = pd.Series({k: len(set(v)) for k, v in truth.items()}, dtype=float)
    cols = ["s1_id", "cand_id"] + [c for c in ("blk_rank", "blk_mask") if c in cands.columns]
    hit = gt.merge(cands[cols].drop_duplicates(["s1_id", "cand_id"]), on=["s1_id", "cand_id"], how="inner")

    def summary(pairs: pd.DataFrame, found: pd.DataFrame) -> dict:
        per_s1 = pairs.groupby("s1_id").size().reindex(n_true.index, fill_value=0)
        tp = found.groupby("s1_id").size().reindex(n_true.index, fill_value=0).to_numpy()
        return {
            "n_candidate_pairs": int(len(pairs)),
            "pair_recall": float(len(found) / max(len(gt), 1)),
            "mean_candidates_per_s1": float(per_s1.mean()) if len(per_s1) else 0.0,
            "upper_bound_macro_f": float(f_beta_from_counts(tp, tp, n_true.to_numpy(), beta).mean()),
        }

    report = summary(cands, hit)
    report["reduction_ratio"] = float(1 - len(cands) / max(n_s1 * n_targets, 1))
    report["max_candidates_per_s1"] = int(cands.groupby("s1_id").size().max()) if len(cands) else 0
    for src in ("S2-", "S3-"):
        gt_src = gt[gt["cand_id"].str.startswith(src)]
        hit_src = hit[hit["cand_id"].str.startswith(src)]
        report[f"pair_recall_{src[:2]}"] = float(len(hit_src) / max(len(gt_src), 1))
    if "blk_rank" in cands.columns:
        report["at_cut"] = {int(n): summary(cands[cands["blk_rank"] < n], hit[hit["blk_rank"] < n]) for n in cuts}
    if "blk_mask" in cands.columns:
        report["channels"] = {}
        for name, bit in BLOCKER_BITS.items():
            mask = hit["blk_mask"].to_numpy()
            if not (cands["blk_mask"].to_numpy() & bit).any():
                continue
            report["channels"][name] = {
                "pair_recall": float(((mask & bit) > 0).sum() / max(len(gt), 1)),
                "only_this_recall": float((mask == bit).sum() / max(len(gt), 1)),
                "share_of_pairs": float(((cands["blk_mask"].to_numpy() & bit) > 0).mean()),
            }
    return report
