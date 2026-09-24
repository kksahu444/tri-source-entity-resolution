"""End-to-end orchestration: data -> normalise -> block -> features -> model -> decision -> files."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .blocking import TextIndex, build_text_index, generate_candidates
from .config import Config
from .decide import select_matches, to_mapping, tune_decision
from .features import build_features, feature_columns
from .io import CANDIDATE_COLUMN, MATCH_COLUMN, SplitData, load_split, write_id_lists
from .metrics import blocking_report, f_beta_from_counts, macro_f_from_selection
from .model import add_stage2_features, cross_validate, feature_importance, predict_average
from .normalize import normalize_records
from .validate import validate_outputs

log = logging.getLogger(__name__)
MODEL_FILE = "model.joblib"


@dataclass
class Prepared:
    """Everything derived from one split before modelling."""

    data: SplitData
    recs: pd.DataFrame
    index: TextIndex
    cands: pd.DataFrame
    feats: pd.DataFrame


class _Timer:
    def __init__(self, what: str):
        self.what = what

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        log.info("%s done in %.1fs", self.what, time.perf_counter() - self.t0)


def prepare(cfg: Config, split: str, with_features: bool = True) -> Prepared:
    """Load, normalise, block and (optionally) featurise one split."""
    with _Timer(f"[{split}] load + normalise"):
        data = load_split(cfg.data_dir, split)
        recs = normalize_records(data.records)
    with _Timer(f"[{split}] text index"):
        index = build_text_index(recs, cfg.blocking)
    with _Timer(f"[{split}] blocking"):
        cands = generate_candidates(recs, index, cfg.blocking)
    feats = pd.DataFrame()
    if with_features:
        with _Timer(f"[{split}] features"):
            feats = build_features(recs, cands, index)
    return Prepared(data, recs, index, cands, feats)


def _n_true(data: SplitData) -> pd.Series:
    gt = data.ground_truth or {}
    return pd.Series({s1: len(set(gt.get(s1, []))) for s1 in data.s1_ids}, dtype=float)


def _label(feats: pd.DataFrame, gt: dict[str, list[str]]) -> np.ndarray:
    pos = {(s1, c) for s1, ids in gt.items() for c in ids}
    return np.fromiter(((a, b) in pos for a, b in zip(feats["s1_id"], feats["cand_id"])), dtype=np.int8,
                       count=len(feats))


def block_report(cfg: Config, split: str = "train") -> dict:
    """Blocking-only run: recall ceiling / reduction ratio on a labelled split."""
    prep = prepare(cfg, split, with_features=False)
    if prep.data.ground_truth is None:
        raise ValueError(f"{split} has no ground truth; blocking recall needs labels")
    src = prep.recs["source"]
    report = blocking_report(prep.cands, prep.data.ground_truth, int((src == 1).sum()), int((src != 1).sum()))
    log.info("blocking report: %s", json.dumps(report, indent=2))
    return report


def _per_group_f(selected: pd.DataFrame, n_true: pd.Series, groups: pd.Series) -> dict:
    """Macro F0.5 per group label (e.g. country) for the chosen selection."""
    agg = selected.groupby("s1_id")["label"].agg(["size", "sum"])
    n_pred = agg["size"].reindex(n_true.index, fill_value=0).to_numpy()
    tp = agg["sum"].reindex(n_true.index, fill_value=0).to_numpy()
    f = pd.Series(f_beta_from_counts(tp, n_pred, n_true.to_numpy()), index=n_true.index)
    out = {str(k): float(v) for k, v in f.groupby(groups.reindex(n_true.index).to_numpy()).mean().items()}
    singles = n_true == 0
    out["_singletons"] = float(f[singles].mean()) if singles.any() else float("nan")
    out["_non_singletons"] = float(f[~singles].mean()) if (~singles).any() else float("nan")
    return out


def train(cfg: Config) -> dict:
    """Fit the matcher on the train split, tune the decision rule on OOF scores, save artifacts."""
    art = Path(cfg.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    prep = prepare(cfg, "train")
    gt = prep.data.ground_truth
    if gt is None:
        raise ValueError("train split has no ground truth file")
    src = prep.recs["source"]
    report: dict = {"version": __version__,
                    "blocking": blocking_report(prep.cands, gt, int((src == 1).sum()), int((src != 1).sum()))}
    log.info("blocking: %s", report["blocking"])

    feats = prep.feats
    feats["label"] = _label(feats, gt)
    n_true = _n_true(prep.data)
    cols1 = feature_columns(feats)
    with _Timer("stage-1 CV"):
        oof1, models1 = cross_validate(feats, cols1, cfg.model)

    def tuned(p: np.ndarray) -> tuple[dict, pd.DataFrame]:
        scored = feats[["s1_id", "cand_id", "label"]].assign(p=p)
        if cfg.decision.tune:
            return tune_decision(scored, n_true)
        d = cfg.decision
        sel = select_matches(scored, d.strategy, d.threshold, d.exclusive)
        return {"strategy": d.strategy, "threshold": d.threshold, "exclusive": d.exclusive,
                "oof_macro_f": macro_f_from_selection(sel, n_true)}, pd.DataFrame()

    decision1, grid1 = tuned(oof1)
    report["stage1"] = decision1
    log.info("stage-1 OOF: %s", decision1)

    bundle = {"version": __version__, "config": cfg.to_dict(), "stage1_models": models1, "stage1_columns": cols1,
              "stage2_models": None, "stage2_columns": None, "decision": decision1}
    final_p, final_grid = oof1, grid1
    if cfg.model.stack:
        feats2 = add_stage2_features(feats, oof1)
        cols2 = feature_columns(feats2)
        with _Timer("stage-2 CV"):
            oof2, models2 = cross_validate(feats2, cols2, cfg.model)
        decision2, grid2 = tuned(oof2)
        report["stage2"] = decision2
        log.info("stage-2 OOF: %s", decision2)
        if decision2["oof_macro_f"] > decision1["oof_macro_f"]:
            bundle.update(stage2_models=models2, stage2_columns=cols2, decision=decision2)
            final_p, final_grid = oof2, grid2
            pd.DataFrame(feature_importance(models2, cols2)).to_csv(art / "feature_importance_stage2.tsv",
                                                                     sep="\t", index=False)
    report["decision"] = bundle["decision"]

    d = bundle["decision"]
    scored = feats[["s1_id", "cand_id", "label"]].assign(p=final_p)
    selected = select_matches(scored, d["strategy"], d["threshold"], d["exclusive"])
    s1_country = prep.recs.loc[src == 1].set_index("entity_id")["country"]
    report["oof_by_group"] = _per_group_f(selected, n_true, s1_country)
    report["n_train_pairs"] = int(len(feats))
    report["positive_rate"] = float(feats["label"].mean())

    joblib.dump(bundle, art / MODEL_FILE)
    feature_importance(models1, cols1).to_csv(art / "feature_importance_stage1.tsv", sep="\t", index=False)
    if not final_grid.empty:
        final_grid.to_csv(art / "decision_grid.tsv", sep="\t", index=False)
    scored.to_csv(art / "oof_scores.tsv", sep="\t", index=False)
    (art / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("final OOF macro F0.5 = %.4f  (%s)", d["oof_macro_f"], json.dumps(report["oof_by_group"]))
    return report


def predict(cfg: Config, split: str = "test") -> dict:
    """Score a split with the saved model and write both submission files."""
    bundle = joblib.load(Path(cfg.artifacts_dir) / MODEL_FILE)
    trained_cfg = Config.from_dict(bundle["config"])
    trained_cfg.data_dir, trained_cfg.output_dir = cfg.data_dir, cfg.output_dir  # paths may differ
    prep = prepare(trained_cfg, split)
    feats = prep.feats
    p = predict_average(bundle["stage1_models"], feats, bundle["stage1_columns"])
    if bundle["stage2_models"]:
        feats2 = add_stage2_features(feats, p)
        p = predict_average(bundle["stage2_models"], feats2, bundle["stage2_columns"])
    d = bundle["decision"]
    scored = feats[["s1_id", "cand_id"]].assign(p=p)
    selected = select_matches(scored, d["strategy"], d["threshold"], d["exclusive"])

    out = Path(cfg.output_dir)
    matches = to_mapping(selected)
    candidates = to_mapping(scored)
    s1_ids = prep.data.s1_ids
    write_id_lists(out / "matching_results.tsv", s1_ids, matches, MATCH_COLUMN)
    write_id_lists(out / "candidate_pairs.tsv", s1_ids, candidates, CANDIDATE_COLUMN)
    scored.to_csv(Path(cfg.artifacts_dir) / f"{split}_scores.tsv", sep="\t", index=False)

    n_matched = sum(1 for s in s1_ids if matches.get(s))
    summary = {"split": split, "n_s1": len(s1_ids), "n_candidate_pairs": int(len(scored)),
               "n_predicted_pairs": int(len(selected)), "s1_with_matches": n_matched,
               "predicted_singleton_rate": 1 - n_matched / max(len(s1_ids), 1), "decision": d}
    if split == "test":
        errors, warnings = validate_outputs(out / "matching_results.tsv", out / "candidate_pairs.tsv",
                                            Path(cfg.data_dir) / split)
        summary.update(validation_errors=errors, validation_warnings=warnings)
        for msg in errors:
            log.error("validation: %s", msg)
        for msg in warnings:
            log.warning("validation: %s", msg)
    log.info("prediction summary: %s", json.dumps(summary, indent=2, default=str))
    return summary


def leave_one_country_out(cfg: Config) -> dict:
    """Generalisation check for unseen countries (France is test-only).

    For each train country C: fit on the other countries (grouped CV for OOF tuning), then score C
    with the averaged fold models and the decision tuned without C.
    """
    prep = prepare(cfg, "train")
    gt = prep.data.ground_truth or {}
    feats = prep.feats
    feats["label"] = _label(feats, gt)
    cols = feature_columns(feats)
    s1_country = prep.recs.loc[prep.recs["source"] == 1].set_index("entity_id")["country"]
    n_true = _n_true(prep.data)
    results = {}
    for country in sorted(s1_country.unique()):
        held = s1_country.index[s1_country == country]
        mask = feats["s1_id"].isin(held).to_numpy()
        if mask.all() or not mask.any():
            continue
        tr = feats[~mask].reset_index(drop=True)
        oof, models = cross_validate(tr, cols, cfg.model)
        n_true_tr = n_true.drop(index=held, errors="ignore")
        decision, _ = tune_decision(tr[["s1_id", "cand_id", "label"]].assign(p=oof), n_true_tr)
        te = feats[mask].reset_index(drop=True)
        scored = te[["s1_id", "cand_id", "label"]].assign(p=predict_average(models, te, cols))
        sel = select_matches(scored, decision["strategy"], decision["threshold"], decision["exclusive"])
        results[str(country)] = {"held_out_macro_f": macro_f_from_selection(sel, n_true.reindex(held)),
                                 "in_domain_oof_macro_f": decision["oof_macro_f"], "decision": decision}
        log.info("LOCO %s: %s", country, results[str(country)])
    Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.artifacts_dir) / "loco_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
