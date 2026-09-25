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
from .blocking import build_text_index, generate_candidates
from .config import Config
from .decide import select_matches, to_mapping, tune_decision
from .features import build_features, feature_columns
from .io import CANDIDATE_COLUMN, MATCH_COLUMN, SplitData, load_split, write_id_lists
from .metrics import blocking_report, f_beta_from_counts, macro_f_from_selection
from .model import add_stage2_features, cross_validate, feature_importance, predict_average
from .normalize import normalize_country, normalize_records
from .validate import validate_outputs

log = logging.getLogger(__name__)
MODEL_FILE = "model.joblib"


@dataclass
class Prepared:
    """Everything derived from one split before modelling."""

    data: SplitData
    recs: pd.DataFrame  # entity_id, source, country (normalised) of every record
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


def _sample_s1(data: SplitData, share: float, seed: int) -> SplitData:
    """Keep a random ``share`` of the Source-1 records (and their labels); every S2/S3 record stays,
    so the candidate pool per S1 record is as large as in the full run."""
    s1 = np.asarray(data.s1_ids)
    keep = s1[np.random.default_rng(seed).random(len(s1)) < share]
    kept = set(keep)
    recs = data.records
    recs = recs[(recs["source"] != 1) | recs["entity_id"].isin(kept)].reset_index(drop=True)
    gt = {k: v for k, v in data.ground_truth.items() if k in kept} if data.ground_truth is not None else None
    log.info("sampled %s of %s S1 records", f"{len(keep):,}", f"{len(s1):,}")
    return SplitData(records=recs, s1_ids=keep.tolist(), ground_truth=gt)


def prepare(cfg: Config, split: str, with_features: bool = True) -> Prepared:
    """Load one split, then normalise, block and (optionally) featurise it country by country.

    There are no cross-country matches, so each country is an independent problem with its own
    TF-IDF vocabulary; only one country's matrices are in memory at a time. Countries are taken
    from the data (open set), and a country with no Source-1 record is skipped.
    """
    with _Timer(f"[{split}] load"):
        data = load_split(cfg.data_dir, split)
    if split == "train" and cfg.blocking.train_s1_sample < 1.0:
        data = _sample_s1(data, cfg.blocking.train_s1_sample, cfg.model.seed)
    raw = data.records["country"]
    country = raw.map({c: normalize_country(c) for c in raw.unique()})
    cand_parts, feat_parts = [], []
    for c in sorted(country[data.records["source"] == 1].unique()):
        part = data.records[(country == c).to_numpy()]
        with _Timer(f"[{split}/{c or '-'}] {len(part):,} records"):
            recs = normalize_records(part)
            index = build_text_index(recs, cfg.blocking, for_features=with_features)
            cands = generate_candidates(recs, index, cfg.blocking)
            if not cands.empty:
                cand_parts.append(cands)
                if with_features:
                    feat_parts.append(build_features(recs, cands, index))
        del recs, index
    cands = pd.concat(cand_parts, ignore_index=True) if cand_parts else pd.DataFrame(columns=["s1_id", "cand_id"])
    feats = pd.concat(feat_parts, ignore_index=True) if feat_parts else pd.DataFrame()
    recs = data.records[["entity_id", "source"]].assign(country=country.to_numpy())
    return Prepared(data, recs, cands, feats)


def _n_true(data: SplitData) -> pd.Series:
    gt = data.ground_truth or {}
    return pd.Series({s1: len(set(gt.get(s1, []))) for s1 in data.s1_ids}, dtype=float)


def _label(feats: pd.DataFrame, gt: dict[str, list[str]]) -> np.ndarray:
    pos = {(s1, c) for s1, ids in gt.items() for c in ids}
    return np.fromiter(((a, b) in pos for a, b in zip(feats["s1_id"], feats["cand_id"])), dtype=np.int8,
                       count=len(feats))


def block_report(cfg: Config, split: str = "train") -> dict:
    """Blocking-only run: recall ceiling, recall at tighter cuts and per channel, overall and per
    country, on a labelled split. On an unlabelled split only the candidate-set size is reported."""
    prep = prepare(cfg, split, with_features=False)
    recs, cands = prep.recs, prep.cands
    per_s1 = cands.groupby("s1_id").size().reindex(prep.data.s1_ids, fill_value=0)
    if prep.data.ground_truth is None:
        report = {"n_candidate_pairs": int(len(cands)), "mean_candidates_per_s1": float(per_s1.mean()),
                  "max_candidates_per_s1": int(per_s1.max()), "s1_without_candidates": int((per_s1 == 0).sum())}
        log.info("blocking report: %s", json.dumps(report, indent=2))
        return report
    gt = prep.data.ground_truth
    src = recs["source"]
    report = blocking_report(cands, gt, int((src == 1).sum()), int((src != 1).sum()))
    s1_country = recs.loc[src == 1].set_index("entity_id")["country"]
    report["by_country"] = {}
    for country, ids in s1_country.groupby(s1_country).groups.items():
        ids = set(ids)
        n_tgt = int(((recs["country"] == country) & (src != 1)).sum())
        sub = blocking_report(cands[cands["s1_id"].isin(ids)], {k: v for k, v in gt.items() if k in ids},
                              len(ids), n_tgt)
        report["by_country"][str(country)] = {k: v for k, v in sub.items() if k != "channels"}
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
