"""Pairwise match classifier: gradient-boosted trees with grouped K-fold cross-validation.

Folds are grouped by Source-1 entity so all candidates of one entity are scored by a model that
never saw that entity - the out-of-fold (OOF) probabilities are then an honest basis for tuning
the decision rule. Test predictions average the fold models.

LightGBM (MIT licence) is used when installed; scikit-learn's HistGradientBoosting is the fallback.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import ModelConfig
from .features import add_group_context

log = logging.getLogger(__name__)

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - exercised only without lightgbm
    lgb = None


def _group_folds(groups: np.ndarray, n_folds: int, seed: int):
    try:
        splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    except TypeError:  # scikit-learn < 1.6 has no shuffle option
        splitter = GroupKFold(n_splits=n_folds)
    return splitter.split(np.zeros(len(groups)), groups=groups)


def _fit_one(X_tr, y_tr, X_va, y_va, cfg: ModelConfig):
    """Train one model with early stopping on the validation fold."""
    if lgb is not None:
        params = {**cfg.lgb_params, "seed": cfg.seed}
        train_set = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        valid_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
        booster = lgb.train(
            params, train_set, num_boost_round=cfg.num_boost_round, valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        )
        return booster
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, early_stopping=True, random_state=cfg.seed)
    return clf.fit(X_tr, y_tr)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Match probability for each row, for either backend."""
    if lgb is not None and isinstance(model, lgb.Booster):
        return model.predict(X, num_iteration=model.best_iteration or None)
    return model.predict_proba(X)[:, 1]


def cross_validate(feats: pd.DataFrame, columns: list[str], cfg: ModelConfig) -> tuple[np.ndarray, list]:
    """Grouped K-fold training. Returns OOF probabilities and the fold models."""
    X = feats[columns]
    y = feats["label"].to_numpy()
    groups = feats["s1_id"].to_numpy()
    oof = np.zeros(len(feats), dtype=np.float64)
    models = []
    for fold, (tr, va) in enumerate(_group_folds(groups, cfg.n_folds, cfg.seed)):
        model = _fit_one(X.iloc[tr], y[tr], X.iloc[va], y[va], cfg)
        oof[va] = predict_proba(model, X.iloc[va])
        models.append(model)
        n_iter = getattr(model, "best_iteration", None)
        log.info("fold %d: %s pairs, best_iteration=%s", fold, f"{len(va):,}", n_iter)
    return oof, models


def predict_average(models: list, feats: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Mean probability of the fold models."""
    X = feats[columns]
    return np.mean([predict_proba(m, X) for m in models], axis=0)


def add_stage2_features(feats: pd.DataFrame, p1: np.ndarray) -> pd.DataFrame:
    """Stage-2 inputs: stage-1 probability and its rank / gap inside each group, plus group totals."""
    out = feats.copy()
    out["p1"] = p1.astype(np.float32)
    out = add_group_context(out, ["p1"], counts=False)
    out["p1_sum_s1"] = out.groupby("s1_id")["p1"].transform("sum").astype(np.float32)
    out["p1_sum_cand"] = out.groupby("cand_id")["p1"].transform("sum").astype(np.float32)
    return out


def feature_importance(models: list, columns: list[str]) -> pd.DataFrame:
    """Mean gain importance across fold models (LightGBM only)."""
    if lgb is None or not models or not isinstance(models[0], lgb.Booster):
        return pd.DataFrame(columns=["feature", "gain"])
    gain = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
    return pd.DataFrame({"feature": columns, "gain": gain}).sort_values("gain", ascending=False).reset_index(drop=True)
