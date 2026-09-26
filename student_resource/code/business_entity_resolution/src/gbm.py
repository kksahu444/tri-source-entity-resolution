"""Gradient-boosting backend: LightGBM (CPU, default) or XGBoost on CUDA (Kaggle T4s).

config `model.backend`: "lightgbm" | "xgboost_gpu". Both grow leaf-wise trees with the same caps
(127 leaves, 63 bins, 0.7 row/feature sampling, eta 0.1), so they are interchangeable GBDTs; the backend
used for a run is recorded in its config hash.
"""
from __future__ import annotations

import os

import numpy as np

LGB_BINARY = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
              "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 1.0,
              "max_bin": 63, "verbose": -1, "num_threads": 0}
XGB_BINARY = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "device": os.environ.get("ER_XGB_DEVICE", "cuda"),
              "grow_policy": "lossguide", "max_leaves": 127, "max_depth": 0, "eta": 0.1, "subsample": 0.7,
              "colsample_bytree": 0.7, "min_child_weight": 10, "lambda": 1.0, "max_bin": 63}


class Model:
    """Uniform predict / importance over either backend (and over bare LightGBM boosters in old pickles)."""

    def __init__(self, booster, kind: str, best_iteration: int):
        self.booster, self.kind, self.best_iteration = booster, kind, best_iteration

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "xgb":
            import xgboost as xgb
            it = (0, self.best_iteration + 1) if self.best_iteration else (0, 0)
            return self.booster.inplace_predict(X, iteration_range=it)
        return self.booster.predict(X, num_iteration=self.best_iteration or None)

    def importance(self, n_features: int) -> np.ndarray:
        if self.kind == "xgb":
            score = self.booster.get_score(importance_type="total_gain")
            out = np.zeros(n_features)
            for k, v in score.items():
                out[int(k[1:])] = v
            return out
        return self.booster.feature_importance("gain")


def backend(cfg: dict) -> str:
    return (cfg.get("model") or {}).get("backend", "lightgbm")


def train_binary(cfg: dict, X, y, rounds: int, X_val=None, y_val=None, early: int | None = None,
                 seed: int = 0, lgb_params: dict | None = None) -> Model:
    if backend(cfg) == "xgboost_gpu":
        import xgboost as xgb
        dtr = xgb.QuantileDMatrix(X, y, max_bin=XGB_BINARY["max_bin"])
        evals, kw = [], {}
        if X_val is not None:
            dva = xgb.QuantileDMatrix(X_val, y_val, ref=dtr, max_bin=XGB_BINARY["max_bin"])
            evals = [(dva, "val")]
            kw["early_stopping_rounds"] = early
        b = xgb.train({**XGB_BINARY, "seed": seed}, dtr, rounds, evals=evals, verbose_eval=False, **kw)
        best = int(getattr(b, "best_iteration", rounds - 1)) if X_val is not None else rounds - 1
        return Model(b, "xgb", best)
    import lightgbm as lgb
    params = {**(lgb_params or LGB_BINARY), "seed": seed}
    if X_val is not None:
        b = lgb.train(params, lgb.Dataset(X, y), rounds, valid_sets=[lgb.Dataset(X_val, y_val)],
                      callbacks=[lgb.early_stopping(early, verbose=False)])
        return Model(b, "lgb", b.best_iteration)
    b = lgb.train(params, lgb.Dataset(X, y), rounds)
    return Model(b, "lgb", 0)


def predict(m, X: np.ndarray) -> np.ndarray:
    if isinstance(m, Model):
        return m.predict(X)
    return m.predict(X, num_iteration=getattr(m, "best_iteration", None) or None)   # bare LightGBM booster
