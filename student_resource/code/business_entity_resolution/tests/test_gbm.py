import os
import sys

import numpy as np
import pytest

os.environ.setdefault("ER_XGB_DEVICE", "cpu")      # CUDA on Kaggle; CPU for local tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import gbm  # noqa: E402


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 6)).astype(np.float32)
    y = ((X[:, 0] + 0.3 * X[:, 1] + 0.1 * rng.standard_normal(n)) > 0.7).astype(int)
    return X, y


@pytest.mark.parametrize("backend", ["lightgbm", "xgboost_gpu"])
def test_backends_train_predict_importance(backend):
    if backend == "xgboost_gpu":
        pytest.importorskip("xgboost")
    cfg = {"model": {"backend": backend}}
    X, y = _data()
    m = gbm.train_binary(cfg, X[:3000], y[:3000], 300, X[3000:3500], y[3000:3500], 20, seed=1)
    p = m.predict(X[3500:])
    assert p.shape == (500,) and ((p >= 0) & (p <= 1)).all()
    acc = ((p > 0.5) == y[3500:]).mean()
    assert acc > 0.85, acc
    imp = m.importance(6)
    assert imp.shape == (6,) and imp[0] == imp.max()
    full = gbm.train_binary(cfg, X, y, 50, seed=1)
    assert gbm.predict(full, X).shape == (len(X),)


def test_bare_lightgbm_booster_still_predicts():
    import lightgbm as lgb
    X, y = _data()
    b = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(X, y), 20)
    assert np.allclose(gbm.predict(b, X), b.predict(X))
