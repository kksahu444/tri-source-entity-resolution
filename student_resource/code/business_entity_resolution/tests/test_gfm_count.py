import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decision import count_model, gfm_rule  # noqa: E402
from src.decision.f05_decision_rule import P_count_model, P_independent, expand_count_classes, gfm  # noqa: E402


def test_gfm_rule_matches_reference_per_entity():
    rng = np.random.default_rng(3)
    s1 = np.repeat(np.arange(40), rng.integers(1, 7, 40))
    p = rng.random(len(s1)) ** 2
    cp = rng.dirichlet(np.ones(4), 40)
    for counts in (None, cp):
        sel, _ = gfm_rule.select(s1, p, counts, skip_below=0.0)
        for e in range(40):
            idx = np.flatnonzero(s1 == e)
            idx = idx[np.argsort(-p[idx], kind="stable")]
            if counts is None:
                P, p0 = P_independent(p[idx])
            else:
                P, p0 = P_count_model(p[idx], expand_count_classes(counts[e], p[idx]))
            chosen, _ = gfm(list(range(len(idx))), P, p0)
            assert set(np.flatnonzero(sel[idx])) == set(chosen)


def test_fast_matches_reference_rule():
    rng = np.random.default_rng(7)
    s1 = np.repeat(np.arange(300), rng.integers(1, 12, 300))     # includes m = 1 and m = 2 lists
    p = rng.random(len(s1)) ** rng.choice([0.5, 1, 3], len(s1))
    for counts in (None, rng.dirichlet(np.ones(4), 300), rng.dirichlet(np.ones(9), 300)):
        a, ea = gfm_rule.select(s1, p, counts)
        b, eb = gfm_rule.select_reference(s1, p, counts)
        assert (a == b).all()
        assert all(abs(ea[k] - eb[k]) < 1e-9 for k in eb)


def test_fast_expand_matches_reference_edge_cases():
    from src.decision.gfm_fast import _expand
    rng = np.random.default_rng(11)
    for K in (4, 6, 9):
        for m in (1, 2, 3, 5, 12):
            pc = rng.dirichlet(np.ones(K))
            pr = rng.random(m)
            assert np.allclose(_expand(pc, np.clip(pr, 1e-6, 1 - 1e-6)), expand_count_classes(pc, pr))
    # the m = 1 fix: P(exactly 2) moves to s = 1 instead of being dropped
    assert np.allclose(_expand(np.array([0.2, 0.5, 0.2, 0.1]), np.array([0.7])), [0.2, 0.8])


def test_likely_singleton_is_empty():
    sel, ef = gfm_rule.select(np.array([0, 0, 0]), np.array([0.05, 0.03, 0.02]), None)
    assert not sel.any() and ef[0] > 0.9


def test_choose_k_and_k_class_fit():
    counts = np.array([0] * 5 + [1] * 5 + [2] * 17 + [3] * 24 + [4] * 22 + [5] * 15 + [6] * 8 + [7] * 3 + [8] * 1)
    k = count_model.choose_k(counts)            # smallest K with share(count >= K-1) <= 5%
    assert k == 8 and np.mean(counts >= k - 1) <= 0.05 and np.mean(counts >= k - 2) > 0.05
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((600, 3)), columns=["a", "b", "c"])
    y = np.minimum((X["a"] * 9).astype(int).to_numpy(), k - 1)
    fold = np.arange(600) % 5
    probs, models, temp = count_model.fit_oof(X, y, fold, 0, k)
    assert probs.shape == (600, k) and np.allclose(probs.sum(1), 1) and temp > 0
    assert count_model.predict(models, temp, X).shape == (600, k)


def test_count_target_and_features():
    s1 = np.array([0, 0, 0, 0, 1, 2, 2])
    y = np.array([1, 1, 1, 1, 0, 1, 0])
    assert count_model.target(s1, y, 4).tolist() == [3, 0, 1, 0]
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.5, 0.2])
    f = count_model.entity_features(s1, p, np.array([0, 1, 0, 1, 0, 0, 1]), np.zeros(7), 4, np.ones(4), np.ones(4))
    assert f.shape[0] == 4 and abs(f.loc[0, "p1"] - 0.9) < 1e-6 and f.loc[3, "n_cand"] == 0
    assert abs(f.loc[2, "gap12"] - 0.3) < 1e-6 and "country" not in f.columns
