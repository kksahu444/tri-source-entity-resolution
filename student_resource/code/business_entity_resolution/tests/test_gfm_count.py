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


def test_likely_singleton_is_empty():
    sel, ef = gfm_rule.select(np.array([0, 0, 0]), np.array([0.05, 0.03, 0.02]), None)
    assert not sel.any() and ef[0] > 0.9


def test_count_target_and_features():
    s1 = np.array([0, 0, 0, 0, 1, 2, 2])
    y = np.array([1, 1, 1, 1, 0, 1, 0])
    assert count_model.target(s1, y, 4).tolist() == [3, 0, 1, 0]
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.5, 0.2])
    f = count_model.entity_features(s1, p, np.array([0, 1, 0, 1, 0, 0, 1]), np.zeros(7), 4, np.ones(4), np.ones(4))
    assert f.shape[0] == 4 and abs(f.loc[0, "p1"] - 0.9) < 1e-6 and f.loc[3, "n_cand"] == 0
    assert abs(f.loc[2, "gap12"] - 0.3) < 1e-6 and "country" not in f.columns
