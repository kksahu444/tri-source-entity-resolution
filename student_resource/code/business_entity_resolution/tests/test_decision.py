import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "decision"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from f05_decision_rule import f05  # noqa: E402
from src.assign import hard_assign, soft_assign  # noqa: E402
from src.decision import two_threshold  # noqa: E402
from src.metrics import entity_f, per_entity  # noqa: E402


def test_entity_f_matches_reference():
    rng = np.random.default_rng(0)
    for _ in range(500):
        truth = set(rng.choice(8, rng.integers(0, 4), replace=False).tolist())
        pred = set(rng.choice(8, rng.integers(0, 4), replace=False).tolist())
        v = entity_f(np.array([len(pred)]), np.array([len(pred & truth)]), np.array([len(truth)]))[0]
        assert abs(v - f05(pred, truth)) < 1e-12


def test_official_example():
    assert abs(entity_f(np.array([3]), np.array([2]), np.array([2]))[0] - 0.7142857142857143) < 1e-12


def test_per_entity_counts_unretrieved_truth():
    # S1 0: 2 true matches, only one retrieved and predicted -> 1.25 / (0.5 + 1)
    s1 = np.array([0, 0, 1])
    sel = np.array([True, False, False])
    y = np.array([1, 0, 0])
    f = per_entity(s1, sel, y, np.array([2, 0]))
    assert abs(f[0] - 1.25 / 1.5) < 1e-12 and f[1] == 1.0


def _brute(s1, p, y, n_true, t1, t2):
    sel = two_threshold.select(s1, p, t1, t2)
    return per_entity(s1, sel, y, n_true).mean()


def test_score_grid_matches_direct_selection():
    rng = np.random.default_rng(1)
    n = 60
    s1 = np.repeat(np.arange(n), rng.integers(1, 5, n))
    p = rng.random(len(s1))
    y = (rng.random(len(s1)) < p).astype(int)
    n_true = np.bincount(s1[y == 1], minlength=n + 5)       # 5 S1s without candidates
    n_true[n:] = rng.integers(0, 2, 5)
    grid = two_threshold.score_grid(s1, p, y, n_true)
    for i, j in [(0, 0), (10, 40), (49, 72), (80, 20), (98, 98)]:
        t1, t2 = two_threshold.GRID[i], two_threshold.GRID[j]
        assert abs(grid[i, j] - _brute(s1, p, y, n_true, t1, t2)) < 1e-12


def test_select_keeps_first_then_second_threshold():
    s1 = np.array([0, 0, 0, 1, 1])
    p = np.array([0.6, 0.5, 0.2, 0.4, 0.9])
    sel = two_threshold.select(s1, p, 0.55, 0.45)
    assert sel.tolist() == [True, True, False, False, True]
    assert two_threshold.select(s1, p, 0.95, 0.1).tolist() == [False] * 5


def test_hard_assign_keeps_best_s1_per_record():
    ri = np.array([7, 7, 3, 7])
    p = np.array([0.2, 0.9, 0.1, 0.5])
    assert hard_assign(ri, p).tolist() == [False, True, True, False]


def test_soft_assign_caps_record_mass():
    ri = np.array([1, 1, 2])
    p = np.array([0.8, 0.6, 0.3])
    out = soft_assign(ri, p)
    assert abs(out[0] + out[1] - 1) < 1e-12 and out[2] == 0.3
