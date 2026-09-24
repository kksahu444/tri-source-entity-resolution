"""The metric must match the challenge definition exactly."""
import pandas as pd
import pytest

from entity_resolution.metrics import f_beta, macro_f_beta, macro_f_from_selection


def test_problem_statement_example():
    pred = ["S2-00047", "S2-00193", "S3-00812"]
    true = ["S2-00047", "S3-00812"]
    assert f_beta(pred, true) == pytest.approx(0.714, abs=1e-3)


def test_singleton_rules():
    assert f_beta([], []) == 1.0  # correct singleton
    assert f_beta(["S2-1"], []) == 0.0  # false merge on a singleton
    assert f_beta([], ["S2-1"]) == 0.0  # missed everything
    assert f_beta(["S2-9"], ["S2-1"]) == 0.0  # wrong match


def test_macro_average_includes_singletons():
    truth = {"a": ["S2-1"], "b": []}
    assert macro_f_beta({"a": ["S2-1"]}, truth) == 1.0
    assert macro_f_beta({"a": ["S2-1"], "b": ["S3-4"]}, truth) == 0.5


def test_vectorised_matches_reference():
    truth = {"a": ["x", "y"], "b": [], "c": ["z"], "d": []}
    pred = {"a": ["x", "q", "y"], "b": [], "c": [], "d": ["w"]}
    sel = pd.DataFrame([(s, c, int(c in truth[s])) for s, ids in pred.items() for c in ids],
                       columns=["s1_id", "cand_id", "label"])
    n_true = pd.Series({k: len(v) for k, v in truth.items()}, dtype=float)
    assert macro_f_from_selection(sel, n_true) == pytest.approx(macro_f_beta(pred, truth))
