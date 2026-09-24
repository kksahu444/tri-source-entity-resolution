"""Decision rules: exclusivity and expected-F selection."""
import pandas as pd

from entity_resolution.decide import select_matches, to_mapping


def scored():
    return pd.DataFrame({
        "s1_id": ["a", "a", "b", "c"],
        "cand_id": ["x", "y", "x", "z"],
        "p": [0.9, 0.2, 0.6, 0.3],
    })


def test_threshold_and_exclusive():
    sel = select_matches(scored(), "threshold", 0.5, exclusive=False)
    assert set(zip(sel.s1_id, sel.cand_id)) == {("a", "x"), ("b", "x")}
    sel = select_matches(scored(), "threshold", 0.5, exclusive=True)
    assert set(zip(sel.s1_id, sel.cand_id)) == {("a", "x")}  # x goes to its best S1 only


def test_expected_f_prefers_empty_for_weak_candidates():
    sel = select_matches(scored(), "expected_f", 0.0, exclusive=True)
    pairs = set(zip(sel.s1_id, sel.cand_id))
    assert ("a", "x") in pairs
    assert ("c", "z") not in pairs  # p=0.3: predicting a singleton has higher expected F0.5


def test_to_mapping_orders_by_probability():
    assert to_mapping(scored()[scored().s1_id == "a"]) == {"a": ["x", "y"]}
