import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.stack_b import context_features, record_keys  # noqa: E402


def test_context_features_siblings_and_competition():
    r = pd.DataFrame({"core": ["acme", "acme", "beta", ""], "addr_core": ["1 main", "1 main", "2 elm", ""],
                      "house": ["1", "1", "2", ""]})
    keys = record_keys(r)
    qi = np.array([0, 0, 0, 0, 1])
    ri = np.array([0, 1, 2, 3, 2])
    pa = np.array([0.9, 0.4, 0.3, 0.2, 0.8])
    f = context_features(qi, ri, pa, keys)
    assert f["pa_rank"].tolist() == [1, 2, 3, 4, 1]
    assert np.allclose(f["pa_sib_name"], [0.4, 0.9, 0.0, 0.0, 0.0])      # other "acme" candidate of the same S1
    assert np.allclose(f["pa_rev_gap"], [0, 0, 0.5, 0, 0])               # record 2 is contested by S1 1 (0.8)
    assert np.allclose(f["pa_others_sum"][:4], [0.9, 1.4, 1.5, 1.6])
    assert np.isfinite(f.to_numpy()).all()
