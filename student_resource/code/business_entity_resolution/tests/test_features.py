import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features import _abbr_chunk, cheap_features, full_features  # noqa: E402
from src.featurize import FEAT_COLS  # noqa: E402


def _table(rows):
    df = pd.DataFrame(rows)
    for c in FEAT_COLS:
        num = c.startswith("f_") or c in ("n_parts", "src")
        df[c] = df[c].fillna(0 if num else "") if c in df else (0 if num else "")
    return df[FEAT_COLS]


def test_features_smoke():
    q = _table([{"entity_id": "S1-1", "core": "garza empire", "name_b": "garza empire llc", "legal": "llc",
                 "addr_core": "2739 pebble glen circle millcreek", "house": "2739", "street": "pebble glen circle",
                 "city": "millcreek", "region": "ut", "nums": "2739", "n_parts": 3}])
    r = _table([{"entity_id": "S3-9", "core": "garza empire", "name_b": "garza empire llc", "legal": "llc",
                 "addr_core": "2739 pebble glen cir", "house": "2739", "street": "pebble glen cir", "city": "",
                 "region": "ut", "nums": "2739", "n_parts": 2, "src": 3},
                {"entity_id": "S2-4", "core": "grz empire", "name_b": "grz empire", "legal": "",
                 "addr_core": "", "house": "", "src": 2}])
    r = pd.DataFrame({c: pd.array(r[c].tolist(), dtype="string[pyarrow]") if r[c].dtype == object else r[c]
                      for c in r.columns})
    pairs = pd.DataFrame({"qi": [0, 0], "ri": [0, 1], "in_c1": [1, 1], "in_c2": [1, 0], "in_c5a": [1, 0],
                          "in_c5b": [0, 0], "c1_rank": [1, 2], "crowd_size": [1, 0], "cos_name": [1.0, 0.4],
                          "cos_addr": [0.8, 0.0], "c1_score": [0.9, 0.2], "union_rank": [1, 2], "src": [3, 2]})
    ch = cheap_features(q, r, pairs)
    f = full_features(q, r, pairs, ch)
    assert len(f) == 2 and np.isfinite(f.to_numpy()).all()
    assert f["house_eq"].tolist() == [1, 0] and f["legal_eq"].tolist() == [1, 0]
    assert f["a_abbr_r"].iloc[0] == 1.0          # "cir" is a subsequence-abbreviation of "circle"
    assert f["n_skel_cont_q"].iloc[1] == 1.0     # grz / garza share the skeleton
    assert "country" not in f.columns


def test_abbr_and_acronym():
    fa, fb, acr = _abbr_chunk((np.array(["r pierre dignac", "ibm"]), np.array(["rue pierre dignac", "intl business machines"])))
    assert fa[0] == 1.0 and fb[0] == 1.0 and acr[1] == 1.0
