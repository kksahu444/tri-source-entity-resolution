import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.error_analysis import _classify_misses, write_budget  # noqa: E402
from src.metrics import entity_f  # noqa: E402


def test_budget_sums_to_loss_and_classifies_misses():
    # entities: singleton wrongly linked; abstained with truth retrieved; all truth blocked; partial with an extra link
    n_true = np.array([0, 2, 1, 3])
    n_pred = np.array([1, 0, 0, 3])
    tp = np.array([0, 0, 0, 2])
    n_ret = np.array([0, 2, 0, 2])
    f = entity_f(n_pred, tp, n_true)
    df = pd.DataFrame({"country": ["US", "US", "India", "India"], "n_true": n_true, "n_pred": n_pred, "n_ret": n_ret,
                       "f": f, "f_nofp": entity_f(tp, tp, n_true), "f_ceil": entity_f(n_ret, n_ret, n_true)})
    qt = pa.table({"core": ["acme co", "beta"], "house": ["12", "5"], "city": ["pune", "tyler"]})
    rt = pa.table({"core": ["acme", "zeta", "beta"], "house": ["12", "5", "9"], "city": ["pune", "tyler", "tyler"],
                   "addr_core": ["", "5 main", "9 elm"], "f_nonlatin": [1, 0, 0]})
    misses = _classify_misses("US", qt, rt, np.array([0, 1, 1]), np.array([0, 1, 2]), np.array([0, 5]))
    assert misses["noise"].tolist() == ["empty record address", "name replaced (no shared token)", "house number conflict"]
    assert misses["script"].tolist() == ["native", "Latin", "Latin"]
    cfg = {"paths": {"runs": tempfile.mkdtemp()}}
    write_budget(cfg, df, misses, "t", "gfm_indep")       # asserts the five causes sum to 1 - F per entity
    text = open(os.path.join(cfg["paths"]["runs"], "error_budget_t.md"), encoding="utf-8").read()
    assert "false merge on a singleton" in text and "| US | native | empty record address | 1 |" in text
