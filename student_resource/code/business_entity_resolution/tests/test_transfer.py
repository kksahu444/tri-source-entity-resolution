import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import transfer  # noqa: E402


def test_transfer_checks_run_and_flag():
    tmp = tempfile.mkdtemp()
    for d in ("block", "feat", "pred"):
        os.makedirs(f"{tmp}/{d}")
    rng = np.random.default_rng(0)
    for split, shift in (("dev", 0.0), ("test", 0.3)):
        n = 600
        pd.DataFrame({"x": np.arange(100)}).to_parquet(f"{tmp}/block/{split}_US_q.parquet")
        pd.DataFrame({"x": np.arange(800)}).to_parquet(f"{tmp}/block/{split}_US_r.parquet")
        pd.DataFrame({"x": [0]}).to_parquet(f"{tmp}/block/{split}_US_pairs.parquet")
        p = np.clip(rng.random(n) + shift, 0, 1)
        d = pd.DataFrame({"qi": np.arange(n) // 6, "ri": rng.integers(0, 800, n), "p": p})
        if split == "dev":
            d["y"] = (rng.random(n) < p).astype(int)
        d.to_parquet(f"{tmp}/pred/{split}_US.parquet")
        d[["qi", "ri", "p"]].to_parquet(f"{tmp}/pred/{split}_US_stageA.parquet")
        pd.DataFrame({"src3": rng.integers(0, 2, n), "addr_r_empty": 0.0, "n_parts_q": 3.0,
                      "q_name_city_crowd": 1.0}).to_parquet(f"{tmp}/feat/{split}_US.parquet")
    iso = IsotonicRegression(out_of_bounds="clip").fit([0, 1], [0, 1])
    pickle.dump({"iso": iso, "rule": "gfm_indep", "thresholds": {"pooled": {"tau1": 0.5, "tau2": 0.7}}},
                open(f"{tmp}/decision.pkl", "wb"))
    cfg_path = os.path.join(tmp, "cfg.yaml")
    import yaml
    yaml.safe_dump({"seed": 0, "paths": {"artifacts": tmp, "runs": tmp, "dataset": tmp, "output": tmp,
                                         "submissions": tmp}}, open(cfg_path, "w"))
    transfer.main(["--config", cfg_path, "--tag", "t"])
    text = open(f"{tmp}/transfer_checks_t.md", encoding="utf-8").read()
    assert "FLAG" in text and "d_stageA_p_mean" in text     # the test split was shifted upwards
