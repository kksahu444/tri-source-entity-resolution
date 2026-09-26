import os

import numpy as np
import pandas as pd
import pytest

from src import ce_lane_export as ce_export


def _run(tmp_path):
    art = tmp_path / "art"
    for d in ("block", "feat", "norm_tl"):
        (art / d).mkdir(parents=True)
    pd.DataFrame({"entity_id": ["S1-a", "S1-b"]}).to_parquet(art / "block" / "dev_US_q.parquet")
    pd.DataFrame({"entity_id": ["S2-x", "S3-y", "S2-z"]}).to_parquet(art / "block" / "dev_US_r.parquet")
    pd.DataFrame({"qi": [0]}).to_parquet(art / "block" / "dev_US_pairs.parquet")
    pd.DataFrame({"qi": [1, 0, 0], "ri": [2, 0, 1], "prune_score": [0.2, 0.9, 0.5]}).to_parquet(art / "feat" / "dev_US.parquet")
    rows = {1: [("S1-a", "Garza LLC", "1 Main St"), ("S1-b", "Tata Pvt Ltd", "")],
            2: [("S2-x", "Garza", "1 Main Street"), ("S2-z", "टाटा", "Pune")],
            3: [("S3-y", "Garza L.L.C.", "1 Main St")]}
    for k, rs in rows.items():
        pd.DataFrame(rs, columns=["entity_id", "name_a", "addr_a"]).to_parquet(art / "norm_tl" / f"train_s{k}.parquet")
    return {"paths": {"artifacts": str(art)}, "normalize": {"norm_dir": "norm_tl"}}, str(art / "ce")


def test_export_then_join_keeps_feature_row_order(tmp_path):
    cfg, ce_dir = _run(tmp_path)
    os.makedirs(ce_dir)
    ce_export.export(cfg, "dev", ce_dir, top=2)
    pairs = pd.read_parquet(os.path.join(ce_dir, "score_dev.parquet"))
    assert pairs.values.tolist() == [["S1-b", "S2-z"], ["S1-a", "S2-x"], ["S1-a", "S3-y"]]
    texts = pd.read_parquet(os.path.join(ce_dir, "texts_dev.parquet")).set_index("entity_id")["text"]
    assert texts["S2-z"] == "टाटा | Pune" and texts["S1-b"] == "Tata Pvt Ltd | "
    # the kernel returns scores in any order
    pairs.assign(ce_p=[0.1, 0.9, 0.8]).iloc[::-1].to_parquet(os.path.join(ce_dir, "ce_dev.parquet"))
    ce_export.join(cfg, "dev", ce_dir, top=2, min_cover=1.0)
    got = pd.read_parquet(os.path.join(ce_dir, "ce_dev_US.parquet"))
    assert got["qi"].tolist() == [1, 0, 0] and np.allclose(got["ce_p"], [0.1, 0.9, 0.8])


def test_join_refuses_scores_from_another_run(tmp_path):
    cfg, ce_dir = _run(tmp_path)
    os.makedirs(ce_dir)
    pd.DataFrame({"s1_id": ["S1-a"], "cand_id": ["S2-x"], "ce_p": [0.5]}).to_parquet(os.path.join(ce_dir, "ce_dev.parquet"))
    with pytest.raises(AssertionError, match="have ce_p"):
        ce_export.join(cfg, "dev", ce_dir, top=2, min_cover=0.98)


def test_top_slice_leaves_the_rest_missing(tmp_path):
    cfg, ce_dir = _run(tmp_path)
    os.makedirs(ce_dir)
    ce_export.export(cfg, "dev", ce_dir, top=1)
    pairs = pd.read_parquet(os.path.join(ce_dir, "score_dev.parquet"))
    assert pairs.values.tolist() == [["S1-b", "S2-z"], ["S1-a", "S2-x"]]
    pairs.assign(ce_p=[0.1, 0.9]).to_parquet(os.path.join(ce_dir, "ce_dev.parquet"))
    ce_export.join(cfg, "dev", ce_dir, top=1, min_cover=1.0)
    got = pd.read_parquet(os.path.join(ce_dir, "ce_dev_US.parquet"))
    assert np.allclose(got["ce_p"][:2], [0.1, 0.9]) and np.isnan(got["ce_p"][2])
