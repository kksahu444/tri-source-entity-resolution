"""Section 8: dev universe (city-group sample + orphan distractors) and grouped, stratified folds."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import io_utils
from .normalize import norm_path


def city_key(df: pd.DataFrame) -> pd.Series:
    key = df["country"].astype(str) + "|" + df["city"].astype(str)
    return key.where(df["city"].astype(str) != "", df["country"].astype(str) + "|id|" + df["entity_id"].astype(str))


def match_counts(cfg: dict) -> pd.Series:
    gt = io_utils.load_ground_truth(cfg["paths"]["dataset"])
    return gt.groupby("s1_id").size()


def build_dev_universe(cfg: dict) -> pd.DataFrame:
    """Train S1 rows with columns: entity_id, country, city_key, n_true, bucket, role, fold.

    role: 'query' (dev S1 we predict), 'orphan' (dev S1 withheld so its records act as distractors,
    D0 block 2), 'out' (not in the dev universe). fold: 0..4 for queries (LightGBM stages),
    ce_fold: 0..2 (cross-encoder, same groups).
    """
    path = os.path.join(cfg["paths"]["artifacts"], "dev_universe.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    rng = np.random.default_rng(cfg["seed"])
    s1 = pd.read_parquet(norm_path(cfg, "train", 1), columns=["entity_id", "country", "city"])
    s1["city_key"] = city_key(s1)
    n = match_counts(cfg)
    s1["n_true"] = s1["entity_id"].map(n).fillna(0).astype(np.int16)
    s1["bucket"] = s1["n_true"].clip(upper=3).astype(np.int8)
    groups = s1["city_key"].unique()
    chosen = set(groups[rng.random(len(groups)) < cfg["dev"]["city_group_share"]])
    in_dev = s1["city_key"].isin(chosen).to_numpy()
    orphan = in_dev & (rng.random(len(s1)) < cfg["dev"]["orphan_share"])
    s1["role"] = np.where(orphan, "orphan", np.where(in_dev, "query", "out"))
    s1["fold"] = -1
    s1["ce_fold"] = -1
    q = s1["role"].eq("query").to_numpy()
    strat = (s1.loc[q, "country"] + "_" + s1.loc[q, "bucket"].astype(str)).to_numpy()
    grp = s1.loc[q, "city_key"].to_numpy()
    idx = np.flatnonzero(q)
    for n_splits, col in ((5, "fold"), (3, "ce_fold")):
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
        fold = np.full(len(idx), -1)
        for f, (_, te) in enumerate(skf.split(idx, strat, grp)):
            fold[te] = f
        s1.loc[q, col] = fold
    s1 = s1.drop(columns=["city"])
    s1.to_parquet(path, index=False)
    return s1
