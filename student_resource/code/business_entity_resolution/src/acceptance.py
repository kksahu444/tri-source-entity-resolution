"""Acceptance check for a submission file before it is uploaded (runs/decoy_shift.md).

Upload 2 (dev OOF 0.98049) scored 0.938 public because it linked test decoy entities: same name and street
as the S1, a slightly higher house number. Dev could not see it. This check compares the submission's link
structure with the structure of TRUE links in train (ground truth), per S1:
  links            all links
  near             links whose house number is 1-30 away from the S1's (same sorted core name and street)
  near_beside      near links of S1s that also have a link at their exact house number (the decoy shape)
  near_broad       same city, same first core word, house 1-30 higher (covers word-changed decoys), reported
  near_broad_beside   near_broad links of S1s that also have an exact-house link of that kind, reported
A file fails when near_beside per S1 exceeds the train level by more than ``--tol`` (relative), in any
country with a train reference; France is compared with the US/India average. Nothing is fitted: the
file is accepted or rejected.

python -m src.acceptance --config config_v5.yaml --file ../../submissions/upload_2/matching_results.tsv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import io_utils
from .normalize import norm_path
from .utils import load_config

COLS = ["entity_id", "country", "core", "street", "house", "city"]


def _table(cfg, split, k):
    return pq.read_table(norm_path(cfg, split, k), columns=COLS).to_pandas().set_index("entity_id")


def link_structure(pairs: pd.DataFrame, s1: pd.DataFrame, rec: pd.DataFrame) -> pd.DataFrame:
    """pairs: s1_id, cand_id. Returns per-country rates per S1 (denominator: every S1 of that country)."""
    a, b = s1.reindex(pairs["s1_id"]), rec.reindex(pairs["cand_id"])
    sk = lambda s: s.fillna("").map(lambda x: " ".join(sorted(set(x.split())))).to_numpy()
    same = (sk(a["core"]) == sk(b["core"])) & (a["street"].fillna("").to_numpy() == b["street"].fillna("").to_numpy()) \
        & (a["street"].fillna("").to_numpy() != "")
    ha = pd.to_numeric(a["house"], errors="coerce").to_numpy()
    hb = pd.to_numeric(b["house"], errors="coerce").to_numpy()
    d = np.abs(ha - hb)
    off = hb - ha
    city = (a["city"].fillna("").to_numpy() == b["city"].fillna("").to_numpy()) & (a["city"].fillna("").to_numpy() != "")
    first = np.array([x.split()[:1] == y.split()[:1] and bool(x) for x, y in zip(a["core"].fillna(""), b["core"].fillna(""))])
    x = pd.DataFrame({"s1_id": pairs["s1_id"].to_numpy(), "country": a["country"].to_numpy(),
                      "exact": same & (d == 0), "near": same & (d >= 1) & (d <= 30),
                      "exact_broad": city & first & (d == 0), "near_broad": city & first & (off >= 1) & (off <= 30)})
    has_exact = x.groupby("s1_id")["exact"].transform("any").to_numpy()
    x["near_beside"] = x["near"] & has_exact
    x["near_broad_beside"] = x["near_broad"] & x.groupby("s1_id")["exact_broad"].transform("any").to_numpy()
    n_s1 = s1.groupby("country").size()
    out = x.groupby("country")[["near", "near_beside", "near_broad", "near_broad_beside"]].sum()
    out["links"] = x.groupby("country").size()
    return out.div(n_s1.reindex(out.index), axis=0)


def _pairs_from_lists(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    df = df[df[right].fillna("") != ""]
    x = df.assign(cand_id=df[right].str.split(",")).explode("cand_id")
    return pd.DataFrame({"s1_id": x[left].to_numpy(), "cand_id": x["cand_id"].to_numpy()})


def train_reference(cfg) -> pd.DataFrame:
    gt = io_utils.load_ground_truth(cfg["paths"]["dataset"])
    s1 = _table(cfg, "train", 1)
    rec = pd.concat([_table(cfg, "train", k) for k in (2, 3)])
    return link_structure(gt, s1, rec)


def check(cfg, path: str, tol: float = 0.5) -> dict:
    sub = io_utils.read_tsv(path)
    pairs = _pairs_from_lists(sub, "source1_entity_id", "matched_entity_ids")
    s1 = _table(cfg, "test", 1)
    rec = pd.concat([_table(cfg, "test", k) for k in (2, 3)])
    test = link_structure(pairs, s1, rec)
    ref = train_reference(cfg)
    ref.loc["France"] = ref.mean()
    res = {"file": path, "tol": tol, "countries": {}, "pass": True}
    for c in test.index:
        row = {k: round(float(test.loc[c, k]), 4) for k in test.columns}
        row.update({f"train_{k}": round(float(ref.loc[c, k]), 4) for k in ref.columns})
        limit = ref.loc[c, "near_beside"] * (1 + tol)
        row["fail"] = bool(test.loc[c, "near_beside"] > limit)
        res["pass"] &= not row["fail"]
        res["countries"][c] = row
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v5.yaml")
    ap.add_argument("--file", required=True)
    ap.add_argument("--tol", type=float, default=0.5)
    args = ap.parse_args(argv)
    res = check(load_config(args.config), args.file, args.tol)
    print(json.dumps(res, indent=1))
    out = os.path.join(load_config(args.config)["paths"]["runs"], "acceptance.jsonl")
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
