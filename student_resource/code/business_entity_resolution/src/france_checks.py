"""Label-free checks of a test submission, per country (France has no labels; runs/france_research.md).

For each country: links per S1, empty-list rate, the share of S2/S3 records in no candidate list (orphans),
and links per S1 by the S1's name-crowd size (S1s of the country sharing its sorted core name). Train truth is
flat at ~3.46 links per S1 in every crowd bucket, so a steep drop with crowd size is lost recall. Compare a new
file with a baseline (upload 1 by default).

    python -m src.france_checks --config config_v7.yaml --matching <new>/matching_results.tsv \
        --candidates <new>/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .blocking.union import block_dir
from .utils import get_logger, load_config

log = get_logger()
BINS = [0, 1, 4, 29, 10**9]
LABELS = ["1", "2-4", "5-29", "30+"]


def _lists(path: str) -> pd.Series:
    m = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return m.set_index(m.columns[0])[m.columns[1]]


def orphan_share(cand_path: str, rec_ids: pd.Index) -> float:
    """Share of the given records that appear in no candidate list (streams the file)."""
    pos = {e: i for i, e in enumerate(rec_ids)}
    hit = np.zeros(len(rec_ids), bool)
    with open(cand_path, encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            ids = line.rstrip("\n").split("\t", 1)[1]
            for e in ids.split(",") if ids else ():
                i = pos.get(e)
                if i is not None:
                    hit[i] = True
    return float(1 - hit.mean())


def country_stats(cfg: dict, c: str, match_path: str, cand_path: str | None) -> dict:
    q = pd.read_parquet(os.path.join(block_dir(cfg), f"test_{c}_q.parquet"), columns=["entity_id", "core"])
    key = q["core"].map(lambda x: " ".join(sorted(set(x.split()))))
    crowd = pd.cut(key.map(key.value_counts()), BINS, labels=LABELS)
    lists = _lists(match_path).reindex(q["entity_id"]).fillna("")
    n = lists.map(lambda s: len(s.split(",")) if s else 0).to_numpy()
    out = {"links_per_s1": round(float(n.mean()), 4), "empty_rate": round(float((n == 0).mean()), 4),
           "links_by_crowd": {k: round(float(n[(crowd == k).to_numpy()].mean()), 4) for k in LABELS},
           "crowd_share": {k: round(float((crowd == k).mean()), 4) for k in LABELS}}
    if cand_path:
        r = pd.read_parquet(os.path.join(block_dir(cfg), f"test_{c}_r.parquet"), columns=["entity_id"])
        out["orphan_share"] = round(orphan_share(cand_path, pd.Index(r["entity_id"])), 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_v7.yaml")
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--base-matching", default="../../submissions/upload_1/matching_results.tsv")
    ap.add_argument("--base-candidates", default="../../submissions/upload_1/candidate_pairs.tsv")
    ap.add_argument("--countries", default="France,US,India")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    res = {}
    for c in args.countries.split(","):
        res[c] = {"new": country_stats(cfg, c, args.matching, args.candidates),
                  "base": country_stats(cfg, c, args.base_matching,
                                        args.base_candidates if args.candidates else None)}
        log.info("%s: %s", c, json.dumps(res[c]))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
