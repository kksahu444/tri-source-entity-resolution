"""Exploratory data analysis: the numbers that drive blocking and decision design.

Answers, among others:
* how many Source-1 entities are singletons (drives how conservative the decision must be),
* whether a Source-2/3 record ever matches more than one Source-1 entity (validates exclusivity),
* whether matches ever cross country labels (can blocking stay within a country?),
* how often postcodes / landmarks / empty fields occur per source and country.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from .io import load_split
from .normalize import normalize_records

log = logging.getLogger(__name__)


def _field_stats(recs: pd.DataFrame) -> dict:
    out = {}
    for (src, country), g in recs.groupby(["source", "country"]):
        out[f"S{src}|{country}"] = {
            "n": int(len(g)),
            "empty_name": float((g["business_name"] == "").mean()),
            "empty_address": float((g["business_address"] == "").mean()),
            "has_postcode": float((g["addr_postcodes"] != "").mean()),
            "has_landmark": float(g["has_landmark"].mean()),
            "has_legal_form": float((g["name_legal"] != "").mean()),
            "has_dba": float((g["name_variants"] != "").mean()),
            "name_len_median": float(g["business_name"].str.len().median()),
            "addr_len_median": float(g["business_address"].str.len().median()),
        }
    return out


def run_eda(data_dir: str, split: str = "train", out_dir: str = "artifacts", n_examples: int = 15) -> dict:
    """Print and save (``<out_dir>/eda_<split>.json``) a summary of one split."""
    data = load_split(data_dir, split)
    recs = normalize_records(data.records)
    report: dict = {
        "split": split,
        "records_per_source": {f"S{k}": int(v) for k, v in recs["source"].value_counts().sort_index().items()},
        "country_per_source": {f"S{k}": {str(c): int(n) for c, n in g["country"].value_counts().items()}
                               for k, g in recs.groupby("source")},
        "fields": _field_stats(recs),
    }
    gt = data.ground_truth
    if gt is not None:
        sizes = pd.Series({s1: len(gt.get(s1, [])) for s1 in data.s1_ids})
        owners = Counter(c for ids in gt.values() for c in ids)
        country = recs.set_index("entity_id")["country"]
        cross = sum(1 for s1, ids in gt.items() for c in ids if country.get(s1) != country.get(c))
        n_pairs = int(sizes.sum())
        targets = recs[recs["source"] != 1]
        report["labels"] = {
            "n_s1": int(len(sizes)),
            "singleton_rate": float((sizes == 0).mean()),
            "matches_per_s1": {str(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
            "n_true_pairs": n_pairs,
            "pairs_to_S2": int(sum(1 for ids in gt.values() for c in ids if c.startswith("S2-"))),
            "pairs_to_S3": int(sum(1 for ids in gt.values() for c in ids if c.startswith("S3-"))),
            "target_ids_matched_to_multiple_s1": int(sum(1 for v in owners.values() if v > 1)),
            "cross_country_pairs": int(cross),
            "unmatched_target_record_rate": float(1 - len(owners) / max(len(targets), 1)),
            "gt_ids_not_in_split": int(sum(1 for c in owners if c not in country.index)),
        }
        by_id = recs.set_index("entity_id")
        examples = []
        for s1 in sizes[sizes > 0].sample(min(n_examples, int((sizes > 0).sum())), random_state=0).index:
            row = {"S1": f"{by_id.at[s1, 'business_name']} | {by_id.at[s1, 'business_address']}"}
            for c in gt[s1]:
                if c in by_id.index:
                    row[c] = f"{by_id.at[c, 'business_name']} | {by_id.at[c, 'business_address']}"
            examples.append(row)
        report["examples"] = examples

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"eda_{split}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2, ensure_ascii=False))
    log.info("EDA written to %s", path)
    return report
