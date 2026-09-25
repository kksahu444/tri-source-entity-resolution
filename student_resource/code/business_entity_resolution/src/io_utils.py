"""TSV reading and submission writing exactly as the spec requires (sections 3 and 10)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def count_data_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f) - 1


def read_tsv(path: str) -> pd.DataFrame:
    """Default quoting (CSV-escaped quotes decode correctly), every field a string, no NaN."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    n = count_data_lines(path)
    assert len(df) == n, f"{path}: parsed {len(df)} rows but the file has {n} data lines"
    return df


def source_path(root: str, split: str, k: int) -> str:
    return os.path.join(root, split, f"{split}_source{k}.tsv")


def load_source(root: str, split: str, k: int) -> pd.DataFrame:
    df = read_tsv(source_path(root, split, k))
    assert list(df.columns) == ["entity_id", "business_name", "business_address", "country"], df.columns
    return df


def load_ground_truth(root: str) -> pd.DataFrame:
    """Long table (s1_id, cand_id) of true pairs plus the per-S1 match count."""
    gt = read_tsv(os.path.join(root, "train", "train_ground_truth.tsv"))
    pairs = gt.assign(cand_id=gt["matched_entity_ids"].str.split(",")).explode("cand_id")
    pairs = pairs[pairs["cand_id"].fillna("") != ""]
    return pairs.rename(columns={"source1_entity_id": "s1_id"})[["s1_id", "cand_id"]].reset_index(drop=True)


def _write_lists(path: str, header: tuple[str, str], s1_order: list[str], lists: dict[str, list[str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for s1 in s1_order:
            f.write(f"{s1}\t{','.join(lists.get(s1, ()))}\n")


def _group(s1_ids: np.ndarray, cand_ids: np.ndarray, valid: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for s1, c in zip(s1_ids.tolist(), cand_ids.tolist()):
        if c not in valid or c[:3] not in ("S2-", "S3-"):
            raise ValueError(f"invalid id {c!r} for {s1}")
        lst = out.setdefault(s1, [])
        if c not in lst:
            lst.append(c)
    return out


def write_submission(match_path: str, cand_path: str, test_s1_ids: list[str], valid_ids: set[str],
                     match_s1: np.ndarray, match_cand: np.ndarray,
                     cand_s1: np.ndarray, cand_cand: np.ndarray) -> dict:
    """Write both files, one row per test S1 in test_source1.tsv order; matches must be candidates."""
    assert len(set(test_s1_ids)) == len(test_s1_ids)
    required = set(test_s1_ids)
    assert set(np.unique(cand_s1)) <= required and set(np.unique(match_s1)) <= required
    cands = _group(np.asarray(cand_s1), np.asarray(cand_cand), valid_ids)
    matches = _group(np.asarray(match_s1), np.asarray(match_cand), valid_ids)
    for s1, lst in matches.items():
        missing = set(lst) - set(cands.get(s1, ()))
        assert not missing, f"{s1}: matched ids not in its candidate list: {sorted(missing)[:5]}"
    _write_lists(cand_path, CAND_HEADER, test_s1_ids, cands)
    _write_lists(match_path, MATCH_HEADER, test_s1_ids, matches)
    return {"rows": len(test_s1_ids), "s1_with_candidates": len(cands), "s1_with_matches": len(matches),
            "n_candidates": int(sum(map(len, cands.values()))), "n_matches": int(sum(map(len, matches.values())))}
