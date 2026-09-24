"""Reading the challenge TSVs and writing submission files.

All challenge files are tab-separated because names, addresses and ID lists contain commas.
Everything is read as strings (IDs like ``S1-000123`` must never become numbers or NaN).
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCH_COLUMN = "matched_entity_ids"
CANDIDATE_COLUMN = "candidate_entity_ids"


@dataclass
class SplitData:
    """One split (train or test): all records of the three sources plus optional labels."""

    records: pd.DataFrame  # RECORD_COLUMNS + "source" (1, 2 or 3)
    s1_ids: list[str]  # Source-1 ids in file order (the submission row order)
    ground_truth: dict[str, list[str]] | None  # S1 id -> matched S2/S3 ids (train only)


def _count_data_lines(path: Path) -> int:
    with open(path, encoding="utf-8-sig") as fh:
        return max(sum(1 for line in fh if line.strip()) - 1, 0)


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a challenge TSV as all-string columns.

    A stray double quote inside an address makes the default CSV quoting swallow following
    lines; if the parsed row count disagrees with the physical line count we re-read with
    quoting disabled.
    """
    path = Path(path)
    opts = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8-sig")
    df = pd.read_csv(path, **opts)
    n_lines = _count_data_lines(path)
    if len(df) != n_lines:
        alt = pd.read_csv(path, quoting=csv.QUOTE_NONE, **opts)
        if len(alt) == n_lines:
            log.warning("%s: quote characters broke default parsing; re-read with QUOTE_NONE", path.name)
            df = alt
        else:
            log.warning("%s: parsed %d rows but file has %d data lines", path.name, len(df), n_lines)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def parse_id_list(value: str) -> list[str]:
    """``"S2-1,S3-9"`` -> ``["S2-1", "S3-9"]`` (order kept, blanks and duplicates removed)."""
    seen: dict[str, None] = {}
    for part in (value or "").split(","):
        part = part.strip()
        if part:
            seen.setdefault(part, None)
    return list(seen)


def read_ground_truth(path: str | Path) -> dict[str, list[str]]:
    df = read_tsv(path)
    missing = set(GT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return {row.source1_entity_id: parse_id_list(row.matched_entity_ids) for row in df.itertuples(index=False)}


def load_split(data_dir: str | Path, split: str) -> SplitData:
    """Load ``<data_dir>/<split>/<split>_source{1,2,3}.tsv`` (+ ground truth for train)."""
    split_dir = Path(data_dir) / split
    frames = []
    for source in (1, 2, 3):
        path = split_dir / f"{split}_source{source}.tsv"
        if not path.exists():
            raise FileNotFoundError(f"missing {path} - copy the challenge dataset into {Path(data_dir).resolve()}")
        df = read_tsv(path)
        missing = set(RECORD_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        bad_prefix = ~df["entity_id"].str.startswith(f"S{source}-")
        if bad_prefix.any():
            log.warning("%s: %d ids without prefix S%d-", path.name, int(bad_prefix.sum()), source)
        df = df[RECORD_COLUMNS].copy()
        df["source"] = source
        frames.append(df)
    records = pd.concat(frames, ignore_index=True)
    dupes = records["entity_id"].duplicated()
    if dupes.any():
        log.warning("%s: dropping %d duplicate entity_id rows", split, int(dupes.sum()))
        records = records[~dupes].reset_index(drop=True)

    gt_path = split_dir / f"{split}_ground_truth.tsv"
    ground_truth = read_ground_truth(gt_path) if gt_path.exists() else None
    s1_ids = records.loc[records["source"] == 1, "entity_id"].tolist()
    log.info(
        "%s: %s records (S1=%d, S2=%d, S3=%d), labels=%s",
        split,
        f"{len(records):,}",
        *(int((records["source"] == s).sum()) for s in (1, 2, 3)),
        ground_truth is not None,
    )
    return SplitData(records=records, s1_ids=s1_ids, ground_truth=ground_truth)


def write_id_lists(path: str | Path, s1_ids: list[str], mapping: dict[str, list[str]], column: str) -> None:
    """Write one row per Source-1 id: ``<s1>\\t<id,id,...>`` (empty list -> empty field), no quoting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"source1_entity_id\t{column}\n")
        for s1 in s1_ids:
            ids = parse_id_list(",".join(mapping.get(s1, [])))
            fh.write(f"{s1}\t{','.join(ids)}\n")
