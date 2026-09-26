"""Stage `prep`: mine vocabularies, then normalise every source file to parquet (D1)."""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import io_utils
from .normalize import fold, make_pool, mine_vocab, norm_path, normalize_frame
from .parse_address import clean_part
from .utils import Timer, get_logger

log = get_logger()


def _parts(addr: str, translit: bool) -> list[str]:
    return [c for c in (clean_part(p) for p in fold(addr, translit).split(",")) if c]


def mine_region_alias(cfg: dict, vocab: dict, n_pairs: int = 400_000) -> dict:
    """Region spellings of S2/S3 (full state names, codes, native scripts) mapped to the S1 region.

    A record part p becomes an alias of region R when, over sampled true pairs whose S1 region is R
    and whose record does not contain R, p co-occurs with R at least 30 times, and p almost never
    appears together with R itself (so cities, which usually sit next to the state, are excluded).
    """
    root, translit = cfg["paths"]["dataset"], cfg["flags"].get("TRANSLITERATE", True)
    pairs = io_utils.load_ground_truth(root).sample(n_pairs, random_state=cfg["seed"])
    s1 = io_utils.load_source(root, "train", 1).set_index("entity_id")
    need = set(pairs["cand_id"])
    recs = pd.concat([df[df["entity_id"].isin(need)] for df in
                      (io_utils.load_source(root, "train", k) for k in (2, 3))]).set_index("entity_id")
    regions = {c: set(v) for c, v in vocab["regions"].items()}
    co, seen = Counter(), Counter()
    for s1_id, r_id in zip(pairs["s1_id"], pairs["cand_id"]):
        c = s1.at[s1_id, "country"]
        p1 = _parts(s1.at[s1_id, "business_address"], translit)
        if not p1 or p1[-1] not in regions.get(c, ()):
            continue
        R = p1[-1]
        pr = set(_parts(recs.at[r_id, "business_address"], translit))
        for p in pr:
            if p in regions[c] or len(p.split()) > 3:
                continue
            seen[(c, p)] += 1
            if R not in pr:
                co[(c, p, R)] += 1
    alias: dict[str, dict[str, str]] = {}
    for (c, p, R), n in co.items():
        if n >= 30 and n / seen[(c, p)] >= 0.9:
            alias.setdefault(c, {})[p] = R
    return alias


def run(cfg: dict) -> dict:
    t = Timer()
    root, art = cfg["paths"]["dataset"], cfg["paths"]["artifacts"]
    os.makedirs(os.path.join(art, "norm"), exist_ok=True)
    vocab_path = os.path.join(art, "vocab.json")
    if os.path.exists(vocab_path):
        vocab = json.load(open(vocab_path, encoding="utf-8"))
    else:
        s1 = [io_utils.load_source(root, s, 1) for s in ("train", "test")]
        with make_pool(cfg) as pool:
            vocab = mine_vocab(s1, cfg, pool)
        del s1
        vocab["region_alias"] = mine_region_alias(cfg, vocab)
        json.dump(vocab, open(vocab_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log.info("vocab: legal %s | regions %s | aliases %s (%.0fs)",
             {c: len(v) for c, v in vocab["legal"].items()}, {c: len(v) for c, v in vocab["regions"].items()},
             {c: len(v) for c, v in vocab["region_alias"].items()}, t())
    stats = {}
    with make_pool(cfg, vocab) as pool:
        for split in ("test", "train"):
            for k in (1, 2, 3):
                out = norm_path(cfg, split, k, "norm")
                if os.path.exists(out):
                    continue
                df = io_utils.load_source(root, split, k)
                writer = None
                for i in range(0, len(df), 1_000_000):
                    nd = normalize_frame(df.iloc[i:i + 1_000_000], pool)
                    nd["src"] = np.int8(k)
                    table = pa.Table.from_pandas(nd, preserve_index=False)
                    writer = writer or pq.ParquetWriter(out + ".tmp", table.schema)
                    writer.write_table(table)
                writer.close()
                os.replace(out + ".tmp", out)
                stats[f"{split}_s{k}"] = len(df)
                log.info("normalised %s s%d: %s rows (%.0fs)", split, k, f"{len(df):,}", t())
                del df, nd, table
    return {"runtime_s": t(), **stats}
