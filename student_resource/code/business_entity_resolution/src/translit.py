"""Romanised-token dictionary mined from training true pairs (C-5: vocabulary mined from provided data).

India S2/S3 names are often written in native scripts; anyascii romanises them ("vijn imphotek praivet"),
but romanisations differ from the Latin spelling S1 uses ("vision infotech private"). For true pairs
whose record name is non-Latin and whose token counts agree, tokens are aligned by position; a
romanised token t maps to Latin token s when the pair occurs >= min_count times and s is the most
frequent partner of t with share >= min_share. The map is applied to the name fields of every
non-Latin record (train and test) and the legal field is re-split with the country's vocabulary.
"""
from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import io_utils
from .normalize import norm_path, split_legal
from .utils import Timer, get_logger

log = get_logger()


SRC, DST = "norm", "norm_tl"


def mine(cfg: dict, min_count: int = 3, min_share: float = 0.5) -> dict:
    s1 = pq.read_table(norm_path(cfg, "train", 1, SRC), columns=["entity_id", "name_b"]).to_pandas().set_index("entity_id")["name_b"]
    recs = []
    for k in (2, 3):
        t = pq.read_table(norm_path(cfg, "train", k, SRC), columns=["entity_id", "name_b", "f_nonlatin"])
        t = t.filter(pc.equal(t["f_nonlatin"], 1))
        recs.append(t.select(["entity_id", "name_b"]).to_pandas())
    recs = pd.concat(recs).set_index("entity_id")["name_b"]
    gt = io_utils.load_ground_truth(cfg["paths"]["dataset"])
    gt = gt[gt["cand_id"].isin(recs.index)]
    pair_cnt, tok_cnt = Counter(), Counter()
    for a, b in zip(recs.reindex(gt["cand_id"]).to_numpy(), s1.reindex(gt["s1_id"]).to_numpy()):
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            continue
        for x, y in zip(ta, tb):
            tok_cnt[x] += 1
            if x != y:
                pair_cnt[(x, y)] += 1
    best = defaultdict(lambda: (None, 0))
    for (x, y), n in pair_cnt.items():
        if n > best[x][1]:
            best[x] = (y, n)
    mapping = {x: y for x, (y, n) in best.items() if n >= min_count and n / tok_cnt[x] >= min_share}
    log.info("translit dict: %d tokens mapped from %d aligned pairs", len(mapping), len(gt))
    return mapping


def apply(cfg: dict, mapping: dict, vocab: dict) -> dict:
    legal = {c: frozenset(v) for c, v in vocab["legal"].items()}
    stats = {}
    os.makedirs(os.path.dirname(norm_path(cfg, "train", 1, DST)), exist_ok=True)
    for split in ("train", "test"):
        shutil.copy(norm_path(cfg, split, 1, SRC), norm_path(cfg, split, 1, DST))
        for k in (2, 3):
            path = norm_path(cfg, split, k, DST)
            tbl = pq.read_table(norm_path(cfg, split, k, SRC))
            df = tbl.select(["name_b", "alt", "country", "f_nonlatin"]).to_pandas()
            idx = np.flatnonzero(df["f_nonlatin"].to_numpy() == 1)
            name_b, core, leg, alt = (tbl[c].to_pylist() for c in ("name_b", "core", "legal", "alt"))
            changed = 0
            for i in idx:
                toks = [mapping.get(t, t) for t in name_b[i].split()]
                new = " ".join(toks)
                if new == name_b[i]:
                    continue
                changed += 1
                c, lg = split_legal(toks, legal.get(df["country"].iat[i], frozenset()))
                name_b[i], core[i], leg[i] = new, " ".join(c), " ".join(sorted(set(lg)))
                if alt[i]:
                    alt[i] = " ".join(mapping.get(t, t) for t in alt[i].split())
            for col, vals in (("name_b", name_b), ("core", core), ("legal", leg), ("alt", alt)):
                tbl = tbl.set_column(tbl.schema.get_field_index(col), col, pa.array(vals, pa.string()))
            pq.write_table(tbl, path + ".tmp")
            os.replace(path + ".tmp", path)
            stats[f"{split}_s{k}"] = {"nonlatin": len(idx), "changed": changed}
            log.info("translit applied %s s%d: %s", split, k, stats[f"{split}_s{k}"])
    return stats


def run(cfg: dict, split: str | None = None) -> dict:
    t = Timer()
    art = cfg["paths"]["artifacts"]
    done = os.path.join(art, DST, "translit_applied.json")
    if os.path.exists(done):
        return {"stage": "translit", "notes": "already applied"}
    mapping = mine(cfg)
    json.dump(mapping, open(os.path.join(art, "translit_dict.json"), "w", encoding="utf-8"), ensure_ascii=False)
    vocab = json.load(open(os.path.join(art, "vocab.json"), encoding="utf-8"))
    stats = apply(cfg, mapping, vocab)
    json.dump(stats, open(done, "w"))
    return {"stage": "translit", "runtime_s": t(), "notes": json.dumps({"mapped_tokens": len(mapping), **stats})}
