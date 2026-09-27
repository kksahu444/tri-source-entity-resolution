"""Stage `prep`: mine vocabularies, then normalise every source file to parquet (D1)."""
from __future__ import annotations

import json
import os
import re
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


FIRST_TOK = re.compile(r"^([a-z]{1,4})(?= |\d|$)")
MARKED_STREET = re.compile(r"^([a-z]{1,4}) ?\d+[a-z]? [a-z]{2,}")


def numbering_markers(parts_lists, min_count: int = 200, min_share: float = 0.5) -> list[str]:
    """Numbering markers ("no", "ndeg" from N°): part-initial tokens of <= 4 letters followed by a number and
    street words ("no 5 r etienne marcel") in >= ``min_share`` of their part-initial uses and >= ``min_count``
    times. Unit parts ("apt 5", "fl 2") carry no street words, so unit words are not markers."""
    marked, anyf = Counter(), Counter()
    for parts in parts_lists:
        for p in parts:
            if (m := FIRST_TOK.match(p)):
                anyf[m.group(1)] += 1
                if MARKED_STREET.match(p):
                    marked[m.group(1)] += 1
    return sorted(t for t, n in marked.items() if n >= min_count and n / anyf[t] >= min_share)


def open_region_alias(s1_parts, rec_parts, regions: set, min_count: int = 30) -> dict:
    """Region spellings for a country without labelled pairs, from its S1 and record addresses only.

    The S1 table maps each city to its region (a digit-free part beside exactly one region part, >= 5 times,
    purity >= 0.9). A record part is an alias of region R when it is not a city, has no digits, sits last in
    >= half of its records, co-occurs (in records without a region part) with cities of R only (>= min_count,
    purity >= 0.9), and appears together with a region part in <= 10% of its records: a department written
    in place of the region ("nord" for "hauts de france"). Without it the department is taken for the city
    whenever the city comes first ("lille, 94 r jules valles, nord").
    """
    cr, tot = Counter(), Counter()
    for ps in s1_parts:
        rs = [p for p in ps if p in regions]
        if len(rs) != 1:
            continue
        for p in ps:
            if p not in regions and not p[:1].isdigit():
                cr[(p, rs[0])] += 1
                tot[p] += 1
    city = {p: R for (p, R), n in cr.items() if n >= 5 and n / tot[p] >= 0.9}
    seen, with_r, last, co = Counter(), Counter(), Counter(), Counter()
    for ps in rec_parts:
        if len(ps) < 2:
            continue
        has_r = any(p in regions for p in ps)
        cities = {city[p] for p in ps if p in city}
        for i, p in enumerate(ps):
            if p in regions or p in city or any(ch.isdigit() for ch in p) or len(p.split()) > 3:
                continue
            seen[p] += 1
            last[p] += i == len(ps) - 1
            if has_r:
                with_r[p] += 1
            elif len(cities) == 1:
                co[(p, next(iter(cities)))] += 1
    return {p: R for (p, R), n in co.items()
            if n >= min_count and n / max(seen[p] - with_r[p], 1) >= 0.9 and with_r[p] / seen[p] <= 0.1
            and last[p] / seen[p] >= 0.5}


def _address_parts(root: str, split: str, k: int, translit: bool, countries=None, sample: int = 0, seed: int = 0):
    df = io_utils.load_source(root, split, k)[["business_address", "country"]]
    if countries is not None:
        df = df[df["country"].isin(countries)]
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=seed)
    out = {}
    for c, g in df.groupby("country"):
        out[c] = [_parts(a, translit) for a in g["business_address"].fillna("")]
    return out


def mine_address_vocab(cfg: dict, vocab: dict, train_countries: set) -> None:
    """normalize.num_markers / normalize.open_region_alias (config_v7): label-free address vocabularies."""
    root, translit, seed = cfg["paths"]["dataset"], cfg["flags"].get("TRANSLITERATE", True), cfg["seed"]
    nc = cfg["normalize"]
    rec = {}
    for split in ("train", "test"):
        for k in (2, 3):
            for c, parts in _address_parts(root, split, k, translit, sample=nc.get("vocab_sample", 600_000),
                                           seed=seed).items():
                rec.setdefault(c, []).extend(parts)
    if nc.get("num_markers", False) and "num_markers" not in vocab:
        vocab["num_markers"] = {c: m for c, parts in rec.items() if (m := numbering_markers(parts))}
    if nc.get("open_region_alias", False) and not vocab.get("open_alias_done"):
        open_cs = [c for c in vocab["regions"] if c not in train_countries]
        s1 = {}
        for split in ("train", "test"):
            for c, parts in _address_parts(root, split, 1, translit, countries=open_cs).items():
                s1.setdefault(c, []).extend(parts)
        for c in open_cs:
            alias = open_region_alias(s1.get(c, []), rec.get(c, []), set(vocab["regions"][c]))
            if alias:
                vocab.setdefault("region_alias", {}).setdefault(c, {}).update(alias)
        vocab["open_alias_done"] = True


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
    nc = cfg["normalize"]
    if nc.get("zero_strip") and not vocab.get("addr_opts", {}).get("zero_strip"):
        vocab.setdefault("addr_opts", {})["zero_strip"] = True        # carried to the workers with the vocabularies
        json.dump(vocab, open(vocab_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if (nc.get("num_markers") and "num_markers" not in vocab) or (nc.get("open_region_alias") and not vocab.get("open_alias_done")):
        train_countries = set(io_utils.load_source(root, "train", 1)["country"].unique())
        mine_address_vocab(cfg, vocab, train_countries)
        json.dump(vocab, open(vocab_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log.info("vocab: legal %s | regions %s | aliases %s | numbering markers %s (%.0fs)",
             {c: len(v) for c, v in vocab["legal"].items()}, {c: len(v) for c, v in vocab["regions"].items()},
             {c: len(v) for c, v in vocab["region_alias"].items()}, vocab.get("num_markers", {}), t())
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
