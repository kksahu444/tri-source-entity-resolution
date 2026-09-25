"""D1 normalisation: view A (cross-encoder input) and view B (blocking and features).

Only data-mined vocabularies plus the problem statement's own examples are used (flag C-5):
  * legal-form phrases: the frequent 1-3-token name endings per country (share >= legal_min_share);
  * regions: the frequent last address parts of S1 per country;
  * canonical legal tokens: Corp/Corporation, Pvt/Private, Ltd/Limited (problem statement), plus
    Inc/Incorporated and Co/Company as "legal-suffix variation".
"""
from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .parse_address import FIELDS as ADDR_FIELDS
from .parse_address import clean_part, parse_address

try:
    from anyascii import anyascii
except ImportError:  # pragma: no cover
    anyascii = None

WS = re.compile(r"\s+")
TAG = re.compile(r"\((?:india|france)\)")
LEAD_JUNK = re.compile(r"""^(?:[\s(#@<>"'*\-.~\[\]_=+:;,|!?/\\])+""")
TRAIL_BANG = re.compile(r"!+\s*$")
DBA = re.compile(r"\b(?:d\s*/\s*b\s*/\s*a|dba|doing business as|t\s*/\s*a|trading as)\b")
NON_ALNUM = re.compile(r"[^0-9a-z]+")
LONG_DIGITS = re.compile(r"\b\d{5,}\b")
LEGAL_CANON = {"corporation": "corp", "private": "pvt", "limited": "ltd", "incorporated": "inc", "company": "co"}
VOWELS = re.compile(r"(?<=.)[aeiou]")
DOUBLE = re.compile(r"(.)\1+")


def skeleton(tok: str) -> str:
    """Consonant skeleton (D1 step 9): vowels after the first letter dropped, doubled letters collapsed."""
    return DOUBLE.sub(r"\1", VOWELS.sub("", tok)) if not tok.isdigit() else tok


# romanised Indic spellings of the statement's legal words ("praivet" -> pvt) share their skeleton
SKEL_LEGAL = {skeleton(k): v for k, v in LEGAL_CANON.items()}


def canon_token(t: str) -> str:
    if t in LEGAL_CANON:
        return LEGAL_CANON[t]
    if len(t) >= 5:
        return SKEL_LEGAL.get(skeleton(t), t)
    return t

NAME_FIELDS = ("name_b", "core", "legal", "alt", "f_tag", "f_lead", "f_bang", "f_digits", "f_nonlatin")


def view_a(s: str) -> str:
    return WS.sub(" ", unicodedata.normalize("NFKC", s)).strip()


def fold(s: str, translit: bool = True) -> str:
    """NFKC + lower-case + accent folding; non-Latin scripts romanised with anyascii when enabled."""
    s = unicodedata.normalize("NFKC", s).lower()
    if s.isascii():
        return s
    if translit and anyascii is not None:
        return anyascii(s).lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def name_tokens(folded: str) -> tuple[list[str], list[str], dict]:
    """Junk removal (flagged), & -> and, dotted abbreviations joined, DBA split. Returns
    (tokens of the main name, tokens of the DBA alternative, flags)."""
    s = folded
    flags = {"f_tag": int(bool(TAG.search(s)))}
    s = TAG.sub(" ", s).strip()
    m = LEAD_JUNK.match(s)
    flags["f_lead"] = int(bool(m))
    if m:
        s = s[m.end():]
    flags["f_bang"] = int(bool(TRAIL_BANG.search(s)))
    s = TRAIL_BANG.sub("", s)
    s = s.replace("&", " and ").replace(".", "").replace("'", "").replace("’", "")
    s, n_long = LONG_DIGITS.subn(" ", s)
    flags["f_digits"] = int(n_long > 0)
    parts = [p for p in DBA.split(s) if p.strip()]
    toks = lambda p: [canon_token(t) for t in NON_ALNUM.sub(" ", p).split()]
    main = toks(parts[0]) if parts else []
    alt = toks(parts[1]) if len(parts) > 1 else []
    return main, alt, flags


def split_legal(toks: list[str], vocab: frozenset) -> tuple[list[str], list[str]]:
    """Move legal-form tokens into their own field wherever they sit (the noise moves them to the
    front and appends words after them); keep >= 1 core token."""
    legal = [t for t in toks if t in vocab]
    core = [t for t in toks if t not in vocab]
    while core and core[-1] == "and":
        core.pop()
    while core and core[0] == "and":
        core.pop(0)
    if not core:
        return toks[:1], legal[1:] if toks and toks[0] in vocab else legal
    return core, legal


# ------------------------------------------------------------------ vocab mining
STATEMENT_LEGAL = ("corp", "pvt", "ltd")   # problem statement: Corp/Corporation, Pvt/Private, Ltd/Limited


def _fold_chunk(args):
    names, addrs, translit = args
    out_n, out_a = [], []
    for n, a in zip(names, addrs):
        main, _, _ = name_tokens(fold(n, translit))
        out_n.append(main)
        out_a.append([c for c in (clean_part(p) for p in fold(a, translit).split(",")) if c])
    return out_n, out_a


def mine_vocab(s1_frames: list[pd.DataFrame], cfg: dict, pool: Pool) -> dict:
    """Legal-form tokens and regions per country, mined from S1 names/addresses of every split.

    legal:   last name token of >= legal_min_share of names AND at the end in >= 85% of the names
             that contain it (llc, inc, ltd, sarl, ...; generic endings such as "center" fail the
             second test), plus the problem statement's Corp/Pvt/Ltd.
    regions: last address part of >= region_min_share of addresses AND last in >= 50% of the
             addresses that contain it (states/regions ~0.85; cities of reordered addresses ~0.07).
    """
    ncfg, translit = cfg["normalize"], cfg["flags"].get("TRANSLITERATE", True)
    df = pd.concat([f[["business_name", "business_address", "country"]] for f in s1_frames], ignore_index=True)
    chunks = [(df["business_name"].iloc[i:i + 50_000].tolist(), df["business_address"].iloc[i:i + 50_000].tolist(),
               translit) for i in range(0, len(df), 50_000)]
    names, parts = [], []
    for n, a in pool.imap(_fold_chunk, chunks):
        names += n
        parts += a
    df["tok"], df["parts"] = names, parts
    vocab = {"legal": {}, "regions": {}}
    for c, g in df.groupby("country"):
        end, anyw = Counter(), Counter()
        for t in g["tok"]:
            if len(t) >= 2:
                end[t[-1]] += 1
                anyw.update(set(t))
        n = len(g)
        vocab["legal"][c] = sorted({t for t, v in end.items() if v >= ncfg["legal_min_share"] * n
                                    and v / anyw[t] >= 0.85 and not t.isdigit()} | set(STATEMENT_LEGAL))
        last, anyp = Counter(), Counter()
        for ps in g["parts"]:
            if ps:
                last[ps[-1]] += 1
                anyp.update(set(ps))
        vocab["regions"][c] = sorted(r for r, v in last.items()
                                     if v >= ncfg["region_min_share"] * n and v / anyp[r] >= 0.5)
    return vocab


# ------------------------------------------------------------------ record normalisation
_W: dict = {}


def _init_worker(vocab: dict, translit: bool) -> None:
    _W["legal"] = {c: frozenset(v) for c, v in vocab["legal"].items()}
    _W["regions"] = {c: {r: r for r in v} for c, v in vocab["regions"].items()}
    for c, alias in vocab.get("region_alias", {}).items():
        _W["regions"].setdefault(c, {}).update(alias)
    _W["translit"] = translit


def _norm_chunk(args):
    names, addrs, countries = args
    rows = []
    empty_legal, empty_reg = frozenset(), {}
    for n, a, c in zip(names, addrs, countries):
        fn = fold(n, _W["translit"])
        main, alt, flags = name_tokens(fn)
        core, legal = split_legal(main, _W["legal"].get(c, empty_legal))
        alt_core, _ = split_legal(alt, _W["legal"].get(c, empty_legal)) if alt else ([], [])
        nonlatin = int(any(ord(ch) > 0x24F for ch in n))
        rows.append((" ".join(main), " ".join(core), " ".join(sorted(set(legal))), " ".join(alt_core),
                     flags["f_tag"], flags["f_lead"], flags["f_bang"], flags["f_digits"], nonlatin)
                    + parse_address(fold(a, _W["translit"]), _W["regions"].get(c, empty_reg)))
    return rows


def normalize_frame(df: pd.DataFrame, pool: Pool, chunk: int = 50_000) -> pd.DataFrame:
    chunks = [(df["business_name"].iloc[i:i + chunk].tolist(), df["business_address"].iloc[i:i + chunk].tolist(),
               df["country"].iloc[i:i + chunk].tolist()) for i in range(0, len(df), chunk)]
    rows = []
    for r in pool.imap(_norm_chunk, chunks):
        rows += r
    out = pd.DataFrame(rows, columns=list(NAME_FIELDS) + list(ADDR_FIELDS))
    for col in ("f_tag", "f_lead", "f_bang", "f_digits", "f_nonlatin", "n_parts"):
        out[col] = out[col].astype(np.int8)
    out.insert(0, "entity_id", df["entity_id"].to_numpy())
    out.insert(1, "country", df["country"].to_numpy())
    out["name_a"] = [view_a(s) for s in df["business_name"]]
    out["addr_a"] = [view_a(s) for s in df["business_address"]]
    return out


def make_pool(cfg: dict, vocab: dict | None = None) -> Pool:
    translit = cfg["flags"].get("TRANSLITERATE", True)
    return Pool(cfg["normalize"]["workers"], initializer=_init_worker,
                initargs=(vocab or {"legal": {}, "regions": {}}, translit))


def norm_path(cfg: dict, split: str, k: int) -> str:
    return os.path.join(cfg["paths"]["artifacts"], "norm", f"{split}_s{k}.parquet")
