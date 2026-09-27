"""Stage `block` (D2): union of C1 (TF-IDF), C2 (name crowd), C5 (keys) and optional C6 (second hop),
C7 (reverse name for empty-address records), C8 (address only) per country.

Outputs per (split, country) in artifacts/block/:
  {split}_{country}_q.parquet      query S1 rows (normalised view), row order = qi
  {split}_{country}_r.parquet      S2 u S3 rows of the same country, row order = ri
  {split}_{country}_pairs.parquet  union pairs with every channel's score/rank
The dev split reports pair completeness per channel and for the union (spec D2).
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

from .. import io_utils
from ..cv import build_dev_universe
from ..normalize import norm_path
from ..utils import Timer, get_logger
from .crowd import crowd_channel, key_channel, second_hop_channel
from .tfidf import TfidfSpace, combined, pair_cos, topk

log = get_logger()
BLOCK_COLS = ["entity_id", "core", "addr_core", "house", "street", "city", "src"]


def block_dir(cfg):
    d = os.path.join(cfg["paths"]["artifacts"], "block")
    os.makedirs(d, exist_ok=True)
    return d


def read_norm(cfg, split, k, country=None):
    filters = [("country", "=", country)] if country else None
    return pq.read_table(norm_path(cfg, split, k), filters=filters).to_pandas()


def sorted_key(s: pd.Series) -> pd.Series:
    return s.map(lambda x: " ".join(sorted(set(x.split()))))


def first_alpha(s: pd.Series, min_len=3) -> pd.Series:
    def f(x):
        for t in x.split():
            if len(t) >= min_len and not t[0].isdigit():
                return t
        return ""
    return s.map(f)


def rarest_alpha(s: pd.Series, df_counts: dict, min_len=3) -> pd.Series:
    """Rarest alphabetic token (by document frequency in the index) of each string: skips generic
    street-type words ("rue", "road") without a language-specific list."""
    def f(x):
        best, bdf = "", None
        for t in set(x.split()):
            if len(t) >= min_len and not t[0].isdigit():
                d = df_counts.get(t, 0)
                if bdf is None or d < bdf:
                    best, bdf = t, d
        return best
    return s.map(f)


def token_df(s: pd.Series) -> dict:
    from collections import Counter
    c = Counter()
    for x in s:
        c.update(set(x.split()))
    return c


def house_key(df: pd.DataFrame, street_df: dict | None = None) -> pd.Series:
    st = rarest_alpha(df["street"], street_df) if street_df is not None else first_alpha(df["street"])
    ct = first_alpha(df["city"], 2)
    tail = st.where(st != "", ct)
    return (df["house"] + "|" + tail).where((df["house"] != "") & (tail != ""), "")


def reverse_name_channel(rec_idx: np.ndarray, rn, qn, k: int, min_cos: float, chunk: int):
    """C7: the given records (empty address) retrieve their top-k S1 queries by name cosine.
    Returns (qi, ri) int32 arrays."""
    eq, ek, _ = topk(rn[rec_idx], qn.T.tocsr(), k, threshold=min_cos, chunk=chunk)
    return ek.astype(np.int32), rec_idx[eq].astype(np.int32)


def address_only_channel(qa, ra, k: int, min_cos: float, generate: int, cap: int, chunk: int):
    """C8: each S1 retrieves its top-k records by address cosine alone (candidates generated on the
    address features with df <= cap, then rescored exactly). Targets records whose name was replaced
    (no shared name token): C1 caps their score at 0.5 and ranks them below name look-alikes.
    Returns (qi, ri) int32 arrays."""
    ra_t = ra.T.tocsr()
    ra_t = sp.csr_matrix(sp.diags((np.diff(ra_t.indptr) <= cap).astype(np.float32)) @ ra_t)
    ra_t.eliminate_zeros()
    gq, gr, _ = topk(qa, ra_t, generate, threshold=0.0, chunk=chunk)
    s = pair_cos(qa, ra, gq, gr)
    keep = s >= min_cos
    aq, ar, _ = _topn_per_q(gq[keep], gr[keep], s[keep], k)
    return aq.astype(np.int32), ar.astype(np.int32)


def reverse_all_channel(rec_idx: np.ndarray, qn, qa, rn, ra, k: int, generate: int, cap: int, chunk: int,
                        min_score: float = 0.0):
    """C9: the given records retrieve their top-k S1s by the C1 score (mean of name and address cosines),
    candidates generated on S1 features with df <= cap and rescored exactly. On dense streets full of
    same-name businesses (France: 41 S1s per street, up to 420 S1s with one name in a city) a record with a
    misspelt street falls outside every S1's top-k list (test France: 4.1% of records in no list, US 2.6%),
    while its own best S1s are few. Returns (qi, ri) int32 arrays."""
    if len(rec_idx) == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32)
    qc_t = combined(qn, qa).T.tocsr()
    qc_t = sp.csr_matrix(sp.diags((np.diff(qc_t.indptr) <= cap).astype(np.float32)) @ qc_t)
    qc_t.eliminate_zeros()
    rc = combined(rn[rec_idx], ra[rec_idx])
    gl, gq, _ = topk(rc, qc_t, generate, threshold=0.0, chunk=chunk)
    del rc, qc_t
    gr = rec_idx[gl]
    s = 0.5 * (pair_cos(qn, rn, gq, gr) + pair_cos(qa, ra, gq, gr))
    keep = s >= min_score
    br, bq, _ = _topn_per_q(gr[keep], gq[keep], s[keep], k)
    return bq.astype(np.int32), br.astype(np.int32)


TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _shift_house(text: str, old: str, new: str) -> str | None:
    """Replace the first standalone occurrence of house number ``old`` (leading zeros allowed); None if absent."""
    out, n = re.subn(rf"(?<![0-9])0*{old}(?![0-9])", new, text, count=1)
    return out if n else None


def suffix_distribution(s1_names: list[str], min_share: float = 0.002) -> tuple[list[str], np.ndarray, set]:
    """Name endings of the country's S1 names (the generator's legal/suffix vocabulary: Inc, LLC, Group,
    Partners, Center, ...): alphabetic last tokens that end >= min_share of the names, each in its most
    common surface form, with their frequencies. Returns (surface forms, probabilities, folded tokens)."""
    from ..normalize import fold
    last = Counter()
    surface: dict[str, Counter] = {}
    for n in s1_names:
        toks = TOKEN.findall(n)
        if not toks:
            continue
        f = fold(toks[-1])
        if f.isalpha() and len(f) >= 2:
            last[f] += 1
            surface.setdefault(f, Counter())[toks[-1]] += 1
    n = max(len(s1_names), 1)
    keep = [(t, k) for t, k in last.most_common() if k / n >= min_share]
    if not keep:
        return [], np.zeros(0), set()
    forms = [surface[t].most_common(1)[0][0] for t, _ in keep]
    w = np.array([k for _, k in keep], float)
    return forms, w / w.sum(), {t for t, _ in keep}


def _strip_ending(name: str, endings: set) -> str:
    """The S1 name without its trailing legal-form tokens (at least one token kept)."""
    from ..normalize import fold
    toks = name.split()
    while len(toks) > 1:
        words = TOKEN.findall(toks[-1])
        if words and all(fold(w) in endings for w in words):
            toks = toks[:-1]
        else:
            break
    return " ".join(toks).rstrip(",")


ACCENT = str.maketrans("aeiouAEIOU", "\u00e1\u00e9\u00ed\u00f3\u00fa\u00c1\u00c9\u00cd\u00d3\u00da")


def word_distribution(s1_names: list[str], endings: set, top: int = 400) -> tuple[list[str], np.ndarray]:
    """Non-first, non-legal alphabetic words of the country's S1 names (len >= 3), most frequent ``top``."""
    from ..normalize import fold
    cnt, surface = Counter(), {}
    for n in s1_names:
        for t in n.split()[1:]:
            w = TOKEN.findall(t)
            if len(w) == 1 and w[0].isalpha() and len(w[0]) >= 3 and fold(w[0]) not in endings:
                cnt[fold(w[0])] += 1
                surface.setdefault(fold(w[0]), Counter())[w[0]] += 1
    keep = cnt.most_common(top)
    if not keep:
        return [], np.zeros(0)
    w = np.array([k for _, k in keep], float)
    return [surface[t].most_common(1)[0][0] for t, _ in keep], w / w.sum()


def synthetic_decoys(q_tbl: pa.Table, r_tbl: pa.Table, gt: pd.DataFrame, per_query: float, group_size: int,
                     seed: int, vocab: dict, country: str, translit_map: dict | None = None,
                     offset_max: int = 21, neg_share: float = 0.03, translit: bool = True, mode: str = "suffix",
                     id_prefix: str = "SYN") -> pa.Table:
    """Test-like dev pool: multi-record near-duplicate decoy entities (runs/decoy_shift.md).

    Test pools hold decoy entities next to many S1s: ~2 records named like the S1 with a different ending
    ("Group X", "X Corp", "X Partners"), at the S1's street with a slightly higher house number (offset > 0 in
    97%, |offset| quantiles 1/3/5/11/16), and no S1 of their own. Train has almost none, so "other candidates
    support this house" is a safe signal on dev and a trap on test (iterations 2-3: +0.37 links per US S1).
    For a random share of dev queries with >= 2 true records carrying a house number, this builds one decoy
    entity of ``group_size`` records (one more with probability 0.1): name = the S1's name without its legal
    form plus an ending drawn from the country's S1 endings (front with probability 0.15, upper case 0.25,
    one vowel replaced by U+FFFD 0.05, drawn per record); address = one of the S1's true records' addresses
    (their own noise) with the house number shifted by one offset for the whole entity (log-uniform
    1..offset_max, negative with probability neg_share). Rows go through the same normaliser as real records.
    ``per_query`` = extra decoy records per dev query. Ground truth is unchanged: decoys match nothing.
    mode "word": the decoy entity keeps the S1's first word and replaces another word (or adds one) with a word
    drawn from the country's S1 name words, once per entity ("EG Amis" -> "EG Institut", "Snack & Cie" ->
    "Snack & Cie Holding"); each record then gets an ending with probability 0.7. France's test pools hold
    0.30 such group records per S1 (US 0.033, India 0.080; runs/decoy_shift.md section 7).
    """
    from .. import normalize as nz
    rng = np.random.default_rng(seed)
    if per_query <= 0:
        return r_tbl.schema.empty_table()
    rpos = pd.Series(np.arange(r_tbl.num_rows), index=r_tbl["entity_id"].to_numpy(zero_copy_only=False))
    house = r_tbl["house"].to_numpy(zero_copy_only=False)
    q = q_tbl.select(["entity_id", "name_a", "house"]).to_pandas().set_index("entity_id")
    g = gt[gt["s1_id"].isin(q.index) & gt["cand_id"].isin(rpos.index)]
    g = g[house[rpos.reindex(g["cand_id"]).to_numpy()] != ""]
    groups = g.groupby("s1_id")["cand_id"].apply(list)
    groups = groups[groups.map(len) >= 2]
    forms, probs, fsuf = suffix_distribution(q["name_a"].tolist())
    if not len(groups) or not forms:
        return r_tbl.schema.empty_table()
    share = min(1.0, per_query / (group_size + 0.1) * q_tbl.num_rows / len(groups))
    groups = groups[rng.random(len(groups)) < share]
    need = np.unique(rpos.reindex([c for cs in groups for c in cs]).to_numpy())
    r = r_tbl.select(["entity_id", "addr_a", "house", "src"]).take(pa.array(need)).to_pandas().set_index("entity_id")
    endings = set(vocab.get("legal", {}).get(country, []))       # legal forms only: business words stay
    words, wprobs = word_distribution(q["name_a"].tolist(), endings | fsuf) if mode == "word" else ([], None)
    if mode == "word" and not words:
        return r_tbl.schema.empty_table()
    names, addrs, srcs = [], [], []
    for s1, cands in groups.items():
        base = _strip_ending(q.at[s1, "name_a"], endings)
        if mode == "word":
            toks = base.split()
            w = words[rng.choice(len(words), p=wprobs)]
            if len(toks) >= 2 and rng.random() < 0.6:
                i = int(rng.integers(1, len(toks)))
                toks[i] = w if w.lower() != toks[i].lower() else words[(words.index(w) + 1) % len(words)]
            else:
                toks.append(w)
            base = " ".join(toks)
        off = max(1, int(round(np.exp(rng.uniform(0, np.log(offset_max)))))) * (-1 if rng.random() < neg_share else 1)
        for cid in list(rng.permutation(cands))[:group_size + int(rng.random() < 0.1)]:
            old = r.at[cid, "house"]
            new = str(max(1, int(old) + off)) if old.isdigit() else None
            addr = _shift_house(r.at[cid, "addr_a"], old, new) if new and new != old else None
            if addr is None:
                continue
            suf = forms[rng.choice(len(forms), p=probs)]
            if mode == "word" and rng.random() >= 0.7:
                name = base
            else:
                name = f"{suf} {base}" if rng.random() < 0.15 else f"{base} {suf}"
            if rng.random() < 0.25:
                name = name.upper()
            if rng.random() < 0.05:                     # accent noise, as on real records ("Capital" -> "C\u00e1pital")
                vow = [i for i, ch in enumerate(name) if ch in "aeiouAEIOU"]
                if vow:
                    i = vow[rng.integers(len(vow))]
                    name = name[:i] + name[i].translate(ACCENT) + name[i + 1:]
            names.append(name)
            addrs.append(addr)
            srcs.append(r.at[cid, "src"])
    if not names:
        return r_tbl.schema.empty_table()
    nz._init_worker(vocab, translit)
    rows = nz._norm_chunk((names, addrs, [country] * len(names)))
    out = pd.DataFrame(rows, columns=list(nz.NAME_FIELDS) + list(nz.ADDR_FIELDS))
    if translit_map:
        lg = frozenset(vocab.get("legal", {}).get(country, []))
        for i in np.flatnonzero(out["f_nonlatin"].to_numpy() == 1):
            toks = [translit_map.get(t, t) for t in out.at[i, "name_b"].split()]
            core, leg = nz.split_legal(toks, lg)
            out.at[i, "name_b"], out.at[i, "core"] = " ".join(toks), " ".join(core)
            out.at[i, "legal"] = " ".join(sorted(set(leg)))
    out.insert(0, "entity_id", [f"S{int(sv)}-{id_prefix}{i}" for i, sv in enumerate(srcs)])
    out.insert(1, "country", country)
    out["name_a"] = [nz.view_a(x) for x in names]
    out["addr_a"] = [nz.view_a(x) for x in addrs]
    out["src"] = srcs
    out = out[r_tbl.schema.names]
    return pa.Table.from_pandas(out, schema=r_tbl.schema, preserve_index=False)


def record_key_codes(r: pd.DataFrame, names: list[str]) -> dict:
    """Exact record keys for C6: int code per record, -1 where a key part is empty."""
    core = r["core"].astype(str)
    parts = {"name": core, "house": r["house"].astype(str), "city": r["city"].astype(str),
             "addr": r["addr_core"].astype(str)}
    out = {}
    for name in names:
        cols = [parts[p] for p in name.split("+")]
        key = cols[0]
        for c in cols[1:]:
            key = key + "|" + c
        empty = np.logical_or.reduce([(c == "").to_numpy() for c in cols])
        codes, _ = pd.factorize(key)
        out[name] = np.where(empty, -1, codes).astype(np.int64)
    return out


def name_city_key(df: pd.DataFrame) -> pd.Series:
    ft = df["core"].str.split(n=1).str[0].fillna("")
    return (ft + "|" + df["city"]).where((ft != "") & (df["city"] != ""), "")


def _rank_within(qi: np.ndarray, score: np.ndarray) -> np.ndarray:
    order = np.lexsort((-score, qi))
    rank = np.empty(len(qi), np.int16)
    q_sorted = qi[order]
    starts = np.r_[0, np.flatnonzero(np.diff(q_sorted)) + 1]
    pos = np.arange(len(qi)) - np.repeat(starts, np.diff(np.r_[starts, len(qi)]))
    rank[order] = (pos + 1).clip(max=32000)
    return rank


def _topn_per_q(qi, ri, score, n):
    r = _rank_within(qi, score)
    keep = r <= n
    return qi[keep], ri[keep], score[keep]


def block_country(cfg, q: pd.DataFrame, r: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    bc, t = cfg["blocking"], Timer()
    info = {}
    space = TfidfSpace()
    rn, ra = space.fit_transform(r["core"].tolist(), r["addr_core"].tolist())
    qn, qa = space.transform(q["core"].tolist(), q["addr_core"].tolist())
    rc_t = combined(rn, ra).T.tocsr()
    qc = combined(qn, qa)
    # candidate generation on the rarer features only (short posting lists), then exact rescoring
    cap = max(bc.get("rare_df_min", 20000), int(bc.get("rare_df_share", 0.005) * len(r)))
    col_df = np.diff(rc_t.indptr)
    rc_t = sp.csr_matrix(sp.diags((col_df <= cap).astype(np.float32)) @ rc_t)
    rc_t.eliminate_zeros()
    info["t_vectorize"] = t()
    info["rare_df_cap"] = cap
    log.info("  vectorised %s queries, %s records, rare cap %d (%.0fs)", f"{len(q):,}", f"{len(r):,}", cap, t())
    gq, gr, _ = topk(qc, rc_t, bc.get("c1_generate", 100), threshold=0.0, chunk=bc["query_chunk"])
    del rc_t, qc
    gs = 0.5 * (pair_cos(qn, rn, gq, gr) + pair_cos(qa, ra, gq, gr))
    c1q, c1r, c1s = _topn_per_q(gq, gr, gs, bc["k_tfidf"])
    del gq, gr, gs
    info["t_c1"] = t()
    log.info("  C1: %s pairs (%.0fs)", f"{len(c1q):,}", t())
    c2q, c2r, c2size = crowd_channel(sorted_key(q["core"]), sorted_key(r["core"]), qa, ra, bc["crowd_cap"])
    info["t_c2"] = t()
    log.info("  C2: %s pairs (%.0fs)", f"{len(c2q):,}", t())
    keys = {}
    sdf = token_df(r["street"]) if bc.get("key_rare_token", False) else None
    hk = lambda d: house_key(d, sdf)
    for name, fn in (("c5a", hk), ("c5b", name_city_key)):
        kq, kr, skipped = key_channel(fn(q), fn(r), bc["key_block_cap"])
        s = 0.5 * (pair_cos(qn, rn, kq, kr) + pair_cos(qa, ra, kq, kr))
        keys[name] = _topn_per_q(kq, kr, s, bc.get("key_keep", 10))
        info[f"{name}_skipped_q"] = skipped
    info["t_c5"] = t()

    n_r = np.int64(len(r))
    chans = {"c1": (c1q, c1r), "c2": (c2q, c2r), **{k: v[:2] for k, v in keys.items()}}
    rv = bc.get("reverse_empty") or {}
    if rv.get("enabled", False):
        # C7: records with an empty address retrieve their top-k S1s by name alone. In C1 their
        # combined score is capped at 0.5 and in large name crowds C2 ranks them last, so the
        # error analysis found 59% of the blocking loss in entities with such a missed record.
        emp = np.flatnonzero(r["addr_core"].to_numpy() == "")
        if len(emp):
            c7q, c7r = reverse_name_channel(emp, rn, qn, rv.get("k", 10), rv.get("min_cos", 0.3), bc["query_chunk"])
            if rv.get("forward_k", 0):
                # A record's top-k S1s depend on how many S1s compete: the dev universe holds ~30% of the
                # queries test has, so reverse PC on dev is optimistic (runs/c7_scale.md). The forward
                # direction searches the same record pool in dev and test.
                fq, fr, _ = topk(qn, rn[emp].T.tocsr(), rv["forward_k"], threshold=rv.get("min_cos", 0.3),
                                 chunk=bc["query_chunk"])
                c7q, c7r = np.concatenate([c7q, fq.astype(np.int32)]), np.concatenate([c7r, emp[fr].astype(np.int32)])
            chans["c7"] = (c7q, c7r)
        info["c7_records"] = int(len(emp))
        info["t_c7"] = t()
        log.info("  C7: %s pairs from %s empty-address records (%.0fs)", f"{len(chans.get('c7', ([],))[0]):,}",
                 f"{len(emp):,}", t())
    ad = bc.get("address_only") or {}
    if ad.get("enabled", False):
        # C8 (error budget T4: "name replaced" = 26-27% of blocking misses in both countries)
        chans["c8"] = address_only_channel(qa, ra, ad.get("k", 8), ad.get("min_cos", 0.6), ad.get("generate", 50),
                                           cap, bc["query_chunk"])
        info["t_c8"] = t()
        log.info("  C8: %s pairs (%.0fs)", f"{len(chans['c8'][0]):,}", t())
    ra_cfg = bc.get("reverse_all") or {}
    if ra_cfg.get("enabled", False):
        # only records no other channel retrieved, located in a city some S1 of this country sits in (on dev
        # this leaves out the other cities' distractor records, which have no S1 to find)
        covered = np.zeros(len(r), bool)
        for _, b in chans.values():
            covered[b] = True
        in_city = r["city"].isin(set(q["city"]) - {""}).to_numpy()
        orph = np.flatnonzero(~covered & in_city & (r["addr_core"].to_numpy() != ""))
        chans["c9"] = reverse_all_channel(orph, qn, qa, rn, ra, ra_cfg.get("k", 3), ra_cfg.get("generate", 30), cap,
                                          bc["query_chunk"], ra_cfg.get("min_score", 0.3))
        info["c9_records"] = int(len(orph))
        info["t_c9"] = t()
        log.info("  C9: %s pairs from %s uncovered records (%.0fs)", f"{len(chans['c9'][0]):,}", f"{len(orph):,}", t())
    sh = bc.get("second_hop") or {}
    if sh.get("enabled", False):
        # C6: expand from confident first-pass pairs through exact record keys (query expansion)
        first = np.unique(np.concatenate([a.astype(np.int64) * n_r + b for a, b in chans.values()]))
        fq, fr = (first // n_r).astype(np.int32), (first % n_r).astype(np.int32)
        fs = 0.5 * (pair_cos(qn, rn, fq, fr) + pair_cos(qa, ra, fq, fr))
        anc = fs >= sh.get("anchor_c1", 0.8)
        c6q, c6r = second_hop_channel(fq[anc], fr[anc], record_key_codes(r, sh.get("keys", ["name+house"])),
                                      sh.get("cap", 20))
        chans["c6"] = (c6q, c6r)
        info["c6_anchors"] = int(anc.sum())
        info["t_c6"] = t()
        log.info("  C6: %s pairs from %s anchors (%.0fs)", f"{len(c6q):,}", f"{anc.sum():,}", t())
    all_keys = np.unique(np.concatenate([a.astype(np.int64) * n_r + b for a, b in chans.values()]))
    pairs = pd.DataFrame({"qi": (all_keys // n_r).astype(np.int32), "ri": (all_keys % n_r).astype(np.int32)})
    for name, (a, b) in chans.items():
        pos = np.searchsorted(all_keys, a.astype(np.int64) * n_r + b)
        pairs[f"in_{name}"] = np.zeros(len(pairs), np.int8)
        pairs.loc[pos, f"in_{name}"] = 1
        if name == "c1":
            rank = _rank_within(a, c1s)
            pairs["c1_rank"] = np.int16(bc["k_tfidf"] + 1)
            pairs.loc[pos, "c1_rank"] = rank
        if name == "c2":
            pairs["crowd_size"] = np.int32(0)
            pairs.loc[pos, "crowd_size"] = c2size
    qi, ri = pairs["qi"].to_numpy(), pairs["ri"].to_numpy()
    pairs["cos_name"] = pair_cos(qn, rn, qi, ri)
    pairs["cos_addr"] = pair_cos(qa, ra, qi, ri)
    pairs["c1_score"] = 0.5 * (pairs["cos_name"] + pairs["cos_addr"])
    pairs["union_rank"] = _rank_within(qi, pairs["c1_score"].to_numpy())
    pairs["src"] = r["src"].to_numpy()[ri]
    info["t_union"] = t()
    info["pairs"] = len(pairs)
    return pairs, info


def completeness(pairs: pd.DataFrame, q: pd.DataFrame, r: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Pair completeness per channel and for the union, by source and match-count bucket."""
    qpos = pd.Series(np.arange(len(q)), index=q["entity_id"])
    rpos = pd.Series(np.arange(len(r)), index=r["entity_id"])
    g = gt[gt["s1_id"].isin(qpos.index)]
    tq = qpos.reindex(g["s1_id"]).to_numpy()
    tr = rpos.reindex(g["cand_id"]).to_numpy()
    assert not np.isnan(tr.astype(float)).any(), "true records missing from the index"
    n_r = np.int64(len(r))
    true_keys = tq.astype(np.int64) * n_r + tr.astype(np.int64)
    pk = pairs["qi"].to_numpy(np.int64) * n_r + pairs["ri"].to_numpy(np.int64)
    out = {"n_true": len(true_keys)}
    found = np.isin(true_keys, pk)
    out["PC_union"] = round(float(found.mean()), 5)
    for ch in [c for c in pairs.columns if c.startswith("in_")]:
        out[f"PC_{ch[3:]}"] = round(float(np.isin(true_keys, pk[pairs[ch].to_numpy() == 1]).mean()), 5)
    src = r["src"].to_numpy()[tr]
    for s in (2, 3):
        out[f"PC_union_S{s}"] = round(float(found[src == s].mean()), 5)
    n_true = g.groupby("s1_id").size()
    b = n_true.reindex(g["s1_id"]).clip(upper=3).to_numpy()
    for v in (1, 2, 3):
        out[f"PC_union_bucket{v}{'+' if v == 3 else ''}"] = round(float(found[b == v].mean()), 5)
    out["cands_per_s1"] = round(len(pairs) / len(q), 2)
    out["RR"] = round(1 - len(pairs) / (len(q) * len(r)), 7)
    # upper bound: share of queries whose true set is fully retrieved
    return out


def run(cfg: dict, split: str = "dev", frac: float | None = None) -> dict:
    """frac: benchmark mode; only that share of the queries is blocked and nothing is saved."""
    t = Timer()
    d = block_dir(cfg)
    src_split = "train" if split == "dev" else "test"
    s1_tbl = pq.read_table(norm_path(cfg, src_split, 1))
    n_full = {}
    if split == "dev":
        dev = build_dev_universe(cfg)
        ids = pa.array(dev.loc[dev["role"] == "query", "entity_id"].tolist())
        s1_tbl = s1_tbl.filter(pc.is_in(s1_tbl["entity_id"], value_set=ids))
        gt = io_utils.load_ground_truth(cfg["paths"]["dataset"])
    if frac:
        n_full = pd.Series(s1_tbl["country"].to_pylist()).value_counts().to_dict()
        rng = np.random.default_rng(cfg["seed"])
        s1_tbl = s1_tbl.filter(pa.array(rng.random(s1_tbl.num_rows) < frac))
    res = {"stage": "block_bench" if frac else "block", "notes": ""}
    stats, n_syn = {}, {}
    for country in sorted(set(s1_tbl["country"].to_pylist())):
        q_tbl = s1_tbl.filter(pc.equal(s1_tbl["country"], country))
        q = q_tbl.select(BLOCK_COLS).to_pandas()
        tbl_r = pa.concat_tables([pq.read_table(norm_path(cfg, src_split, k), filters=[("country", "=", country)])
                                  for k in (2, 3)])
        dc = (cfg.get("dev") or {}).get("decoys") or {}
        if split == "dev" and dc.get("enabled", False):
            art = cfg["paths"]["artifacts"]
            vocab = json.load(open(os.path.join(art, "vocab.json"), encoding="utf-8"))
            tpath = os.path.join(art, "translit_dict.json")
            tmap = json.load(open(tpath, encoding="utf-8")) \
                if cfg["normalize"].get("norm_dir") == "norm_tl" and os.path.exists(tpath) else None
            syn = synthetic_decoys(q_tbl, tbl_r, gt, dc.get("per_query", {}).get(country, 0.0), dc.get("group_size", 2),
                                   cfg["seed"] + sum(map(ord, country)), vocab, country, tmap, dc.get("offset_max", 21),
                                   dc.get("neg_share", 0.03), cfg["flags"].get("TRANSLITERATE", True))
            if dc.get("per_query_word", {}).get(country, 0.0) > 0:
                syw = synthetic_decoys(q_tbl, tbl_r, gt, dc["per_query_word"][country], dc.get("group_size", 2),
                                       cfg["seed"] + 7 + sum(map(ord, country)), vocab, country, tmap,
                                       dc.get("offset_max", 21), dc.get("neg_share", 0.03),
                                       cfg["flags"].get("TRANSLITERATE", True), mode="word", id_prefix="SYW")
                syn = pa.concat_tables([syn, syw])
            tbl_r = pa.concat_tables([tbl_r, syn])
            n_syn[country] = syn.num_rows
            log.info("block dev %s: %s synthetic decoy records (%.3f per query)", country, f"{syn.num_rows:,}",
                     syn.num_rows / max(q_tbl.num_rows, 1))
        r = tbl_r.select(BLOCK_COLS).to_pandas()
        log.info("block %s %s: loaded %s queries, %s records (%.0fs)", split, country, f"{len(q):,}", f"{len(r):,}", t())
        pairs, info = block_country(cfg, q, r)
        if frac:
            per_q = (info["t_c1"] - info["t_vectorize"] + info["t_union"] - info["t_c5"]) / max(len(q), 1)
            info["projected_full_s"] = round(info["t_union"] + per_q * (n_full[country] - len(q)), 0)
        else:
            pq.write_table(q_tbl, os.path.join(d, f"{split}_{country}_q.parquet"))
            pq.write_table(tbl_r, os.path.join(d, f"{split}_{country}_r.parquet"))
            pairs.to_parquet(os.path.join(d, f"{split}_{country}_pairs.parquet"), index=False)
        del tbl_r
        if split == "dev":
            info.update(completeness(pairs, q, r, gt))
            res[f"PC_union_{country}"] = info["PC_union"]
            res[f"RR_{country}"] = info["RR"]
        info.update({"n_q": len(q), "n_r": len(r), "n_synthetic_decoys": n_syn.get(country, 0)})
        stats[country] = info
        log.info("block %s %s: %s", split, country, json.dumps(info))
        del q, r, pairs, q_tbl
    res["cands_per_s1"] = round(sum(s["pairs"] for s in stats.values()) / sum(s["n_q"] for s in stats.values()), 2)
    res["runtime_s"] = t()
    res["notes"] = json.dumps(stats)
    name = f"block_{split}{'_bench' if frac else ''}.json"
    json.dump(stats, open(os.path.join(cfg["paths"]["runs"], name), "w"), indent=1)
    return res
