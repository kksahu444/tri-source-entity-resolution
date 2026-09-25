"""C2 (name crowd) and C5 (hash keys) channels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tfidf import pair_cos


def _codes(q_keys: pd.Series, r_keys: pd.Series):
    codes, uniq = pd.factorize(pd.concat([q_keys, r_keys], ignore_index=True))
    qc, rc = codes[:len(q_keys)], codes[len(q_keys):]
    empty = np.flatnonzero(uniq == "")
    if len(empty):
        qc = np.where(qc == empty[0], -1, qc)
        rc = np.where(rc == empty[0], -1, rc)
    return qc, rc


def _join(qc: np.ndarray, rc: np.ndarray, max_r: int | None):
    """All (qi, ri) with equal non-empty codes; blocks with more than max_r records are split off."""
    r_df = pd.DataFrame({"k": rc, "ri": np.arange(len(rc), dtype=np.int32)})
    r_df = r_df[r_df["k"] >= 0]
    size = r_df.groupby("k").size()
    q_df = pd.DataFrame({"k": qc, "qi": np.arange(len(qc), dtype=np.int32)})
    q_df = q_df[(q_df["k"] >= 0) & q_df["k"].isin(size.index)]
    q_df["rs"] = q_df["k"].map(size).to_numpy()
    return q_df, r_df, size


def crowd_channel(q_keys, r_keys, qa, ra, cap: int, max_pairs: int = 20_000_000):
    """Exact core-name key. Crowds of <= cap records: all members; bigger crowds: the cap members
    with the highest address cosine. Returns (qi, ri, address cosine, crowd size)."""
    qc, rc = _codes(q_keys, r_keys)
    q_df, r_df, size = _join(qc, rc, None)
    out = []
    small_q = q_df[q_df["rs"] <= cap]
    m = small_q.merge(r_df, on="k")
    out.append((m["qi"].to_numpy(np.int32), m["ri"].to_numpy(np.int32), m["rs"].to_numpy(np.int32)))
    big_q = q_df[q_df["rs"] > cap]
    if len(big_q):
        big_q = big_q.sort_values("k")
        cum = np.cumsum(big_q["rs"].to_numpy(np.int64))
        starts = np.searchsorted(cum, np.arange(0, cum[-1], max_pairs), side="right")
        bounds = list(starts) + [len(big_q)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if a >= b:
                continue
            m = big_q.iloc[a:b].merge(r_df, on="k")
            qi, ri = m["qi"].to_numpy(np.int32), m["ri"].to_numpy(np.int32)
            s = pair_cos(qa, ra, qi, ri)
            m = pd.DataFrame({"qi": qi, "ri": ri, "s": s, "rs": m["rs"].to_numpy(np.int32)})
            m = m.sort_values(["qi", "s"], ascending=[True, False]).groupby("qi", sort=False).head(cap)
            out.append((m["qi"].to_numpy(np.int32), m["ri"].to_numpy(np.int32), m["rs"].to_numpy(np.int32)))
    qi = np.concatenate([o[0] for o in out])
    ri = np.concatenate([o[1] for o in out])
    rs = np.concatenate([o[2] for o in out])
    return qi, ri, rs


def key_channel(q_keys, r_keys, cap: int):
    """Hash join; blocks holding more than ``cap`` records are skipped (counted)."""
    qc, rc = _codes(q_keys, r_keys)
    q_df, r_df, size = _join(qc, rc, cap)
    skipped = int((q_df["rs"] > cap).sum())
    m = q_df[q_df["rs"] <= cap].merge(r_df, on="k")
    return m["qi"].to_numpy(np.int32), m["ri"].to_numpy(np.int32), skipped
