"""D0 audit of the challenge data (spec section 4).

RECONSTRUCTION NOTICE: the tested Appendix B file did not reach the implementation agent (the
spec message was truncated). This script re-implements every audit block named in the section-4
trigger table with the heuristics the spec states (house number = first digit run, city =
second-to-last address part). If the original Appendix B turns up, replace this file with it.

Run from ``student_resource/``:  python code/business_entity_resolution/src/audit_d0.py --root dataset
"""
import argparse
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

FIRST_NUM = re.compile(r"\d+")
LATIN_OK = re.compile(r"^[\x00-\x7fÀ-ɏ‘-‟ -¿]*$")


def read(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    with open(path, "rb") as f:
        n_lines = sum(1 for _ in f)
    assert len(df) == n_lines - 1, f"{path}: {len(df)} rows vs {n_lines - 1} data lines"
    return df


def pct(x):
    return f"{100 * x:.2f}%"


def first_num(s):
    m = FIRST_NUM.search(s)
    return m.group(0) if m else ""


def last_token(name):
    toks = re.sub(r"[^\w\s]", " ", name.lower()).split()
    return toks[-1] if toks else ""


def city_of(addr):
    parts = [p.strip().lower() for p in addr.split(",") if p.strip()]
    return parts[-2] if len(parts) >= 2 else (parts[0] if parts else "")


def core_name(name, legal_vocab):
    toks = re.sub(r"[^\w\s]", " ", name.lower()).split()
    while toks and toks[-1] in legal_vocab:
        toks.pop()
    return " ".join(toks)


def mined_legal_vocab(names, min_share=0.001):
    c = Counter(last_token(n) for n in names)
    n = max(len(names), 1)
    return {t for t, k in c.items() if t and k / n >= min_share and not t.isdigit()}


def section(title):
    print(f"\n=== {title} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--sample", type=int, default=0, help="rows per file (0 = all)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    R = args.root

    tr = {k: read(f"{R}/train/train_source{k}.tsv") for k in (1, 2, 3)}
    te = {k: read(f"{R}/test/test_source{k}.tsv") for k in (1, 2, 3)}
    gt = read(f"{R}/train/train_ground_truth.tsv")
    print("rows:", {f"train_s{k}": len(v) for k, v in tr.items()}, {f"test_s{k}": len(v) for k, v in te.items()},
          "gt", len(gt))

    section("Block 1: country labels")
    for split, d in (("train", tr), ("test", te)):
        for k, df in d.items():
            print(f"  {split} s{k}:", dict(df["country"].value_counts()))

    section("Block 2: pool ratio (S2+S3)/S1 per country")
    for split, d in (("train", tr), ("test", te)):
        c1 = d[1]["country"].value_counts()
        c23 = d[2]["country"].value_counts().add(d[3]["country"].value_counts(), fill_value=0)
        for c in sorted(c1.index):
            print(f"  {split} {c}: S1={c1[c]:,} S2+S3={int(c23.get(c, 0)):,} ratio={c23.get(c, 0) / c1[c]:.3f}"
                  f"  (S2 {d[2]['country'].eq(c).sum() / c1[c]:.3f}, S3 {d[3]['country'].eq(c).sum() / c1[c]:.3f})")

    section("Block 3: integrity")
    s1_ids = set(tr[1]["entity_id"])
    gt_ids = set(gt["source1_entity_id"])
    other_ids = set(tr[2]["entity_id"]) | set(tr[3]["entity_id"])
    pairs = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    pairs = pairs[pairs["m"].fillna("") != ""][["source1_entity_id", "m"]]
    print(f"  S1 missing from GT: {len(s1_ids - gt_ids)}; GT ids not in S1: {len(gt_ids - s1_ids)};"
          f" duplicate GT rows: {gt['source1_entity_id'].duplicated().sum()}")
    print(f"  matched ids missing from S2/S3: {(~pairs['m'].isin(other_ids)).sum()} of {len(pairs):,} pairs")
    print(f"  matched ids with S1 prefix: {pairs['m'].str.startswith('S1-').sum()}")

    section("Block 4: singletons and match counts by country")
    cnt = gt.set_index("source1_entity_id")["matched_entity_ids"].map(lambda s: len(s.split(",")) if s else 0)
    ctry = tr[1].set_index("entity_id")["country"]
    cdf = pd.DataFrame({"n": cnt, "country": ctry.reindex(cnt.index)})
    for c, g in cdf.groupby("country"):
        vc = g["n"].clip(upper=8).value_counts(normalize=True).sort_index()
        print(f"  {c}: singleton share {pct((g['n'] == 0).mean())}; mean matches {g['n'].mean():.3f};"
              f" dist(0..8+) {[round(v, 4) for v in vc.values]}")

    section("Block 5: matches per source per S1")
    pairs["src"] = pairs["m"].str[:2]
    per = pairs.groupby(["source1_entity_id", "src"]).size().unstack(fill_value=0)
    for s in ("S2", "S3"):
        vc = per[s].clip(upper=6).value_counts(normalize=True).sort_index()
        print(f"  {s} per S1 (among S1s with any match): {dict((int(k), round(v, 4)) for k, v in vc.items())};"
              f" share with >=2: {pct((per[s] >= 2).mean())}")

    section("Block 6: cross-country true pairs")
    oc = pd.concat([tr[2][["entity_id", "country"]], tr[3][["entity_id", "country"]]]).set_index("entity_id")["country"]
    pairs["c1"] = pairs["source1_entity_id"].map(ctry)
    pairs["c2"] = pairs["m"].map(oc)
    x = pairs["c1"] != pairs["c2"]
    print(f"  cross-country pairs: {x.sum()} of {len(pairs):,} ({pct(x.mean())})")
    if x.any():
        print(pairs[x].groupby(["c1", "c2"]).size())

    section("Block 7: many-to-one (S2/S3 record matched to 2+ S1)")
    mm = pairs["m"].value_counts()
    print(f"  records matched to 2+ S1: {(mm >= 2).sum()} of {len(mm):,} ({pct((mm >= 2).mean())})")
    n_unmatched = len(other_ids) - len(mm)
    print(f"  S2/S3 records never matched (distractors): {n_unmatched:,} of {len(other_ids):,} ({pct(n_unmatched / len(other_ids))})")

    section("Block 8: ID leakage (numeric id gap of true pairs vs shuffled)")
    num = lambda s: s.str[3:].astype(np.int64)
    gap_true = np.abs(num(pairs["source1_entity_id"]).values - num(pairs["m"]).values)
    rng = np.random.default_rng(0)
    shuf = pairs["m"].values.copy()
    for c in pairs["c1"].dropna().unique():
        idx = np.flatnonzero(pairs["c1"].values == c)
        shuf[idx] = shuf[rng.permutation(idx)]
    gap_shuf = np.abs(num(pairs["source1_entity_id"]).values - num(pd.Series(shuf)).values)
    print(f"  median gap true {np.median(gap_true):,.0f} vs shuffled within country {np.median(gap_shuf):,.0f}")

    section("Block 9: field completeness and scripts")
    for split, d in (("train", tr), ("test", te)):
        for k, df in d.items():
            for c, g in df.groupby("country"):
                nonlatin = ~g["business_name"].str.match(LATIN_OK)
                print(f"  {split} s{k} {c}: empty name {pct(g['business_name'].str.strip().eq('').mean())},"
                      f" empty addr {pct(g['business_address'].str.strip().eq('').mean())},"
                      f" non-Latin name {pct(nonlatin.mean())},"
                      f" non-Latin addr {pct((~g['business_address'].str.match(LATIN_OK)).mean())}")

    section("Block 10: name crowds (S1)")
    for split, d in (("train", tr), ("test", te)):
        df = d[1]
        for c, g in df.groupby("country"):
            names = g["business_name"].str.lower().str.strip()
            dup = names.duplicated(keep=False).mean()
            vocab = mined_legal_vocab(g["business_name"].tolist())
            core = [core_name(n, vocab) for n in g["business_name"]]
            city = [city_of(a) for a in g["business_address"]]
            key = pd.Series([f"{a}|{b}" for a, b in zip(core, city)])
            print(f"  {split} {c}: exact-name crowd share {pct(dup)}; same core name+city {pct(key.duplicated(keep=False).mean())};"
                  f" largest name crowd {names.value_counts().iloc[0]}")

    section("Block 11: true-pair number conflicts and legal-token flips")
    s1 = tr[1].set_index("entity_id")
    oth = pd.concat([tr[2], tr[3]]).set_index("entity_id")
    samp = pairs.sample(min(len(pairs), 500_000), random_state=0)
    a1 = s1.loc[samp["source1_entity_id"], "business_address"].values
    a2 = oth.loc[samp["m"], "business_address"].values
    n1 = s1.loc[samp["source1_entity_id"], "business_name"].values
    n2 = oth.loc[samp["m"], "business_name"].values
    h1 = np.array([first_num(a) for a in a1])
    h2 = np.array([first_num(a) for a in a2])
    both = (h1 != "") & (h2 != "")
    print(f"  pairs sampled: {len(samp):,}; both have a number: {pct(both.mean())}")
    print(f"  num_conflict on true pairs (first digit run differs | both present) r = {pct((h1[both] != h2[both]).mean())}")
    lt1 = np.array([last_token(n) for n in n1])
    lt2 = np.array([last_token(n) for n in n2])
    print(f"  legal_token_equal (last name token equal) on true pairs: {pct((lt1 == lt2).mean())}")
    print(f"  S2/S3 address empty on true pairs: {pct(np.mean([a.strip() == '' for a in a2]))}")
    print(f"  city (2nd-to-last part) equal on true pairs: {pct(np.mean([city_of(x) == city_of(y) for x, y in zip(a1, a2)]))}")


if __name__ == "__main__":
    main()
