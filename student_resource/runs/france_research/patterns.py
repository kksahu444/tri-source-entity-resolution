"""Pair pattern classes (name relation x house relation): prevalence per S1, link rate, dev truth rate.

A pattern much more common on test than on dev, and linked at the dev rate, is where test-only decoys
turn into false links. Only pairs with calibrated p > 0.05 are classified (the ones that can be linked).
"""
import os, pickle, sys
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/code/business_entity_resolution")
from src.assign import hard_assign
from src.decision import gfm_rule

A = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/artifacts"
iso = pickle.load(open(os.path.join(A, "decision.pkl"), "rb"))["iso"]


def tok_match(a, b):
    return a == b or (len(a) > 3 and len(b) > 3 and Levenshtein.normalized_similarity(a, b) >= 0.75)


def name_rel(qc, rc):
    Q, R = qc.split(), rc.split()
    if not R:
        return "r_empty"
    if sorted(Q) == sorted(R):
        return "equal"
    q_hit = [any(tok_match(a, b) for b in R) for a in Q]
    r_hit = [any(tok_match(b, a) for a in Q) for b in R]
    nq, nr = sum(not x for x in q_hit), sum(not x for x in r_hit)
    if nq == 0 and nr == 0:
        return "fuzzy_equal"
    if sum(q_hit) == 0:
        return "disjoint"
    if nq == 0:
        return "added"            # record = S1 name + extra word(s)
    if nr == 0:
        return "dropped"
    return "substituted"          # S1 word(s) replaced by other word(s)


def house_rel(hq, hr):
    if not hr:
        return "r_nohouse"
    if not hq:
        return "q_nohouse"
    if hq == hr:
        return "same"
    try:
        d = abs(int(hr) - int(hq))
    except ValueError:
        return "other"
    return "near" if d <= 30 else "far"


def main():
    rows = []
    for split, c in [("dev", "US"), ("dev", "India"), ("test", "US"), ("test", "India"), ("test", "France")]:
        d = pd.read_parquet(os.path.join(A, "pred", f"{split}_{c}_stageA.parquet"))
        q = pd.read_parquet(os.path.join(A, "block", f"{split}_{c}_q.parquet"), columns=["core", "house"])
        r = pd.read_parquet(os.path.join(A, "block", f"{split}_{c}_r.parquet"), columns=["core", "house"])
        n_s1 = len(q)
        d["pc"] = iso.predict(d["p"].to_numpy())
        keep = hard_assign(d["ri"].to_numpy(), d["pc"].to_numpy())
        sel, _ = gfm_rule.select(d["qi"].to_numpy()[keep], d["pc"].to_numpy()[keep], None)
        d["sel"] = False
        d.loc[np.flatnonzero(keep)[sel], "sel"] = True
        d = d[d.pc > 0.05].reset_index(drop=True)
        qc, qh = q["core"].to_numpy()[d.qi], q["house"].to_numpy()[d.qi]
        rc, rh = r["core"].to_numpy()[d.ri], r["house"].to_numpy()[d.ri]
        d["nrel"] = [name_rel(a, b) for a, b in zip(qc, rc)]
        d["hrel"] = [house_rel(a, b) for a, b in zip(qh, rh)]
        g = d.groupby(["nrel", "hrel"])
        t = pd.DataFrame({"per_s1": g.size() / n_s1, "sel_per_s1": g.sel.sum() / n_s1, "mean_p": g.pc.mean()})
        if "y" in d:
            t["true_rate"] = g.y.mean()
            t["fp_sel_per_s1"] = g.apply(lambda x: ((x.y == 0) & x.sel).sum()) / n_s1
        t["key"] = f"{split}_{c}"
        rows.append(t.reset_index())
        print(split, c, "done", flush=True)
        d[["qi", "ri", "pc", "sel", "nrel", "hrel"] + (["y"] if "y" in d else [])].to_parquet(f"pat_{split}_{c}.parquet")

    T = pd.concat(rows)
    T.to_csv("patterns.csv", index=False)
    W = T.pivot_table(index=["nrel", "hrel"], columns="key", values=["per_s1", "sel_per_s1"])
    pd.set_option("display.width", 250)
    print(W.round(4).to_string())
    print(T[T.key.str.startswith("dev")].set_index(["key", "nrel", "hrel"])[["true_rate", "fp_sel_per_s1"]].round(4).to_string())


if __name__ == '__main__':
    main()
