"""Stratified transport of dev error rates to test links.

Strata: name relation x house relation x S1 co-location (how many S1s share the S1's exact address).
Dev: precision of selected links per stratum (iteration-2 stage-A OOF, GFM). Test: upload-1 links per S1 per
stratum. Expected false links per S1 on test = sum(links * (1 - dev precision)).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from patterns import name_rel, house_rel

A = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/artifacts"
S = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/submissions"


def coloc(q):
    k = q.house + "|" + q.house_sfx + "|" + q.street + "|" + q.city
    n = k.map(k.value_counts())
    n = np.where(q.house.to_numpy() == "", 0, n)
    return np.minimum(n, 3)             # 0 = S1 has no house, 1 = alone, 2, 3 = 3+


def rev_coloc(q, r):
    """For each record: how many S1s sit at the record's exact address (street + house + city)."""
    kq = q.house + "|" + q.house_sfx + "|" + q.street + "|" + q.city
    kr = r.house + "|" + r.house_sfx + "|" + r.street + "|" + r.city
    return np.minimum(kr.map(kq.value_counts()).fillna(0).to_numpy(), 3)


cols = ["entity_id", "core", "house", "house_sfx", "street", "city"]
prec = []
for c in ["US", "India"]:
    q = pd.read_parquet(os.path.join(A, "block", f"dev_{c}_q.parquet"), columns=cols)
    d = pd.read_parquet(f"pat_dev_{c}.parquet")
    d["col"] = coloc(q)[d.qi]
    s = d[d.sel]
    prec.append(s.groupby(["nrel", "hrel", "col"]).agg(n=("y", "size"), prec=("y", "mean")).assign(c=c))
P = pd.concat(prec).reset_index()
Pp = P.groupby(["nrel", "hrel", "col"]).apply(lambda x: pd.Series({"n": x.n.sum(), "prec": np.average(x.prec, weights=x.n)}),
                                                include_groups=False)
print(Pp[Pp.n > 200].round(4).to_string())

out = {}
for c in ["France", "US", "India"]:
    q = pd.read_parquet(os.path.join(A, "block", f"test_{c}_q.parquet"), columns=cols)
    r = pd.read_parquet(os.path.join(A, "block", f"test_{c}_r.parquet"), columns=["entity_id", "core", "house"])
    qpos = pd.Series(np.arange(len(q)), index=q.entity_id)
    rpos = pd.Series(np.arange(len(r)), index=r.entity_id)
    colq = coloc(q)
    print(c, "S1 co-location share", {int(k): round(v, 4) for k, v in pd.Series(colq).value_counts(normalize=True).sort_index().items()})
    m = pd.read_csv(os.path.join(S, "upload_1", "matching_results.tsv"), sep="\t", dtype=str, keep_default_na=False)
    m = m[m.source1_entity_id.isin(qpos.index) & (m.matched_entity_ids != "")]
    m = m.assign(rid=m.matched_entity_ids.str.split(",")).explode("rid")
    qi, ri = qpos.reindex(m.source1_entity_id).to_numpy(), rpos.reindex(m.rid).to_numpy()
    L = pd.DataFrame({"nrel": [name_rel(a, b) for a, b in zip(q.core.to_numpy()[qi], r.core.to_numpy()[ri])],
                      "hrel": [house_rel(a, b) for a, b in zip(q.house.to_numpy()[qi], r.house.to_numpy()[ri])],
                      "col": colq[qi]})
    t = L.groupby(["nrel", "hrel", "col"]).size().rename("links").to_frame()
    t = t.join(Pp, how="left")
    t["prec"] = t["prec"].fillna(P.prec.mean())
    t["exp_fp"] = t.links * (1 - t.prec)
    out[c] = t
    print(c, "links/S1 %.4f  expected FP/S1 %.4f" % (t.links.sum() / len(q), t.exp_fp.sum() / len(q)))
    top = (t.exp_fp / len(q)).sort_values(ascending=False).head(8)
    print("  top strata by expected FP/S1:", top.round(4).to_dict())
