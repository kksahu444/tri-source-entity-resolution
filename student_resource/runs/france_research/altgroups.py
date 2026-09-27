"""Alternative-name groups: 2+ candidates of one S1 that share the same name (sorted core tokens) and house,
where that name differs from the S1's (word changed/added, first-word stem kept) - the signature of a
separate entity sitting beside the S1. Measured by house relation; dev gives the true rate."""
import os
import numpy as np, pandas as pd

A = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/artifacts"
res = []
for split, c in [("dev", "US"), ("dev", "India"), ("test", "US"), ("test", "India"), ("test", "France")]:
    d = pd.read_parquet(f"pat_{split}_{c}.parquet")
    q = pd.read_parquet(os.path.join(A, "block", f"{split}_{c}_q.parquet"), columns=["core", "house"])
    r = pd.read_parquet(os.path.join(A, "block", f"{split}_{c}_r.parquet"), columns=["core", "house", "street"])
    n_s1 = len(q)
    d = d[d.nrel.isin(["substituted", "added"])].copy()
    rs = r.core.str.split().map(lambda t: " ".join(sorted(t))).to_numpy()
    d["rkey"] = rs[d.ri]
    d["rh"] = r.house.to_numpy()[d.ri]
    d["gsize"] = d.groupby(["qi", "rkey", "rh"]).qi.transform("size")
    d["grp"] = np.where(d.gsize >= 2, "group", "single")
    g = d.groupby(["hrel", "grp"])
    t = pd.DataFrame({"per_s1": g.size() / n_s1, "sel_per_s1": g.sel.sum() / n_s1})
    if "y" in d:
        t["true_rate"] = g.y.mean()
    t["key"] = f"{split}_{c}"
    res.append(t.reset_index())
T = pd.concat(res)
pd.set_option("display.width", 250)
print(T[T.hrel.isin(["same", "near"])].pivot_table(index=["hrel", "grp"], columns="key",
      values=["per_s1", "sel_per_s1", "true_rate"]).round(4).to_string())
