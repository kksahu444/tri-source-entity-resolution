"""Links per S1 by pattern class for uploads 1 and 2 (test) vs dev truth, per country."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from patterns import name_rel, house_rel  # noqa: E402  (module-level loop guarded below)

A = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/artifacts"
S = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/submissions"
res = {}
for c in ["France", "US", "India"]:
    q = pd.read_parquet(os.path.join(A, "block", f"test_{c}_q.parquet"), columns=["entity_id", "core", "house"])
    r = pd.read_parquet(os.path.join(A, "block", f"test_{c}_r.parquet"), columns=["entity_id", "core", "house"])
    qpos = pd.Series(np.arange(len(q)), index=q.entity_id)
    rpos = pd.Series(np.arange(len(r)), index=r.entity_id)
    for up in ["upload_1", "upload_2"]:
        m = pd.read_csv(os.path.join(S, up, "matching_results.tsv"), sep="\t", dtype=str, keep_default_na=False)
        m = m[m.source1_entity_id.isin(qpos.index)]
        m = m[m.matched_entity_ids != ""]
        m = m.assign(rid=m.matched_entity_ids.str.split(",")).explode("rid")
        qi, ri = qpos.reindex(m.source1_entity_id).to_numpy(), rpos.reindex(m.rid).to_numpy()
        nrel = [name_rel(a, b) for a, b in zip(q.core.to_numpy()[qi], r.core.to_numpy()[ri])]
        hrel = [house_rel(a, b) for a, b in zip(q.house.to_numpy()[qi], r.house.to_numpy()[ri])]
        t = pd.Series(1, index=pd.MultiIndex.from_arrays([nrel, hrel])).groupby(level=[0, 1]).sum() / len(q)
        res[f"{c}_{up}"] = t
        print(c, up, "links/S1", round(len(m) / len(q), 4), flush=True)
for c in ["US", "India"]:
    d = pd.read_parquet(f"pat_dev_{c}.parquet")
    n_s1 = pd.read_parquet(os.path.join(A, "block", f"dev_{c}_q.parquet"), columns=["entity_id"]).shape[0]
    t = d[d.y == 1].groupby(["nrel", "hrel"]).size() / n_s1
    res[f"{c}_dev_truth"] = t
T = pd.DataFrame(res).fillna(0)
T = T[(T > 0.004).any(axis=1)]
pd.set_option("display.width", 250)
print(T.round(4).to_string())
T.to_csv("upload_patterns.csv")
