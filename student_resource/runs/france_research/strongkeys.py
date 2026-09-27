"""Recall probe without labels: near-certain pairs (strong keys) and where they drop out.

Key A: same house + house_sfx + city, street token-set >= 0.8 and >= 1 shared name token (address anchor).
Key B: same sorted core name (>= 2 tokens) + same house (name anchor, street free).
For each key pair: retrieved (block union), kept (pruned set), linked (upload 1 / dev selection), and on dev
the truth rate. A key pair that is true on dev 99% of the time and is lost more often in France than in the
US points at a France-specific recall leak.
"""
import os
import numpy as np, pandas as pd
from rapidfuzz import fuzz

A = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/artifacts"
S = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/submissions"
DS = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/6ab10eb3b23ba_student_resource/student_resource/dataset"
cols = ["entity_id", "core", "house", "house_sfx", "street", "city"]


def keypairs(q, r):
    q = q.assign(qi=np.arange(len(q)))
    r = r.assign(ri=np.arange(len(r)))
    ka_q = q.house + "|" + q.house_sfx + "|" + q.city
    ka_r = r.house + "|" + r.house_sfx + "|" + r.city
    qa = q[(q.house != "") & (ka_q.map(ka_q.value_counts()) <= 5)]
    ra = r[(r.house != "") & (ka_r.map(ka_r.value_counts()) <= 20)]
    a = qa.merge(ra, on=["house", "house_sfx", "city"], suffixes=("_q", "_r"))
    a = a[[fuzz.token_set_ratio(x, y) >= 80 for x, y in zip(a.street_q, a.street_r)]]
    a = a[[bool(set(x.split()) & set(y.split())) for x, y in zip(a.core_q, a.core_r)]]
    q["sk"] = q.core.str.split().map(lambda t: " ".join(sorted(t)) if len(t) >= 2 else "")
    r["sk"] = r.core.str.split().map(lambda t: " ".join(sorted(t)) if len(t) >= 2 else "")
    kb_q, kb_r = q.sk + "|" + q.house, r.sk + "|" + r.house
    b = q[(q.sk != "") & (q.house != "") & (kb_q.map(kb_q.value_counts()) <= 5)].merge(
        r[(r.sk != "") & (r.house != "") & (kb_r.map(kb_r.value_counts()) <= 20)], on=["sk", "house"])
    out = pd.concat([a[["qi", "ri"]].assign(key="A"), b[["qi", "ri"]].assign(key="B")])
    return out


def mark(kp, pairs_path, feat_path, linked_set, n_r):
    code = kp.qi.to_numpy(np.int64) * n_r + kp.ri.to_numpy(np.int64)
    bp = pd.read_parquet(pairs_path, columns=["qi", "ri"])
    kp["retrieved"] = np.isin(code, bp.qi.to_numpy(np.int64) * n_r + bp.ri.to_numpy(np.int64))
    del bp
    fp = pd.read_parquet(feat_path, columns=["qi", "ri"])
    kp["kept"] = np.isin(code, fp.qi.to_numpy(np.int64) * n_r + fp.ri.to_numpy(np.int64))
    del fp
    kp["linked"] = np.isin(code, linked_set)
    return kp


rows = []
for split, c in [("test", "France"), ("dev", "US"), ("test", "US")]:
    q = pd.read_parquet(os.path.join(A, "block", f"{split}_{c}_q.parquet"), columns=cols)
    r = pd.read_parquet(os.path.join(A, "block", f"{split}_{c}_r.parquet"), columns=cols)
    n_r = len(r)
    kp = keypairs(q, r)
    # a record may key to several S1s (co-located / name crowd): note how many
    kp["r_nkey"] = kp.groupby(["key", "ri"]).qi.transform("size")
    if split == "test":
        m = pd.read_csv(os.path.join(S, "upload_1", "matching_results.tsv"), sep="\t", dtype=str, keep_default_na=False)
        qpos = pd.Series(np.arange(len(q)), index=q.entity_id)
        rpos = pd.Series(np.arange(len(r)), index=r.entity_id)
        m = m[m.source1_entity_id.isin(qpos.index) & (m.matched_entity_ids != "")]
        m = m.assign(rid=m.matched_entity_ids.str.split(",")).explode("rid")
        linked = qpos.reindex(m.source1_entity_id).to_numpy(np.int64) * n_r + rpos.reindex(m.rid).to_numpy(np.int64)
    else:
        d = pd.read_parquet(f"pat_dev_{c}.parquet")
        d = d[d.sel]
        linked = d.qi.to_numpy(np.int64) * n_r + d.ri.to_numpy(np.int64)
    kp = mark(kp, os.path.join(A, "block", f"{split}_{c}_pairs.parquet"), os.path.join(A, "feat", f"{split}_{c}.parquet"),
              linked, n_r)
    if split == "dev":
        gt = pd.read_csv(os.path.join(DS, "train", "train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)
        gt = gt[gt.source1_entity_id.isin(set(q.entity_id)) & (gt.matched_entity_ids != "")]
        gt = gt.assign(rid=gt.matched_entity_ids.str.split(",")).explode("rid")
        qpos = pd.Series(np.arange(len(q)), index=q.entity_id)
        rpos = pd.Series(np.arange(len(r)), index=r.entity_id)
        gt = gt[gt.rid.isin(rpos.index)]
        tc = qpos.reindex(gt.source1_entity_id).to_numpy(np.int64) * n_r + rpos.reindex(gt.rid).to_numpy(np.int64)
        kp["y"] = np.isin(kp.qi.to_numpy(np.int64) * n_r + kp.ri.to_numpy(np.int64), tc)
    for (key, uniq), g in kp.groupby(["key", kp.r_nkey == 1]):
        row = {"key": f"{split}_{c}", "anchor": key, "record_unique": uniq, "pairs_per_s1": len(g) / len(q),
               "retrieved": g.retrieved.mean(), "kept": g.kept.mean(), "linked": g.linked.mean()}
        if "y" in g:
            row["true_rate"] = g.y.mean()
            row["linked_given_true"] = g.linked[g.y].mean()
        rows.append(row)
    print(split, c, "done", flush=True); print(pd.DataFrame(rows[-4:]).round(4).to_string(), flush=True)
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(T.round(4).to_string())
