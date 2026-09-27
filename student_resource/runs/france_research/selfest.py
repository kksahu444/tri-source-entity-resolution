"""Model's own expected F0.5 per country (GFM, independence) vs realised F on dev.

If France's self-estimate is far above the probe's 0.928 the model is confidently wrong on France.
"""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/code/business_entity_resolution")
from src.assign import hard_assign
from src.decision import gfm_rule

A = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/artifacts"
dec = pickle.load(open(os.path.join(A, "decision.pkl"), "rb"))
iso = dec["iso"]
variant = sys.argv[1] if len(sys.argv) > 1 else "stageA"
out = {}
for split, cs in (("dev", ["US", "India"]), ("test", ["US", "India", "France"])):
    for c in cs:
        f = f"{split}_{c}_stageA.parquet" if variant == "stageA" else f"{split}_{c}.parquet"
        d = pd.read_parquet(os.path.join(A, "pred", f))
        n_s1 = pd.read_parquet(os.path.join(A, "block", f"{split}_{c}_q.parquet"), columns=["entity_id"]).shape[0]
        p = iso.predict(d["p"].to_numpy())
        keep = hard_assign(d["ri"].to_numpy(), p)
        qi, pk = d["qi"].to_numpy()[keep], p[keep]
        sel, ef = gfm_rule.select(qi, pk, None)
        e = np.ones(n_s1)          # S1s without candidates: the model is sure they are empty
        e[list(ef.keys())] = list(ef.values())
        unc = np.bincount(qi, weights=np.minimum(pk, 1 - pk), minlength=n_s1)
        band = np.bincount(qi, weights=((pk > 0.1) & (pk < 0.9)).astype(float), minlength=n_s1)
        row = {"selfest_F": e.mean(), "unc_mass_per_s1": unc.mean(), "band_0.1-0.9_per_s1": band.mean(),
               "share_s1_with_band": (band > 0).mean(), "links_per_s1": sel.sum() / n_s1,
               "empty_rate": 1 - np.bincount(qi[sel], minlength=n_s1).astype(bool).mean(),
               "pairs_per_s1": len(d) / n_s1}
        if "y" in d:
            from src.metrics import per_entity
            yk = d["y"].to_numpy()[keep]
            # n_true per S1 from all pairs is a lower bound (blocking misses excluded); good enough for the gap
            nt = np.bincount(d["qi"].to_numpy(), weights=d["y"].to_numpy(), minlength=n_s1)
            row["real_F_on_cands"] = per_entity(qi, sel, yk, nt).mean()
        out[(split, c)] = row
        print(split, c, {k: round(v, 5) for k, v in row.items()}, flush=True)
print(pd.DataFrame(out).T.round(5).to_string())
