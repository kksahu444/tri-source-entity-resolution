"""Print France S1s with their candidates (stage-A p, calibrated) for eyeballing."""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/code/business_entity_resolution")
from src.assign import hard_assign
from src.decision import gfm_rule

A = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/artifacts"
C = sys.argv[1] if len(sys.argv) > 1 else "France"
mode = sys.argv[2] if len(sys.argv) > 2 else "band"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 15
split = sys.argv[4] if len(sys.argv) > 4 else "test"
iso = pickle.load(open(os.path.join(A, "decision.pkl"), "rb"))["iso"]
d = pd.read_parquet(os.path.join(A, "pred", f"{split}_{C}_stageA.parquet"))
q = pd.read_parquet(os.path.join(A, "block", f"{split}_{C}_q.parquet"), columns=["entity_id", "name_b", "addr_b"])
r = pd.read_parquet(os.path.join(A, "block", f"{split}_{C}_r.parquet"), columns=["entity_id", "name_b", "addr_b"])
d["pc"] = iso.predict(d["p"].to_numpy())
keep = hard_assign(d["ri"].to_numpy(), d["pc"].to_numpy())
d["kept"] = keep
sel, ef = gfm_rule.select(d["qi"].to_numpy()[keep], d["pc"].to_numpy()[keep], None)
d["sel"] = False
d.loc[np.flatnonzero(keep)[sel], "sel"] = True
d = d[d.pc > 0.01]
band = d[(d.pc > 0.2) & (d.pc < 0.8)].qi.unique()
rng = np.random.default_rng(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
pick = rng.choice(band if mode == "band" else d.qi.unique(), N, replace=False)
for qi in pick:
    s = q.iloc[qi]
    print(f"\n=== {s.entity_id} | {s.name_b} | {s.addr_b}   (expF {ef.get(qi, 1):.3f})")
    g = d[d.qi == qi].sort_values("pc", ascending=False)
    for _, x in g.iterrows():
        rr = r.iloc[int(x.ri)]
        flag = ("SEL " if x.sel else "    ") + ("" if x.kept else "lost ")
        extra = f" y={int(x.y)}" if "y" in g else ""
        print(f"  {x.pc:5.3f} {flag:9s} {rr.entity_id:14s} {rr.name_b[:40]:40s} | {rr.addr_b[:70]}{extra}")
