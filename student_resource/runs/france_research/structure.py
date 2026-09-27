"""Structural differences between France and US/India S1 tables (test) and dev: name/address collisions."""
import os
import numpy as np, pandas as pd

A = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/artifacts/block"
cols = ["core", "house", "street", "city", "region", "name_b", "legal"]
rows = {}
for f in ["dev_US", "dev_India", "test_US", "test_India", "test_France"]:
    q = pd.read_parquet(os.path.join(A, f + "_q.parquet"), columns=cols)
    n = len(q)
    core_city = q.groupby(["core", "city"]).core.transform("size")
    core_any = q.groupby("core").core.transform("size")
    addr = q.groupby(["house", "street", "city"]).core.transform("size")
    street = q.groupby(["street", "city"]).core.transform("size")
    toks = q["core"].str.split()
    vocab = pd.Series([t for ts in toks for t in ts]).value_counts()
    tok_df = toks.map(lambda ts: min((vocab[t] for t in ts), default=0))
    rows[f] = {
        "n_s1": n,
        "n_cities": q["city"].nunique(),
        "s1_per_city_median": q.groupby("city").size().median(),
        "share_core_dup_same_city": (core_city > 1).mean(),
        "share_core_dup_anywhere": (core_any > 1).mean(),
        "share_addr_dup": ((addr > 1) & (q["house"] != "")).mean(),
        "s1_per_street_mean": street.mean(),
        "name_vocab_size": len(vocab),
        "tokens_per_name": toks.map(len).mean(),
        "median_rarest_token_df": tok_df.median(),
        "share_rarest_token_df_gt_100": (tok_df > 100).mean(),
        "share_legal_empty": (q["legal"] == "").mean(),
    }
    print(f, rows[f], flush=True)
print(pd.DataFrame(rows).T.to_string())
