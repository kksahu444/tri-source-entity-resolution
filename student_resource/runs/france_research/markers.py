"""Noise-marker natural experiment (label-free recall/precision damage per noise type).

The generator applies surface noise (N° prefixes, abbreviations, all caps, accents...) independently of which
entity a record belongs to, so each marker should leave a record's chance of being linked unchanged. A marker
whose records are linked less often than unmarked ones shows where the pipeline loses matches. On dev the same
table is computed with the truth (owned share) to check that link-rate gaps track real recall gaps.
"""
import os, re
import numpy as np, pandas as pd

A = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/artifacts"
S = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/submissions"
DS = r"c:/Users/LawLight/Desktop/tri-source-entity-resolution/student_resource/6ab10eb3b23ba_student_resource/student_resource/dataset"

MARKERS = {
    "addr_no_prefix": lambda r: r.addr_b.str.contains(r"(?:^|, )(?:no|ndeg|n) ?\d", regex=True),
    "addr_hash": lambda r: r.addr_a.str.contains("#", regex=False),
    "street_empty_w_house": lambda r: (r.street == "") & (r.house != "") & (r.addr_b != ""),
    "city_has_digit": lambda r: r.city.str.contains(r"\d", regex=True),
    "addr_empty": lambda r: r.addr_b == "",
    "addr_allcaps": lambda r: (r.addr_a == r.addr_a.str.upper()) & r.addr_a.str.contains("[A-Z]", regex=True),
    "name_allcaps": lambda r: (r.name_a == r.name_a.str.upper()) & r.name_a.str.contains("[A-Z]", regex=True),
    "addr_fffd": lambda r: r.addr_a.str.contains("\ufffd", regex=False),
    "name_fffd": lambda r: r.name_a.str.contains("\ufffd", regex=False),
    "street_abbr_r": lambda r: r.addr_b.str.contains(r"(?:^|\s)r\s", regex=True),
    "street_abbr_other": lambda r: r.addr_b.str.contains(r"(?:^|\s)(?:av|bd|imp|pl|ch|rte|all|sq|crs)\s", regex=True),
    "bis_ter": lambda r: r.house_sfx != "",
    "region_missing": lambda r: (r.region == "") & (r.addr_b != ""),
    "website_name": lambda r: r.name_b.str.contains(r"com$|\.com|www", regex=True),
    "brackets": lambda r: r.name_a.str.contains(r"[()\[\]]", regex=True),
    "legal_lead": lambda r: r.f_lead == 1,
    "name_tag": lambda r: r.f_tag == 1,
    "aka": lambda r: r.name_b.str.contains(r"\ba k a\b|\baka\b|\bdba\b", regex=True),
    "addr_3plus_parts": lambda r: r.n_parts >= 4,
}
COLS = ["entity_id", "name_a", "addr_a", "name_b", "addr_b", "house", "house_sfx", "street", "city", "region",
        "f_lead", "f_tag", "n_parts"]


def table(r, linked, owned=None):
    rows = {}
    base = linked.mean()
    for k, f in MARKERS.items():
        m = f(r).to_numpy(bool)
        if m.sum() < 200:
            continue
        row = {"share": m.mean(), "link_rate_marked": linked[m].mean(), "link_rate_clean": linked[~m].mean()}
        row["link_gap"] = row["link_rate_marked"] - row["link_rate_clean"]
        if owned is not None:
            row["owned_marked"] = owned[m].mean()
            row["owned_clean"] = owned[~m].mean()
            row["owned_gap"] = row["owned_marked"] - row["owned_clean"]
            row["recall_marked"] = (linked & owned)[m].sum() / max(owned[m].sum(), 1)
            row["recall_clean"] = (linked & owned)[~m].sum() / max(owned[~m].sum(), 1)
        rows[k] = row
    return pd.DataFrame(rows).T, base


pd.set_option("display.width", 250)
for c in ["France", "US", "India"]:
    r = pd.read_parquet(os.path.join(A, "block", f"test_{c}_r.parquet"), columns=COLS)
    for up in ["upload_1"]:
        m = pd.read_csv(os.path.join(S, up, "matching_results.tsv"), sep="\t", dtype=str, keep_default_na=False)
        ids = set(i for lst in m.matched_entity_ids for i in lst.split(",") if i)
        linked = r.entity_id.isin(ids).to_numpy()
        t, base = table(r, linked)
        print(f"\n### test {c} {up}: records {len(r):,}, base link rate {base:.4f}")
        print(t.round(4).to_string())

# dev with truth (iteration-2 stage-A selection; the dev pool withholds 19% of dev S1s)
gt = pd.read_csv(os.path.join(DS, "train", "train_ground_truth.tsv"), sep="\t", dtype=str, keep_default_na=False)
gt = gt[gt.matched_entity_ids != ""]
for c in ["US", "India"]:
    q = pd.read_parquet(os.path.join(A, "block", f"dev_{c}_q.parquet"), columns=["entity_id"])
    r = pd.read_parquet(os.path.join(A, "block", f"dev_{c}_r.parquet"), columns=COLS)
    g = gt[gt.source1_entity_id.isin(set(q.entity_id))]
    owned_ids = set(i for lst in g.matched_entity_ids for i in lst.split(","))
    owned = r.entity_id.isin(owned_ids).to_numpy()
    d = pd.read_parquet(f"pat_dev_{c}.parquet")
    linked = np.zeros(len(r), bool)
    linked[d.ri[d.sel].to_numpy()] = True
    t, base = table(r, linked, owned)
    print(f"\n### dev {c}: records {len(r):,}, base link rate {base:.4f}, owned {owned.mean():.4f}")
    print(t.round(4).to_string())
