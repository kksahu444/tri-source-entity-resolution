"""French noise audit on the test files (spec section 4). No labels needed.

Every rate is computed per (source, country) so French S1 -> S2/S3 drift can be compared with the
US and India drift, which the model does see in training.

Run from ``student_resource/``:  python code/business_entity_resolution/src/france_audit.py --root dataset
"""
import argparse
import sys

import pandas as pd

LEGAL = ["sarl", "sas", "sasu", "sa", "eurl", "sci", "ei", "snc", "soci[eé]t[eé]", "llc", "inc", "ltd",
         "limited", "private", "pvt", "corp", "corporation", "co", "llp", "pc", "lp"]
STREETS = {
    "rue": r"rue", "r.": r"r\.", "avenue": r"avenue", "av/av.": r"av\.?", "ave": r"ave\.?",
    "boulevard": r"boulevard", "bd/bd.": r"bd\.?", "bld/blvd": r"bl?vd\.?|bld\.?",
    "place": r"place", "pl/pl.": r"pl\.?", "chemin": r"chemin", "ch/ch.": r"ch\.?",
    "allee": r"all[ée]e", "impasse": r"impasse", "imp": r"imp\.?", "quai": r"quai",
    "route": r"route", "rte": r"rte\.?", "faubourg": r"faubourg", "fbg": r"fbg\.?",
    "street": r"street", "st/st.": r"st\.?", "road": r"road", "rd/rd.": r"rd\.?",
}


def read(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    with open(path, "rb") as f:
        n_lines = sum(1 for _ in f)
    assert len(df) == n_lines - 1
    return df


def rate(s, pat):
    return s.str.contains(pat, case=False, regex=True).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--sample", type=int, default=300_000, help="rows per (source, country); 0 = all")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    frames = []
    for k in (1, 2, 3):
        df = read(f"{args.root}/test/test_source{k}.tsv")
        df["src"] = f"S{k}"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if args.sample:
        df = (df.groupby(["src", "country"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), args.sample), random_state=0)))

    rows = []
    for (src, ctry), g in df.groupby(["src", "country"]):
        n, a = g["business_name"], g["business_address"]
        r = {"src": src, "country": ctry, "n": len(g)}
        r["name_accent"] = rate(n, r"[À-ÖØ-öø-ÿ]")
        r["addr_accent"] = rate(a, r"[À-ÖØ-öø-ÿ]")
        r["name_allcaps"] = (n.str.upper() == n).mean()
        r["addr_allcaps"] = ((a.str.upper() == a) & (a != "")).mean()
        r["france_tag"] = rate(n, r"\(france\)")
        r["india_tag"] = rate(n, r"\(india\)")
        r["lead_junk"] = rate(n, r"^\s*(?:[(#@<>\"*\-.~\[]|<<|>>)")
        r["trail_bang"] = rate(n, r"!\s*$")
        r["brackets"] = rate(n, r"[\[\]()]")
        r["website"] = rate(n, r"\.(?:com|net|org|fr|in)\b")
        r["name_digits"] = rate(n, r"\d")
        r["bis_ter"] = rate(a, r"\b\d+\s*(?:bis|ter)\b")
        r["saint_full"] = rate(a, r"\bsaint[e]?\b")
        r["st_dash"] = rate(a, r"\bst[e]?-")
        r["st_space"] = rate(a, r"\bst[e]?\.?\s")
        r["elision"] = rate(a + " " + n, r"\b[ld]['’]")
        r["addr_empty"] = (a.str.strip() == "").mean()
        parts = a.str.count(",") + (a.str.strip() != "")
        r["addr_parts_mean"] = parts.mean()
        r["addr_starts_digit"] = rate(a, r"^\s*\d")
        r["name_tokens_mean"] = n.str.split().str.len().mean()
        r["addr_tokens_mean"] = a.str.split().str.len().mean()
        r["name_chars_mean"] = n.str.len().mean()
        for lf in LEGAL:
            r[f"legal_any_{lf[:8]}"] = rate(n, rf"\b{lf}\b\.?")
            r[f"legal_lead_{lf[:8]}"] = rate(n, rf"^\W*{lf}\b")
        for key, pat in STREETS.items():
            r[f"street_{key}"] = rate(a, rf"(?:^|[\s,]){pat}(?=\s)")
        rows.append(r)

    out = pd.DataFrame(rows).set_index(["country", "src"]).sort_index().T
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 20)
    print("French noise audit (test files; rates are shares of records; sample per (src, country) ="
          f" {args.sample or 'all'})\n")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
