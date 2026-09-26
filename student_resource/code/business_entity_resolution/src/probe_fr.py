"""Work order T1: France wrong-ID probe (diagnostic upload only, never the last upload).

Replaces each French list of an uploaded matching file with one distinct US S2 ID that no current list
uses; every other row stays byte-identical. After the lead uploads it:
    F_FR = (S_upload - S_probe) / 0.1498   (uncertainty: leaderboard rounding / 0.1498, +-0.004 for France's
                                            share of the public subset)

Run from student_resource/:
    python code/business_entity_resolution/src/probe_fr.py --upload submissions/upload_1/matching_results.tsv
"""
import argparse
import csv
import datetime as dt
import os
import subprocess
import sys

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", default="submissions/upload_1/matching_results.tsv")
    ap.add_argument("--out", default="submissions/probe_fr_wrongid/matching_results.tsv")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--base-run", default="0926-073916-submit")
    args = ap.parse_args(argv)
    T = args.test_dir
    rd = lambda p, cols=None: pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, usecols=cols)  # default quoting
    s1 = rd(f"{T}/test_source1.tsv", ["entity_id", "country"])
    s2 = rd(f"{T}/test_source2.tsv", ["entity_id", "country"])
    for path, df in ((f"{T}/test_source1.tsv", s1), (f"{T}/test_source2.tsv", s2)):
        with open(path, "rb") as f:
            assert len(df) == sum(1 for _ in f) - 1, path
    m = rd(args.upload)
    assert list(m.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert len(m) == len(s1) and (m.source1_entity_id.values == s1.entity_id.values).all()
    used = {i for lst in m.matched_entity_ids for i in lst.split(",") if i}
    pool = [i for i in s2.loc[s2.country == "US", "entity_id"] if i not in used]   # 0 cross-country true pairs in train
    fr = (s1.country == "France").values
    assert len(pool) >= fr.sum(), "not enough unused US S2 IDs"
    probe = m.copy()
    probe.loc[fr, "matched_entity_ids"] = pool[: fr.sum()]                         # one distinct wrong ID per French S1
    assert (probe.loc[~fr, "matched_entity_ids"].values == m.loc[~fr, "matched_entity_ids"].values).all()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    probe.to_csv(args.out, sep="\t", index=False, lineterminator="\n")
    with open(args.out, "rb") as f:
        assert b"\r" not in f.read(), "CRLF in output"
    print(f"French rows replaced: {fr.sum():,}; other rows unchanged: {(~fr).sum():,}")
    val = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", args.out,
                          "--candidate", "none", "--test-dir", T], capture_output=True, text=True)
    print(val.stdout[-600:])
    man = "submissions/manifest.csv"
    new = not os.path.exists(man)
    with open(man, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if new:
            w.writerow(["file", "run_id", "created", "oof_f05", "public_score", "notes"])
        w.writerow([args.out.replace("submissions/", ""), f"probe-of-{args.base_run}", dt.datetime.now().isoformat(timespec="seconds"),
                    "", "", f"purpose: diagnostic (France wrong-ID probe of {args.base_run}); F_FR = (S_base - S_probe) / 0.1498; never the last upload"])


if __name__ == "__main__":
    main()
