"""Stage a Kaggle run's submission as a leaderboard upload (work order: every upload validated and logged).

1. download the kernel output (output/, runs/) into artifacts/kaggle_out/<name>/
2. official validator on matching_results.tsv + candidate_pairs.tsv (must exit 0)
3. acceptance check against train truth (src.acceptance; must pass, runs/decoy_shift.md)
4. copy both files to submissions/<name>/, SHA-256 of each, and append a manifest row with the run's
   dev OOF (from its decide dev run-log row), the chosen variant and the acceptance numbers

python -m src.stage_upload --kernel aditya103856/amlc26-pipeline-cpu --name upload_3 --config config_v5.yaml
Nothing is uploaded to the leaderboard: that stays a manual step.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

from .acceptance import check
from .utils import load_config


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default="config_v5.yaml")
    ap.add_argument("--skip_download", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    root = os.path.dirname(os.path.abspath(cfg["paths"]["dataset"]))          # student_resource/
    out = os.path.join(cfg["paths"]["artifacts"], "kaggle_out", args.name)
    if not args.skip_download:
        os.makedirs(out, exist_ok=True)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        subprocess.run(["kaggle", "kernels", "output", args.kernel, "-p", out, "-o"], check=True, env=env)
    match, cand = os.path.join(out, "output", "matching_results.tsv"), os.path.join(out, "output", "candidate_pairs.tsv")
    assert os.path.exists(match), f"no matching_results.tsv in {out}/output"
    val = subprocess.run([sys.executable, os.path.join(root, "utils", "validate_submission.py"), "--matching", match,
                          "--candidate", cand, "--test-dir", os.path.join(cfg["paths"]["dataset"], "test")],
                         capture_output=True, text=True)
    print(val.stdout[-3000:], val.stderr[-2000:])
    acc = check(cfg, match)
    print(json.dumps(acc, indent=1))
    if val.returncode != 0 or not acc["pass"]:
        print(f"NOT STAGED: validator exit {val.returncode}, acceptance pass {acc['pass']}")
        return 1
    log = pd.read_csv(os.path.join(out, "runs", "run_log.csv"), dtype=str, keep_default_na=False)
    dec = log[(log["stage"] == "decide") & (log["data_slice"].str.startswith("dev"))].iloc[-1]
    notes = json.loads(dec["notes"]) if dec.get("notes") else {}
    dst = os.path.join(cfg["paths"]["submissions"], args.name)
    os.makedirs(dst, exist_ok=True)
    for f in (match, cand):
        shutil.copy(f, dst)
    sub = log[log["stage"] == "submit"].iloc[-1]
    row = {"file": f"{args.name}/matching_results.tsv", "run_id": f"{args.kernel.split('/')[-1]}:{sub['run_id']}",
           "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "oof_f05": f"{dec.get('oof_f05', '')} {dec.get('oof_f05_ci', '')} ({dec['run_id']})", "public_score": "",
           "notes": (f"variant {notes.get('stack_variant')}, rule {notes.get('rule')}, variants {notes.get('variants')}; "
                     f"validator PASS; acceptance PASS "
                     + ", ".join(f"{c} near_beside {v['near_beside']} (train {v['train_near_beside']}) links {v['links']}"
                                 for c, v in acc["countries"].items())
                     + f"; sha256 matching {sha256(os.path.join(dst, 'matching_results.tsv'))[:16]}")}
    man = os.path.join(cfg["paths"]["submissions"], "manifest.csv")
    m = pd.read_csv(man, dtype=str, keep_default_na=False)
    pd.concat([m, pd.DataFrame([row])], ignore_index=True).to_csv(man, index=False, lineterminator="\n")
    print("STAGED", json.dumps(row, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
