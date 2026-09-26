"""Stage `submit`: write both TSVs, run the organisers' validator, archive with the run id."""
from __future__ import annotations

import csv
import datetime as dt
import glob
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from . import io_utils
from .blocking.union import block_dir
from .featurize import countries, feat_dir
from .stack import pred_dir
from .utils import Timer, get_logger, new_run_id

log = get_logger()


def run(cfg: dict, split: str = "test") -> dict:
    t = Timer()
    root, out_dir = cfg["paths"]["dataset"], cfg["paths"]["output"]
    s1_order = io_utils.read_tsv(io_utils.source_path(root, "test", 1))["entity_id"].tolist()
    sel = pd.read_parquet(os.path.join(pred_dir(cfg), "test_selected.parquet"))
    cand_s1, cand_c, m_s1, m_c, valid = [], [], [], [], set()
    for c in countries(cfg, "test"):
        q = pd.read_parquet(os.path.join(block_dir(cfg), f"test_{c}_q.parquet"), columns=["entity_id"])["entity_id"].to_numpy()
        r = pd.read_parquet(os.path.join(block_dir(cfg), f"test_{c}_r.parquet"), columns=["entity_id"])["entity_id"].to_numpy()
        valid.update(r.tolist())
        f = pd.read_parquet(os.path.join(feat_dir(cfg), f"test_{c}.parquet"), columns=["qi", "ri"])
        cand_s1.append(q[f["qi"].to_numpy()])
        cand_c.append(r[f["ri"].to_numpy()])
        s = sel[sel["country"] == c]
        m_s1.append(q[s["qi"].to_numpy()])
        m_c.append(r[s["ri"].to_numpy()])
    match_path = os.path.join(out_dir, "matching_results.tsv")
    cand_path = os.path.join(out_dir, "candidate_pairs.tsv")
    stats = io_utils.write_submission(match_path, cand_path, s1_order, valid,
                                      np.concatenate(m_s1), np.concatenate(m_c),
                                      np.concatenate(cand_s1), np.concatenate(cand_c))
    validator = os.path.join(os.path.dirname(root), "utils", "validate_submission.py")
    proc = subprocess.run([sys.executable, validator, "--matching", match_path, "--candidate", cand_path,
                           "--test-dir", os.path.join(root, "test")], capture_output=True, text=True)
    log.info("validator:\n%s", proc.stdout[-2000:])
    passed = "PASS" in proc.stdout and proc.returncode == 0
    assert passed, "validator did not print PASS"
    run_id = new_run_id("submit")
    sub_dir = cfg["paths"]["submissions"]
    os.makedirs(sub_dir, exist_ok=True)
    day = dt.date.today().strftime("%Y%m%d")
    n = len(glob.glob(os.path.join(sub_dir, f"sub_{day}_*_matching.tsv"))) + 1
    base = os.path.join(sub_dir, f"sub_{day}_{n}_{run_id}")
    shutil.copy(match_path, base + "_matching.tsv")
    shutil.copy(cand_path, base + "_candidates.tsv")
    arch = os.path.join(cfg["paths"]["artifacts"], "archive", run_id)
    os.makedirs(arch, exist_ok=True)
    for src in (os.path.join(feat_dir(cfg), "pruner.pkl"), os.path.join(feat_dir(cfg), "stageA.pkl"),
                os.path.join(cfg["paths"]["artifacts"], "decision.pkl")):
        if os.path.exists(src):
            shutil.copy(src, arch)
    shutil.copy(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"), arch)
    man = os.path.join(sub_dir, "manifest.csv")
    new = not os.path.exists(man)
    with open(man, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["file", "run_id", "created", "oof_f05", "public_score", "notes"])
        w.writerow([os.path.basename(base) + "_matching.tsv", run_id, dt.datetime.now().isoformat(timespec="seconds"),
                    "", "", ""])
    return {"stage": "submit", "runtime_s": t(), "notes": f"{stats} validator=PASS archived={os.path.basename(base)}"}
