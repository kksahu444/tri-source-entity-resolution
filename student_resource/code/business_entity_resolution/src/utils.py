"""Config loading, seeding, logging and the run log (spec section 9)."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import logging
import os
import random
import subprocess
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/business_entity_resolution

RUN_LOG_COLUMNS = [
    "run_id", "timestamp", "git_commit", "config_hash", "seed", "stage", "data_slice",
    "PC_union_US", "PC_union_India", "RR_US", "RR_India", "cands_per_s1",
    "oof_f05", "oof_f05_ci", "oof_f05_US", "oof_f05_India", "oof_f05_singleton", "oof_f05_nonsingleton",
    "loco_US_to_India", "loco_India_to_US", "runtime_s", "notes",
]


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = os.path.normpath(os.path.join(ROOT, v))
    cfg["_hash"] = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def new_run_id(stage: str) -> str:
    return f"{dt.datetime.now().strftime('%m%d-%H%M%S')}-{stage}"


def get_logger(name: str = "er") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def append_run_log(cfg: dict, row: dict) -> None:
    path = os.path.join(cfg["paths"]["runs"], "run_log.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    base = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
            "config_hash": cfg["_hash"], "seed": cfg["seed"]}
    base.update(row)
    extra = {k: v for k, v in base.items() if k not in RUN_LOG_COLUMNS}
    if extra:
        base["notes"] = (str(base.get("notes", "")) + " " + json.dumps(extra, default=str)).strip()
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_LOG_COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(base)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return round(time.time() - self.t0, 1)
