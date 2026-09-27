"""Paired comparison of two runs on the same dev entities (decide writes runs/entity_f05_dev.parquet).

Each run is scored with the variant its own decide stage kept (stack_variant in the decide dev row of
runs/run_log.csv). Prints both means, the paired gain of B over A with a 1,000-resample bootstrap interval,
and the same per country when the run log has it.

python -m src.compare_runs --a ../../artifacts/kaggle_out/cpu_v4/runs --b ../../artifacts/kaggle_out/v7b/runs
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def _run(runs_dir: str) -> pd.Series:
    f = pd.read_parquet(os.path.join(runs_dir, "entity_f05_dev.parquet"))
    log = pd.read_csv(os.path.join(runs_dir, "run_log.csv"), dtype=str, keep_default_na=False)
    dec = log[(log["stage"] == "decide") & log["data_slice"].str.startswith("dev")].iloc[-1]
    var = json.loads(dec["notes"]).get("stack_variant", "base")
    return f.set_index("entity_id")[f"f05_{var}"].rename(var)


def paired(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    d = b - a
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="runs/ folder of the reference run")
    ap.add_argument("--b", required=True, help="runs/ folder of the new run")
    args = ap.parse_args(argv)
    a, b = _run(args.a), _run(args.b)
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    g, lo, hi = paired(j["a"].to_numpy(), j["b"].to_numpy())
    print(json.dumps({"entities": len(j), "only_a": int(len(a) - len(j)), "only_b": int(len(b) - len(j)),
                      "a": {"variant": a.name, "f05": round(float(j["a"].mean()), 5)},
                      "b": {"variant": b.name, "f05": round(float(j["b"].mean()), 5)},
                      "gain_b_over_a": round(g, 5), "ci95": [round(lo, 5), round(hi, 5)]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
