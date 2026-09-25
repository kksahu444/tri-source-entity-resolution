"""Single entry point:  python -m src.run_pipeline --config config.yaml --stage <stage>"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from .utils import append_run_log, get_logger, load_config, new_run_id, seed_everything

STAGES = {
    "prep": "src.prep",
    "block": "src.blocking.union",
    "features": "src.features",
    "stack": "src.stack",
    "decide": "src.decide",
    "submit": "src.submit",
}
ORDER = ["prep", "block", "features", "stack", "decide", "submit"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    ap.add_argument("--split", default=None, help="dev | test (stages that support it)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    log = get_logger()
    log.info("config %s flags %s", cfg["_hash"], json.dumps(cfg["flags"]))
    for stage in (ORDER if args.stage == "all" else [args.stage]):
        run_id = new_run_id(stage)
        mod = importlib.import_module(STAGES[stage])
        kwargs = {"split": args.split} if args.split else {}
        res = mod.run(cfg, **kwargs) or {}
        res.setdefault("stage", stage)
        append_run_log(cfg, {"run_id": run_id, "data_slice": args.split or "all", **res})
        log.info("stage %s done: %s", stage, {k: v for k, v in res.items() if k != "notes"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
