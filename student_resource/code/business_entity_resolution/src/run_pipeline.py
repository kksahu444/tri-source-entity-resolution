"""Single entry point:  python -m src.run_pipeline --config config.yaml --stage <stage>"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from .utils import append_run_log, get_logger, load_config, new_run_id, seed_everything

STAGES = {
    "prep": "src.prep",
    "translit": "src.translit",
    "block": "src.blocking.union",
    "features": "src.featurize",
    "stack": "src.stack",
    "ce": "src.cross_encoder",
    "stackb": "src.stack_b",
    "decide": "src.decide",
    "loco": "src.france",
    "submit": "src.submit",
}
ORDER = ["prep", "translit", "block", "features", "stack", "decide", "submit"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    ap.add_argument("--split", default=None, help="dev | test (stages that support it)")
    ap.add_argument("--frac", type=float, default=None, help="benchmark share of queries (block stage)")
    ap.add_argument("--tag", default="", help="label recorded in the run log's data_slice (ablations)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    log = get_logger()
    log.info("config %s flags %s", cfg["_hash"], json.dumps(cfg["flags"]))
    if args.stage == "all":
        use_b = cfg.get("stack_b", False)
        use_ce = use_b and (cfg.get("cross_encoder") or {}).get("enabled", False)
        model = ["block", "features", "stack"] + (["ce"] if use_ce else []) + (["stackb"] if use_b else []) + ["decide"]
        plan = [("prep", None)] + ([("translit", None)] if cfg["normalize"].get("norm_dir") == "norm_tl" else [])
        plan += [(s, "dev") for s in model] + [(s, "test") for s in model] + [("submit", "test")]
    else:
        plan = [(args.stage, args.split)]
    for stage, split in plan:
        run_id = new_run_id(stage)
        mod = importlib.import_module(STAGES[stage])
        kwargs = {"split": split} if split else {}
        if args.frac:
            kwargs["frac"] = args.frac
        res = mod.run(cfg, **kwargs) or {}
        res.setdefault("stage", stage)
        slice_ = (split or "all") + (f":{args.tag}" if args.tag else "")
        append_run_log(cfg, {"run_id": run_id, "data_slice": slice_, **res})
        log.info("stage %s done: %s", stage, {k: v for k, v in res.items() if k != "notes"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
