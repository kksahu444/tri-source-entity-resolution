"""Command-line interface.

    python main.py eda                 # dataset statistics (singletons, exclusivity, countries, ...)
    python main.py block               # blocking recall ceiling / reduction ratio on train
    python main.py train               # CV training + decision tuning -> artifacts/
    python main.py predict             # test predictions -> output/*.tsv (+ local validation)
    python main.py loco                # leave-one-country-out generalisation check
    python main.py validate            # local rule check (+ official validator if --official)
    python main.py package --team X    # X_submission.zip in the required layout
    python main.py all                 # train + predict
    python main.py score --truth T     # macro F0.5 of output/matching_results.tsv against labels T
    python main.py synth               # synthetic data for smoke tests
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import Config


def _config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config) if args.config else Config()
    for key in ("data_dir", "artifacts_dir", "output_dir"):
        value = getattr(args, key, None)
        if value:
            setattr(cfg, key, value)
    if getattr(args, "folds", None):
        cfg.model.n_folds = args.folds
    if getattr(args, "no_stack", False):
        cfg.model.stack = False
    return cfg


def read_tsv_lists(path: Path) -> dict[str, list[str]]:
    """Read a submission-format TSV into ``{s1_id: [ids]}``."""
    from .io import parse_id_list, read_tsv
    df = read_tsv(path)
    return {row[0]: parse_id_list(row[1]) for row in df.itertuples(index=False)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="entity_resolution", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="JSON config override (see configs/default.json)")
    parser.add_argument("--data-dir", help="folder containing train/ and test/ (default: dataset)")
    parser.add_argument("--artifacts-dir", help="model + reports folder (default: artifacts)")
    parser.add_argument("--output-dir", help="submission files folder (default: output)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("eda")
    p.add_argument("--split", default="train")
    p = sub.add_parser("block")
    p.add_argument("--split", default="train")
    for name in ("train", "all", "loco"):
        p = sub.add_parser(name)
        p.add_argument("--folds", type=int)
        p.add_argument("--no-stack", action="store_true", help="skip the stage-2 model")
    p = sub.add_parser("predict")
    p.add_argument("--split", default="test")
    p = sub.add_parser("validate")
    p.add_argument("--official", help="path to the organisers' utils/validate_submission.py")
    p = sub.add_parser("package")
    p.add_argument("--team", required=True)
    p.add_argument("--doc", default="docs/METHODOLOGY.md", help="filled-in methodology markdown")
    p = sub.add_parser("score")
    p.add_argument("--truth", required=True, help="ground-truth TSV (source1_entity_id, matched_entity_ids)")
    p.add_argument("--pred", help="predictions TSV (default: <output-dir>/matching_results.tsv)")
    p = sub.add_parser("synth")
    p.add_argument("--out", default="data_synthetic")
    p.add_argument("--n-train", type=int, default=1500)
    p.add_argument("--n-test", type=int, default=900)
    p.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s", datefmt="%H:%M:%S")
    cfg = _config(args)

    # imports are local so `synth` / `validate` stay fast and light
    if args.command == "eda":
        from .eda import run_eda
        run_eda(cfg.data_dir, args.split, cfg.artifacts_dir)
    elif args.command == "block":
        from .pipeline import block_report
        print(json.dumps(block_report(cfg, args.split), indent=2))
    elif args.command == "train":
        from .pipeline import train
        train(cfg)
    elif args.command == "predict":
        from .pipeline import predict
        summary = predict(cfg, args.split)
        return 1 if summary.get("validation_errors") else 0
    elif args.command == "all":
        from .pipeline import predict, train
        train(cfg)
        summary = predict(cfg, "test")
        return 1 if summary.get("validation_errors") else 0
    elif args.command == "loco":
        from .pipeline import leave_one_country_out
        print(json.dumps(leave_one_country_out(cfg), indent=2))
    elif args.command == "validate":
        from .validate import run_official_validator, validate_outputs
        out, test_dir = Path(cfg.output_dir), Path(cfg.data_dir) / "test"
        errors, warnings = validate_outputs(out / "matching_results.tsv", out / "candidate_pairs.tsv", test_dir)
        for msg in warnings:
            print("WARNING:", msg)
        for msg in errors:
            print("ERROR:", msg)
        print("local validation:", "PASS" if not errors else "FAIL")
        code = 1 if errors else 0
        if args.official:
            code = max(code, run_official_validator(args.official, out / "matching_results.tsv",
                                                    out / "candidate_pairs.tsv", test_dir))
        return code
    elif args.command == "package":
        from .package import build_package
        print(build_package(args.team, ".", cfg.output_dir, args.doc))
    elif args.command == "score":
        from .io import read_ground_truth
        from .metrics import macro_f_beta
        pred_path = Path(args.pred) if args.pred else Path(cfg.output_dir) / "matching_results.tsv"
        pred = read_tsv_lists(pred_path)
        truth = read_ground_truth(args.truth)
        print(f"macro F0.5 = {macro_f_beta(pred, truth, truth.keys()):.5f} over {len(truth)} Source-1 entities")
    elif args.command == "synth":
        from .synthetic import generate
        print(generate(args.out, args.n_train, args.n_test, args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
