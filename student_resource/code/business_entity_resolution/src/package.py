"""Build <team>_submission.zip in the layout the problem statement requires (spec section 11).

python -m src.package --team <TEAM> [--doc path/to/filled_Documentation_template.md]
"""
from __future__ import annotations

import argparse
import os
import zipfile

from .utils import ROOT, load_config

CODE_ITEMS = ["src", "tests", "kaggle", "README.md", "requirements.txt", "requirements-gpu.txt", "LICENSES.md"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--doc", default=None, help="filled Documentation_template.md")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    sr = os.path.dirname(os.path.dirname(ROOT))                       # student_resource/
    doc = args.doc or os.path.join(sr, "Documentation_template.md")
    out = args.out or os.path.join(sr, f"{args.team}_submission.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("matching_results.tsv", "candidate_pairs.tsv"):
            p = os.path.join(cfg["paths"]["output"], name)
            assert os.path.exists(p), f"missing {p}"
            z.write(p, f"output/{name}")
        for item in CODE_ITEMS + sorted(f for f in os.listdir(ROOT) if f.startswith("config") and f.endswith(".yaml")
                                        and f != "config_kaggle.yaml"):
            src = os.path.join(ROOT, item)
            if os.path.isfile(src):
                z.write(src, f"code/business_entity_resolution/{item}")
            elif os.path.isdir(src):
                for dp, dns, fns in os.walk(src):
                    dns[:] = [d for d in dns if d not in ("__pycache__", "out", ".pytest_cache")]
                    for fn in fns:
                        full = os.path.join(dp, fn)
                        rel = os.path.relpath(full, ROOT).replace(os.sep, "/")
                        z.write(full, f"code/business_entity_resolution/{rel}")
        z.write(doc, "Documentation_template.md")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    print(out, len(names), "files")
    for n in sorted(names):
        if n.count("/") <= 2:
            print(" ", n)


if __name__ == "__main__":
    main()
