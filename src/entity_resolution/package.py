"""Build ``<team>_submission.zip`` in the exact layout the organisers require.

    <team>_submission.zip
    ├── output/{matching_results.tsv, candidate_pairs.tsv}
    ├── code/business_entity_resolution/{src/, README.md, requirements.txt, ...}
    └── Documentation_template.md
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

CODE_ITEMS = ["src", "configs", "tests", "main.py", "README.md", "requirements.txt", "requirements-embeddings.txt",
              "pyproject.toml"]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def build_package(team: str, repo_root: str | Path = ".", output_dir: str | Path = "output",
                  doc: str | Path = "docs/METHODOLOGY.md", dest: str | Path | None = None) -> Path:
    """Zip outputs + runnable code + methodology document; returns the zip path."""
    root = Path(repo_root).resolve()
    out = Path(output_dir)
    dest = Path(dest) if dest else root / f"{team}_submission.zip"
    required = [out / "matching_results.tsv", out / "candidate_pairs.tsv", Path(doc)]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"cannot package, missing: {missing}")

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in required[:2]:
            zf.write(p, f"output/{p.name}")
        for item in CODE_ITEMS:
            path = root / item
            if not path.exists():
                continue
            files = [path] if path.is_file() else [f for f in path.rglob("*") if f.is_file()]
            for f in files:
                rel = f.relative_to(root)
                if SKIP_PARTS & set(rel.parts) or f.suffix in {".pyc", ".joblib"}:
                    continue
                zf.write(f, f"code/business_entity_resolution/{rel.as_posix()}")
        zf.write(doc, "Documentation_template.md")
    log.info("wrote %s", dest)
    return dest
