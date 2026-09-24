"""Download the challenge dataset from this repository's GitHub Release.

The raw TSVs (127-510 MB each) exceed GitHub's 100 MB per-file git limit, so they are stored
gzip-compressed as assets of the release ``dataset-v1`` in this private repository. Only
collaborators can download them. The GitHub CLI handles authentication:

    gh auth login                       # once
    python main.py fetch-data           # -> dataset/train/*.tsv, dataset/test/*.tsv
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

REPO = "kksahu444/tri-source-entity-resolution"
RELEASE_TAG = "dataset-v1"
CHECKSUMS = "SHA256SUMS.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(dest: str | Path = "dataset", repo: str = REPO, tag: str = RELEASE_TAG, keep_archives: bool = False) -> Path:
    """Download, verify (SHA-256) and decompress every ``*.tsv.gz`` asset into ``dest/<split>/``."""
    dest = Path(dest)
    download_dir = dest / "_download"
    download_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["gh", "release", "download", tag, "-R", repo, "-D", str(download_dir), "--clobber"]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("GitHub CLI not found: install it from https://cli.github.com and run `gh auth login`") from exc

    expected = {}
    sums = download_dir / CHECKSUMS
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                expected[name.strip().lstrip("*")] = digest

    for archive in sorted(download_dir.glob("*.tsv.gz")):
        if archive.name in expected and _sha256(archive) != expected[archive.name]:
            raise SystemExit(f"checksum mismatch for {archive.name}; delete {download_dir} and retry")
        split = archive.name.split("_", 1)[0]  # "train_source1.tsv.gz" -> "train"
        target = dest / split / archive.name[: -len(".gz")]
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(archive, "rb") as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out, length=1 << 24)
        log.info("wrote %s (%.0f MB)", target, target.stat().st_size / 1e6)
    if not keep_archives:
        shutil.rmtree(download_dir)
    return dest
