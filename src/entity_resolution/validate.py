"""Local re-implementation of the submission rules (plus a runner for the official validator).

Rules checked for both ``matching_results.tsv`` and ``candidate_pairs.tsv``:
exact header, one row per test Source-1 id (no missing / extra / duplicate rows), ID lists contain
only existing Source-2/3 test ids, no duplicates inside a list. Matches outside the candidate set
are reported as a warning (it signals a pipeline bug).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .io import CANDIDATE_COLUMN, MATCH_COLUMN


def _read_first_column(path: Path) -> set[str]:
    """Entity ids of a source TSV without loading names/addresses (the test files are ~0.5 GB each)."""
    with open(path, encoding="utf-8-sig") as fh:
        next(fh, None)
        return {line.split("\t", 1)[0].strip() for line in fh if line.strip()}


def _read_lists(path: Path, column: str, errors: list[str]) -> dict[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        errors.append(f"{path.name}: empty file")
        return {}
    header = lines[0].split("\t")
    if header != ["source1_entity_id", column]:
        errors.append(f"{path.name}: header must be 'source1_entity_id<TAB>{column}', got {header}")
    rows: dict[str, list[str]] = {}
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) > 2:
            errors.append(f"{path.name}:{n}: more than 2 tab-separated fields")
            continue
        s1 = parts[0].strip()
        ids = [x.strip() for x in parts[1].split(",")] if len(parts) == 2 and parts[1].strip() else []
        if s1 in rows:
            errors.append(f"{path.name}:{n}: duplicate row for {s1}")
        if len(ids) != len(set(ids)):
            errors.append(f"{path.name}:{n}: duplicate ids in list for {s1}")
        if any(not x for x in ids):
            errors.append(f"{path.name}:{n}: empty id inside list for {s1}")
        rows[s1] = ids
    return rows


def validate_outputs(matching: str | Path, candidates: str | Path, test_dir: str | Path) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)``; an empty error list means the files follow every rule."""
    test_dir = Path(test_dir)
    split = test_dir.name
    s1_ids = _read_first_column(test_dir / f"{split}_source1.tsv")
    target_ids = _read_first_column(test_dir / f"{split}_source2.tsv") | _read_first_column(
        test_dir / f"{split}_source3.tsv")
    errors: list[str] = []
    warnings: list[str] = []
    parsed = {}
    for path, column in ((Path(matching), MATCH_COLUMN), (Path(candidates), CANDIDATE_COLUMN)):
        if not path.exists():
            errors.append(f"missing file {path}")
            continue
        rows = _read_lists(path, column, errors)
        missing, extra = s1_ids - set(rows), set(rows) - s1_ids
        if missing:
            errors.append(f"{path.name}: {len(missing)} Source-1 ids missing (e.g. {sorted(missing)[:3]})")
        if extra:
            errors.append(f"{path.name}: {len(extra)} unknown Source-1 ids (e.g. {sorted(extra)[:3]})")
        bad = {x for ids in rows.values() for x in ids if x not in target_ids}
        if bad:
            errors.append(f"{path.name}: {len(bad)} ids are not Source-2/3 test ids (e.g. {sorted(bad)[:3]})")
        parsed[column] = rows
    if MATCH_COLUMN in parsed and CANDIDATE_COLUMN in parsed:
        cand = parsed[CANDIDATE_COLUMN]
        outside = sum(1 for s1, ids in parsed[MATCH_COLUMN].items() for x in ids if x not in set(cand.get(s1, [])))
        if outside:
            warnings.append(f"{outside} matched ids are not in candidate_pairs.tsv (pipeline bug?)")
    return errors, warnings


def run_official_validator(script: str | Path, matching: str | Path, candidates: str | Path,
                           test_dir: str | Path) -> int:
    """Run the organisers' ``utils/validate_submission.py`` (stdlib only) and return its exit code."""
    cmd = [sys.executable, str(script), "--matching", str(matching), "--candidate", str(candidates),
           "--test-dir", str(test_dir)]
    return subprocess.run(cmd, check=False).returncode
