"""Whether cells < 30 leak onto the public surface (git tracking · release package).

The 2026-09-11 audit caught two places.
  1. The old version tracked internal tables `T1`–`T5` as-is — cells < 30 were
     present with only a `suppressed` flag.
  2. `era_package_release.copy_era_grounding` copied the whole `era_{wave}`
     directory, so internal `T6_pair_joints.parquet` went out to HF/Zenodo.
Both violate the release rule (public grounding tables: unweighted cell ≥ 30).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 30


def _tracked(pattern: str) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", pattern], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return [ROOT / p for p in out.stdout.split() if p]


def test_tracked_grounding_has_no_subfloor_cells() -> None:
    files = _tracked("data/grounding/**/*.parquet")
    if not files:
        pytest.skip("no tracked grounding tables")
    offenders = []
    for f in files:
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        if "n_unweighted" in df.columns and (df["n_unweighted"] < FLOOR).any():
            offenders.append(
                (f.relative_to(ROOT).as_posix(), int((df["n_unweighted"] < FLOOR).sum()))
            )
    assert not offenders, f"tracked grounding tables have cells <{FLOOR}: {offenders}"


def test_tracked_grounding_is_release_only() -> None:
    """Internal tables (T tables whose names do not end in `*_release`) are not tracked."""
    bad = [
        f.relative_to(ROOT).as_posix()
        for f in _tracked("data/grounding/**/*.parquet")
        if f.name.startswith("T") and not f.stem.endswith("_release")
    ]
    assert not bad, f"internal tables are being tracked: {bad}"


def test_no_individual_level_artifacts_tracked() -> None:
    """Distilled predictions are one row per respondent — never track them."""
    bad = [p for p in _tracked("*") if p.suffix in {".npz", ".pkl", ".sav", ".dta"}]
    assert not bad, f"files that may be individual-level are tracked: {[str(b) for b in bad]}"


def test_packager_copies_only_public_tables() -> None:
    """Check the source that copy_era_grounding keeps only public tables (no run)."""
    src = (ROOT / "scripts" / "era_package_release.py").read_text(encoding="utf-8")
    assert "_release" in src.split("def copy_era_grounding")[1].split("def ")[0], (
        "copy_era_grounding does not select the public tables"
    )
    assert "copytree(\n            src," not in src, "copying a directory wholesale"
