"""Record grounding tables by era (ADR-007 E).

R0 is shared under data/grounding/. Per-wave tables live under data/grounding/era_{wave}/.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sfac_persona.config import settings
from sfac_persona.grounding.official import build_official_anchor_frame, reproduce_gate
from sfac_persona.grounding.tables import build_pair_joints, build_root_population
from sfac_persona.schema import (
    ERA_CORE_ANCHORS,
    ERA_WAVES,
    ROOT_ANCHORS,
    UNWEIGHTED_WAVES,
    WAVE_REGIME,
)

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_era_grounding(wave: int, *, require_gate: bool = True) -> dict[str, Path]:
    gate = reproduce_gate(wave)
    if require_gate and wave not in UNWEIGHTED_WAVES:
        bad = gate[~gate["ok"]]
        if len(bad):
            raise SystemExit(f"era {wave} gate FAIL:\n{bad}")

    frame = build_official_anchor_frame(wave)
    before = len(frame)
    frame = frame.dropna(subset=ROOT_ANCHORS).reset_index(drop=True)
    if len(frame) < before:
        logger.warning(
            "wave %s dropped %d rows with missing root anchors", wave, before - len(frame)
        )
    # tables helpers expect ANCHORS — restrict to era core present in frame
    anchors = [a for a in ERA_CORE_ANCHORS if a in frame.columns]
    missing_root = [a for a in ROOT_ANCHORS if a not in frame.columns]
    if missing_root:
        raise ValueError(f"missing root anchors: {missing_root}")

    settings.grounding_dir.mkdir(parents=True, exist_ok=True)
    r0_path = settings.grounding_dir / "R0_population_root.parquet"
    if not r0_path.exists():
        r0 = build_root_population(wave=wave)
        r0.to_parquet(r0_path, index=False)
    else:
        r0 = pd.read_parquet(r0_path)

    era_dir = settings.era_grounding_dir(wave)
    era_dir.mkdir(parents=True, exist_ok=True)

    # Per-wave R0 (resolves ADR-011). Use that year's resident-registration month when
    # the population CSV has it. Otherwise fall back to the shared R0; the warning
    # below and provenance record that fact.
    wave_r0 = era_dir / "R0_population_root.parquet"
    try:
        r0_w = build_root_population(wave=wave)
        if bool(r0_w["wave_matched"].iloc[0]):
            r0_w.to_parquet(wave_r0, index=False)
            r0, r0_path = r0_w, wave_r0
            logger.info(
                "wave %s: using per-wave R0 (%s, population %d)",
                wave,
                r0_w.reference_month.iloc[0],
                int(r0_w.population.sum()),
            )
    except Exception as exc:  # noqa: BLE001 — no population source, so use the shared R0
        logger.warning("wave %s: per-wave R0 failed (%s) — shared R0", wave, exc)

    # If the wave's population could not be obtained, say so loudly here (ADR-011).
    r0_month = str(r0.reference_month.iloc[0])
    if not r0_month.startswith(str(wave)):
        logger.warning(
            "wave %s: R0 population reference month is %s — the era bank's "
            "sex × age × living-area mix is not that year. Report it as a limitation (ADR-011).",
            wave,
            r0_month,
        )

    paths: dict[str, Path] = {"R0": r0_path}

    # monkey-patch style: build_survey_tables uses ANCHORS — call shares manually via subset frame

    # Temporarily build using only columns present: override by filtering ANCHORS usage
    # Use build_survey_tables on a frame that has all needed cols; patch ANCHORS via local rebuild
    tables = _build_survey_tables_core(frame, anchors)
    for name, df in tables.items():
        p = era_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        paths[name] = p

    non_root = [a for a in anchors if a not in ROOT_ANCHORS]
    pairs = build_pair_joints(frame, anchors=non_root)
    paths["T6"] = era_dir / "T6_pair_joints.parquet"
    pairs.to_parquet(paths["T6"], index=False)
    pub = pairs[~pairs.suppressed]
    paths["T6_release"] = era_dir / "T6_pair_joints_release.parquet"
    pub.to_parquet(paths["T6_release"], index=False)

    gate_path = era_dir / "reproduce_gate.csv"
    gate.to_csv(gate_path, index=False)
    paths["gate"] = gate_path

    official_files = sorted(settings.official_person_dir.glob("official_person_*.parquet"))
    prov = {
        "built_at": datetime.now(UTC).isoformat(),
        "adr": "ADR-007",
        "wave": wave,
        "survey_era": WAVE_REGIME.get(wave),
        "weighting": "none" if wave in UNWEIGHTED_WAVES else "survey",
        "n_respondents": len(frame),
        "weight_sum": float(frame.w.sum()),
        "cell_floor": settings.cell_floor,
        "geo_max_level": settings.geo_max_level,
        "anchors": anchors,
        "official_sources": [{"file": f.name, "sha256": _sha256(f)} for f in official_files],
        "population_root": {
            "path": str(r0_path.name),
            "reference_month": r0_month,
            "population_total": int(r0.population.sum()),
            "wave_matched": r0_month.startswith(str(wave)),
            "note": (
                "Resident-registration population of the wave's December (ADR-011)."
                if r0_month.startswith(str(wave))
                else "No registration month for this wave; the shared latest month is used (ADR-011)."
            ),
        },
        "policy": (
            "개별 응답 기록은 리샘플하지 않는다. 시대 간 관람률 수준을 맞추지 않는다. "
            "공개본은 비가중 셀 < cell_floor 제외. 지역은 생활권까지."
        ),
        "gate_summary": {
            "n": len(gate),
            "n_fail": int((~gate["ok"]).sum()) if "ok" in gate.columns else None,
            "max_abs_diff": float(gate["diff_pp"].abs().max())
            if gate["diff_pp"].notna().any()
            else None,
        },
    }
    for rel_path in write_release_variants(wave):
        paths[rel_path.stem] = rel_path

    paths["provenance"] = era_dir / "provenance.json"
    paths["provenance"].write_text(json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("era_%s grounding written (%d files)", wave, len(paths))
    return paths


def _build_survey_tables_core(frame: pd.DataFrame, anchors: list[str]) -> dict[str, pd.DataFrame]:
    from sfac_persona.grounding.tables import _shares

    tables: dict[str, list[pd.DataFrame]] = {"T1": [], "T2": [], "T3": [], "T4": [], "T5": []}
    for a in anchors:
        tables["T1"].append(_shares(frame, a, []))
        if a in ROOT_ANCHORS:
            continue
        tables["T2"].append(_shares(frame, a, ["age_band"]))
        tables["T3"].append(_shares(frame, a, ["sex"]))
        tables["T4"].append(_shares(frame, a, ["living_area"]))
        tables["T5"].append(_shares(frame, a, ["age_band", "sex"]))
    out = {k: pd.concat(v, ignore_index=True) for k, v in tables.items() if v}
    out["T1_root_joint"] = _shares(
        frame.assign(root=frame[ROOT_ANCHORS].astype(str).agg("|".join, axis=1)), "root", []
    )
    return out


def write_all_eras(*, waves: tuple[int, ...] = ERA_WAVES) -> dict[int, dict[str, Path]]:
    return {w: write_era_grounding(w) for w in waves}


RELEASE_SUFFIX = "_release"


def write_release_variants(wave: int) -> list[Path]:
    """Public grounding tables: a copy with unweighted cells < cell_floor **removed**.

    2026-09-11: only T6 had a public copy; T1–T5 and T1_root_joint existed only as the
    internal copy. That internal copy still held cells < 30, flagged `suppressed=True`,
    and `era_package_release` copied the directory wholesale into the release —
    a violation of the release rule (public grounding tables have unweighted cells ≥ 30).
    The public copy is written separately.
    """
    from sfac_persona.grounding.sdc import secondary_suppress_pairs

    era_dir = settings.era_grounding_dir(wave)
    written: list[Path] = []
    internal: dict[str, pd.DataFrame] = {}
    for src in sorted(era_dir.glob("T*.parquet")):
        if src.stem.endswith(RELEASE_SUFFIX):
            continue
        df = pd.read_parquet(src)
        if "suppressed" not in df.columns:
            logger.warning("%s: no suppressed column — public copy not written", src.name)
            continue
        df = df.drop(columns=["secondary"], errors="ignore")
        df["secondary"] = False
        internal[src.stem] = df

    # 2026-09-23 (ADR-024): primary suppression
    # alone lets suppressed cells come back from row and column sums (2024 T6: 148/410
    # restored exactly). Secondary suppression is applied only to the pair joint T6 —
    # the only constraints are row and column sums against the marginals, so the cost
    # is small (2024: +136 cells). Linked conditional tables T1–T5 are not suppressed
    # this way: aggregation identities cascade and delete more than half the cells
    # (2024 T2 529→192). The floor of 30 is a **reliability rule**; the disclosure
    # argument is the sampling rate, unpublished person records, and living-area
    # geography. `scripts/era_sdc_differencing.py` reports how many cells differencing
    # restores.
    if "T6_pair_joints" in internal and "T1" in internal:
        t1_pub = internal["T1"][~internal["T1"].suppressed.astype(bool)]
        internal["T6_pair_joints"] = secondary_suppress_pairs(internal["T6_pair_joints"], t1_pub)

    for stem, df in internal.items():
        df.to_parquet(
            era_dir / f"{stem}.parquet", index=False
        )  # keep secondary on the internal copy
        hide = df["suppressed"].astype(bool) | df["secondary"].astype(bool)
        pub = df[~hide].drop(columns=["suppressed", "secondary"])
        out = era_dir / f"{stem}{RELEASE_SUFFIX}.parquet"
        pub.to_parquet(out, index=False)
        written.append(out)
        logger.info(
            "wave %s %s: %d → %d rows (cells<%d removed %d · secondary suppression %d)",
            wave,
            stem,
            len(df),
            len(pub),
            settings.cell_floor,
            int(df["suppressed"].astype(bool).sum()),
            int(df["secondary"].astype(bool).sum()),
        )
    return written
