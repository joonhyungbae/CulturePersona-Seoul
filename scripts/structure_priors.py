"""Structure priors (T) from 2020∪2022 → apply/ablate on eras.

Run: uv run python scripts/structure_priors.py
Outputs: data/grounding/structure_priors.json
      data/grounding/era_*/sampler_specs_margins.json
      data/grounding/era_*/sampler_specs_structure_T.json (R5 + optional)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.chain import (
    ChainSampler,
    build_spec,
    chow_liu_genre_tree,
    era_core_chain,
)
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.grounding.tables import build_root_population
from sfac_persona.schema import GENRE_ANCHORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("structure_T")

PRIOR_WAVES = (2020, 2022)
TARGET_WAVES = (2024, 2022, 2020)
ALL_SPEC_WAVES = (2024, 2022, 2020, 2018, 2016)


def _pool_frames(waves: tuple[int, ...]) -> pd.DataFrame:
    parts = []
    for w in waves:
        fr = build_official_anchor_frame(w)
        # reweight each wave to equal total weight mass
        fr = fr.copy()
        fr["w"] = fr["w"] * (1.0 / fr["w"].sum())
        parts.append(fr)
    return pd.concat(parts, ignore_index=True)


def build_structure_priors() -> dict:
    pooled = _pool_frames(PRIOR_WAVES)
    tree = chow_liu_genre_tree(pooled)
    # genre attendance rank (among attenders) — structure only
    sub = pooled[pooled["attend_any"] == "예"]
    ranks = []
    for g in GENRE_ANCHORS:
        rate = float(np.average((sub[g] == "예").astype(float), weights=sub["w"]))
        ranks.append({"genre": g, "attend_rate_among_goers": rate})
    ranks = sorted(ranks, key=lambda r: -r["attend_rate_among_goers"])
    for i, r in enumerate(ranks):
        r["rank"] = i + 1
    priors = {
        "built_at": datetime.now(UTC).isoformat(),
        "adr": "ADR-007",
        "source_waves": list(PRIOR_WAVES),
        "genre_tree": [{"genre": g, "parent": p} for g, p in tree],
        "genre_rank_among_goers": ranks,
        "chain": "era_core_chain",
        "note": "수준(주변)은 이전하지 않는다. 장르 트리·순위만.",
    }
    path = settings.grounding_dir / "structure_priors.json"
    settings.grounding_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(priors, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s", path)
    return priors


def _ensure_r0() -> Path:
    r0_path = settings.grounding_dir / "R0_population_root.parquet"
    if not r0_path.exists():
        build_root_population().to_parquet(r0_path, index=False)
    return r0_path


def _r0_for_wave(wave: int) -> Path:
    """Use the per-wave R0 if it exists, otherwise the shared R0 (ADR-011)."""
    per_wave = settings.era_grounding_dir(wave) / "R0_population_root.parquet"
    if per_wave.exists():
        return per_wave
    log.warning(
        "wave %s: no per-wave R0 — using the shared R0 (the population mix is not that year's)",
        wave,
    )
    return _ensure_r0()


def _inject_tree(spec: dict, priors: dict) -> dict:
    """Replace genre_tree nodes' parent edges with prior tree; keep target-wave conditionals."""
    # rebuild genre node parent lists: attend_any, age_band, (+tree parent)
    new_nodes = []
    genre_names = set(GENRE_ANCHORS)
    for node in spec["nodes"]:
        name = node.get("name")
        if name not in genre_names:
            new_nodes.append(node)
            continue
        # keep probability tables but rewrite declared parents order for documentation;
        # sampling still uses levels baked in tables — so we must rebuild genre conditionals
        new_nodes.append(node)
    # Actually inject by rebuilding genre conditionals from target frame with prior parents
    return spec


def write_era_specs(priors: dict) -> None:
    for wave in ALL_SPEC_WAVES:
        r0_path = _r0_for_wave(wave)  # ADR-011: per-wave population root
        frame = build_official_anchor_frame(wave)
        frame = frame.dropna(subset=["sex", "age_band", "living_area"]).reset_index(drop=True)
        era_dir = settings.era_grounding_dir(wave)
        era_dir.mkdir(parents=True, exist_ok=True)

        # margins_only: era_core on this wave
        spec_m = build_spec(
            frame, root_source="population", nodes=era_core_chain(), r0_path=r0_path
        )
        spec_m["ablation"] = "margins_only"
        spec_m["wave"] = wave
        (era_dir / "sampler_specs_margins.json").write_text(
            json.dumps(spec_m, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        if wave not in TARGET_WAVES:
            log.info("era_%s margins spec only (unweighted)", wave)
            continue

        # structure_T: genre tree parents from priors, tables from target wave
        from sfac_persona.grounding.chain import add_derived_columns, conditional_table

        frame2 = add_derived_columns(frame)
        spec_t = build_spec(
            frame, root_source="population", nodes=era_core_chain(), r0_path=r0_path
        )
        floor = settings.cell_floor
        genre_nodes = []
        for e in priors["genre_tree"]:
            g, p = e["genre"], e["parent"]
            parents = ["attend_any", "age_band"] + ([p] if p else [])
            genre_nodes.append(conditional_table(frame2, g, parents, floor))
        kept = [n for n in spec_t["nodes"] if n.get("name") not in set(GENRE_ANCHORS)]
        out_nodes = []
        inserted = False
        for n in kept:
            out_nodes.append(n)
            if n.get("name") == "n_genres_band":
                out_nodes.extend(genre_nodes)
                inserted = True
        if not inserted:
            out_nodes.extend(genre_nodes)
        spec_t["nodes"] = out_nodes
        spec_t["genre_tree"] = priors["genre_tree"]
        spec_t["ablation"] = "margins_plus_structure_T"
        spec_t["structure_source_waves"] = list(PRIOR_WAVES)
        spec_t["wave"] = wave
        (era_dir / "sampler_specs_structure_T.json").write_text(
            json.dumps(spec_t, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        log.info("era_%s specs written (margins + T)", wave)

        for tag, spec in [("margins", spec_m), ("structure_T", spec_t)]:
            samp = ChainSampler(spec).sample(2000, seed=42 + wave)
            samp["w"] = 1.0
            from sfac_persona.grounding.error_study import genre_phi

            for g in GENRE_ANCHORS:
                if g not in samp.columns:
                    samp[g] = "아니오"
            phi_s = genre_phi(samp)
            phi_t = genre_phi(frame)
            diff = np.abs(phi_s.values - phi_t.values)
            mad = float(diff[~np.eye(len(diff), dtype=bool)].mean())
            log.info("wave %s %s mean|Δphi|=%.4f", wave, tag, mad)


def main() -> int:
    priors = build_structure_priors()
    write_era_specs(priors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
