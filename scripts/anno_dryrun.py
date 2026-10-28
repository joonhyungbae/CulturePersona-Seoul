"""Dry run of the human-audit analysis. Drives `anno_analysis.py` to completion on fake scores.

Real scores (`outputs/anno/annotations/`, `adjudication.jsonl`),
`outputs/reports/anno_results.*`, and `sfac_persona_paper/tables/` are only read, or not
touched at all. LLM decisions are copied from the originals into a temporary store. Output
goes only under `outputs/logs/anno_dryrun/` (gitignore).

One score line matches the nested JSON saved by Apps Script `annotations_()` and the local
`annoPlugin`. Analysis uses the last line per item (`anno_analysis.latest`).

Run:
  uv run python scripts/anno_dryrun.py
  uv run python scripts/anno_dryrun.py --scenario good
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from anno_analysis import P1_BOUND, SCALES, consensus, cp_upper, load
from human_audit_sample_size import kripp_alpha_ordinal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("anno_dryrun")

SEED = 20260926
AGES = ("10대", "20대", "30대", "40대", "50대", "60대", "70대 이상")
ATTEND = ("예", "아니오")
RATERS = ("R1", "R2")
CONTRA_FIELD = "summary"
REAL_ANNO = ROOT / "outputs" / "anno"
REAL_REPORTS = ROOT / "outputs" / "reports"
REAL_TABLES = ROOT / "sfac_persona_paper" / "tables"
LLM_NAMES = ("llm_judge.jsonl", "llm_bank.jsonl")
WATCH = (
    REAL_ANNO / "annotations" / "R1.jsonl",
    REAL_ANNO / "annotations" / "R2.jsonl",
    REAL_ANNO / "adjudication.jsonl",
    REAL_ANNO / "llm_judge.jsonl",
    REAL_ANNO / "llm_bank.jsonl",
    REAL_REPORTS / "anno_results.csv",
    REAL_REPORTS / "anno_results.md",
    REAL_TABLES / "audit.tex",
)
# Keys that load() reads, plus keys export always carries. Sheet columns area_guess and practice
# are absent from export.
ANN_KEYS = frozenset({"rater", "item_id", "stage1", "stage2", "saved_at"})
S1_KEYS = frozenset({"age_guess", "attend_guess", "secs"})
S2_KEYS = frozenset(
    {
        "contradiction",
        "contra_fields",
        "contra_quote",
        "grounded",
        "coherent",
        "plausible",
        "fluent",
        "note",
        "secs",
    }
)
CONS_KEYS = S2_KEYS - {"secs"}
ADJ_KEYS = frozenset({"item_id", "consensus", "changed_by", "reason", "first_speaker", "saved_at"})
AUDIT_ORDER = ("P1", "P2", "S1a", "S1b", "S2a", "S2b", "S4a", "S4b", "S5a", "S5b")
AUDIT_MACROS = (
    "auditPOneBound",
    "auditPTwoBound",
    "auditSOneNull",
    "auditSFiveMargin",
    "auditSFiveSpec",
)


@dataclass(frozen=True)
class Scenario:
    """Planted truth for fake scores. clean_rate and trap_sens are planted rates; the consensus
    rule matches the analysis.
    """

    name: str
    clean_rate: float
    trap_sens: float
    gross_sens: float
    scale_sd: float
    flip: float
    age_hit: float
    attend_hit: float
    incomplete: bool
    resave: bool
    force_gap: bool
    strict_none: int


SCENARIOS: dict[str, Scenario] = {
    "good": Scenario("good", 0.03, 0.80, 1.0, 0.05, 0.01, 0.72, 0.85, False, False, False, 0),
    "poor": Scenario("poor", 0.15, 0.40, 0.50, 0.90, 0.12, 0.20, 0.55, False, False, False, 0),
    "messy": Scenario("messy", 0.03, 0.80, 1.0, 0.05, 0.08, 0.72, 0.85, True, True, True, 2),
}


@dataclass
class Bundle:
    annotations: dict[str, list[dict[str, Any]]]
    adjudication: list[dict[str, Any]]
    truth: dict[str, Any]


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def assert_sandbox(path: Path) -> None:
    """Stop if an output path lies inside the real audit store or the manuscript tables."""
    resolved = path.resolve()
    for banned in (REAL_ANNO, REAL_REPORTS, REAL_TABLES):
        if _inside(resolved, banned):
            raise SystemExit(f"outside the sandbox: {resolved} ({banned})")


def snapshot(paths: tuple[Path, ...]) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for p in paths:
        out[str(p)] = p.stat().st_mtime_ns if p.exists() else None
    return out


def clip_score(x: float) -> int:
    return int(min(5, max(1, round(x))))


def pick(ids: list[str], k: int, rng: np.random.Generator) -> set[str]:
    if k <= 0 or not ids:
        return set()
    k = min(k, len(ids))
    chosen = rng.choice(len(ids), size=k, replace=False)
    return {ids[int(i)] for i in np.atleast_1d(chosen)}


def guess_cat(true: str, options: tuple[str, ...], hit: float, rng: np.random.Generator) -> str:
    if rng.random() < hit:
        return true
    others = [o for o in options if o != true]
    return str(rng.choice(others))


def annotation_record(
    rater: str,
    item_id: str,
    stage1: dict[str, Any],
    stage2: dict[str, Any],
    saved_at: str,
) -> dict[str, Any]:
    """One line as written by the Apps Script export and the local annotate tool."""
    row = {
        "rater": rater,
        "item_id": item_id,
        "stage1": {
            "age_guess": str(stage1["age_guess"]),
            "attend_guess": str(stage1["attend_guess"]),
            "secs": int(stage1["secs"]),
        },
        "stage2": {
            "contradiction": stage2["contradiction"],
            "contra_fields": [str(x) for x in stage2["contra_fields"]],
            "contra_quote": str(stage2["contra_quote"]),
            "grounded": int(stage2["grounded"]),
            "coherent": int(stage2["coherent"]),
            "plausible": int(stage2["plausible"]),
            "fluent": int(stage2["fluent"]),
            "note": str(stage2.get("note", "")),
            "secs": int(stage2["secs"]),
        },
        "saved_at": saved_at,
    }
    _check_annotation(row)
    return row


def adjudication_record(
    item_id: str,
    consensus_row: dict[str, Any],
    changed_by: str,
    reason: str,
    first_speaker: str,
    saved_at: str,
) -> dict[str, Any]:
    """One consensus line. consensus has no secs (server `normStage2(withSecs=false)`)."""
    row = {
        "item_id": item_id,
        "consensus": {k: consensus_row[k] for k in CONS_KEYS},
        "changed_by": changed_by,
        "reason": reason,
        "first_speaker": first_speaker,
        "saved_at": saved_at,
    }
    if set(row) != ADJ_KEYS:
        raise ValueError(f"adjudication keys differ: {set(row) ^ ADJ_KEYS}")
    if row["changed_by"] not in {"R1", "R2", "both", "none"}:
        raise ValueError(row["changed_by"])
    if row["first_speaker"] not in RATERS:
        raise ValueError(row["first_speaker"])
    if not str(row["reason"]).strip():
        raise ValueError("adjudication reason is empty")
    _check_stage2({**row["consensus"], "secs": 0})
    return row


def _check_annotation(row: dict[str, Any]) -> None:
    if set(row) != ANN_KEYS or set(row["stage1"]) != S1_KEYS or set(row["stage2"]) != S2_KEYS:
        raise ValueError("annotation record keys differ from export")
    if row["rater"] not in RATERS:
        raise ValueError(row["rater"])
    _check_stage2(row["stage2"])


def _check_stage2(s2: dict[str, Any]) -> None:
    if s2["contradiction"] not in {"yes", "no"}:
        raise ValueError(s2["contradiction"])
    if not isinstance(s2["contra_fields"], list):
        raise TypeError("contra_fields")
    yes = s2["contradiction"] == "yes"
    if yes and (not s2["contra_fields"] or not str(s2["contra_quote"]).strip()):
        raise ValueError("contradiction is yes but the evidence is missing")
    if not yes and (s2["contra_fields"] or str(s2["contra_quote"]).strip()):
        raise ValueError("contradiction is no but evidence remains")
    for sc in SCALES:
        v = s2[sc]
        if not isinstance(v, int) or not 1 <= v <= 5:
            raise ValueError(f"{sc}={v}")


def _stage2(yes: bool, scales: dict[str, int], rng: np.random.Generator) -> dict[str, Any]:
    return {
        "contradiction": "yes" if yes else "no",
        "contra_fields": [CONTRA_FIELD] if yes else [],
        "contra_quote": "예행연습: 앵커와 양립하지 않는 문장." if yes else "",
        "note": "",
        "secs": int(rng.integers(20, 150)),
        **scales,
    }


def _draw_scales(yes: bool, sd: float, rng: np.random.Generator) -> dict[str, int]:
    out: dict[str, int] = {}
    for sc in SCALES:
        if sc == "grounded":
            center = 1.8 if yes else 4.8
        else:
            center = 2.2 if yes else 4.5
        out[sc] = clip_score(center + float(rng.normal(0, sd)))
    return out


def _draw_pair(
    yes: bool, sd: float, rng: np.random.Generator
) -> tuple[dict[str, int], dict[str, int]]:
    """The two raters share one latent score. A small sd makes the scales nearly identical."""
    r1: dict[str, int] = {}
    r2: dict[str, int] = {}
    for sc in SCALES:
        if sc == "grounded":
            center = 1.8 if yes else 4.8
        else:
            center = 2.2 if yes else 4.5
        latent = center + float(rng.normal(0, 0.05))
        r1[sc] = clip_score(latent + float(rng.normal(0, sd)))
        r2[sc] = clip_score(latent + float(rng.normal(0, sd)))
    return r1, r2


def _rater_yes(
    stratum: str,
    item_id: str,
    rater: str,
    base_yes: set[str],
    detect: dict[str, set[str]],
    gross_detect: dict[str, set[str]],
    flip: float,
    rng: np.random.Generator,
) -> bool:
    if stratum == "trap":
        return item_id in detect[rater]
    if stratum == "trap_gross":
        return item_id in gross_detect[rater]
    yes = item_id in base_yes
    if rng.random() < flip:
        yes = not yes
    return yes


def _who_changed(r1_yes: bool, r2_yes: bool, cons_yes: bool) -> str:
    m1, m2 = r1_yes == cons_yes, r2_yes == cons_yes
    if m1 and m2:
        return "both"
    if m1 and not m2:
        return "R2"
    if m2 and not m1:
        return "R1"
    return "none"


def _disagrees(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["contradiction"] != b["contradiction"]:
        return True
    return abs(int(a["grounded"]) - int(b["grounded"])) >= 2


def _missing_sets(
    main: pd.DataFrame, scenario: Scenario, rng: np.random.Generator
) -> dict[str, set[str]]:
    if not scenario.incomplete:
        return {"R1": set(), "R2": set()}
    by = {
        s: main.loc[main.stratum == s, "item_id"].tolist()
        for s in ("clean", "flagged", "trap", "trap_gross")
    }
    miss_r1 = pick(by["clean"], 6, rng) | pick(by["trap"], 2, rng)
    miss_r2 = pick([i for i in by["clean"] if i not in miss_r1], 6, rng) | pick(
        by["flagged"], 2, rng
    )
    rest_clean = [i for i in by["clean"] if i not in miss_r1 | miss_r2]
    both = pick(rest_clean, 3, rng) | pick(by["trap_gross"], 1, rng)
    return {"R1": miss_r1 | both, "R2": miss_r2 | both}


def generate(key: pd.DataFrame, scenario: Scenario, rng: np.random.Generator) -> Bundle:
    """R1, R2, and consensus for main items (practice excluded). messy adds incomplete items,
    resaves, and adjudication.
    """
    main = key.loc[key["stratum"] != "practice"].reset_index(drop=True)
    if main.empty:
        raise ValueError("no main items")
    by = {s: main.loc[main.stratum == s, "item_id"].tolist() for s in main.stratum.unique()}
    clean_ids = by.get("clean", [])
    trap_ids = by.get("trap", [])
    gross_ids = by.get("trap_gross", [])
    flag_ids = by.get("flagged", [])
    base_yes = pick(clean_ids, round(scenario.clean_rate * len(clean_ids)), rng) | pick(
        flag_ids, round(0.35 * len(flag_ids)), rng
    )
    detect = {r: pick(trap_ids, round(scenario.trap_sens * len(trap_ids)), rng) for r in RATERS}
    gross_detect = {
        r: pick(gross_ids, round(scenario.gross_sens * len(gross_ids)), rng) for r in RATERS
    }
    missing = _missing_sets(main, scenario, rng)
    completed = [
        i for i in main.item_id.tolist() if i not in missing["R1"] and i not in missing["R2"]
    ]
    resave_ids = pick(completed, 15, rng) if scenario.resave else set()
    force_gap = (
        pick([i for i in completed if i not in resave_ids], 8, rng) if scenario.force_gap else set()
    )

    clock = 0
    t0 = datetime(2026, 9, 26, tzinfo=UTC)

    def stamp() -> str:
        nonlocal clock
        clock += 1
        return (t0 + timedelta(seconds=clock)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    annotations: dict[str, list[dict[str, Any]]] = {r: [] for r in RATERS}
    final: dict[str, dict[str, dict[str, Any]]] = {r: {} for r in RATERS}
    planted_clean = {i for i in clean_ids if i in base_yes}

    for row in main.itertuples(index=False):
        iid, stratum = row.item_id, row.stratum
        yes = {
            r: _rater_yes(stratum, iid, r, base_yes, detect, gross_detect, scenario.flip, rng)
            for r in RATERS
        }
        if yes["R1"] == yes["R2"]:
            scales = dict(zip(RATERS, _draw_pair(yes["R1"], scenario.scale_sd, rng), strict=True))
        else:
            scales = {r: _draw_scales(yes[r], scenario.scale_sd, rng) for r in RATERS}
        if iid in force_gap:
            scales["R2"]["grounded"] = clip_score(scales["R1"]["grounded"] - 2)
            if abs(scales["R2"]["grounded"] - scales["R1"]["grounded"]) < 2:
                scales["R2"]["grounded"] = 1 if scales["R1"]["grounded"] >= 3 else 5
        for r in RATERS:
            if iid in missing[r]:
                continue
            stage1 = {
                "age_guess": guess_cat(str(row.true_age_band), AGES, scenario.age_hit, rng),
                "attend_guess": guess_cat(str(row.true_attend), ATTEND, scenario.attend_hit, rng),
                "secs": int(rng.integers(5, 45)),
            }
            real_s2 = _stage2(yes[r], scales[r], rng)
            if iid in resave_ids:
                flipped = not yes[r]
                decoy_scales = dict(scales[r])
                decoy_scales["grounded"] = 1 if scales[r]["grounded"] >= 4 else 5
                decoy = annotation_record(
                    r, iid, stage1, _stage2(flipped, decoy_scales, rng), stamp()
                )
                annotations[r].append(decoy)
            real = annotation_record(r, iid, stage1, real_s2, stamp())
            annotations[r].append(real)
            final[r][iid] = real

    both = [i for i in completed if i in final["R1"] and i in final["R2"]]
    disagreements = [
        i for i in both if _disagrees(final["R1"][i]["stage2"], final["R2"][i]["stage2"])
    ]
    none_pool = [
        i
        for i in disagreements
        if final["R1"][i]["stage2"]["contradiction"] == "no"
        and final["R2"][i]["stage2"]["contradiction"] == "no"
    ]
    strict_ids = set(none_pool[: scenario.strict_none])
    adjudication: list[dict[str, Any]] = []
    for iid in disagreements:
        r1s, r2s = final["R1"][iid]["stage2"], final["R2"][iid]["stage2"]
        planted_yes = iid in base_yes or main.set_index("item_id").loc[iid, "stratum"] in {
            "trap",
            "trap_gross",
        }
        cons_yes = True if iid in strict_ids else bool(planted_yes)
        changed = (
            "none"
            if iid in strict_ids
            else _who_changed(
                r1s["contradiction"] == "yes", r2s["contradiction"] == "yes", cons_yes
            )
        )
        consensus_row = {
            "contradiction": "yes" if cons_yes else "no",
            "contra_fields": [CONTRA_FIELD] if cons_yes else [],
            "contra_quote": "예행연습 합의: 엄격한 쪽." if cons_yes else "",
            "note": "",
        }
        for sc in SCALES:
            consensus_row[sc] = clip_score((int(r1s[sc]) + int(r2s[sc])) / 2)
        if cons_yes:
            consensus_row["grounded"] = min(2, consensus_row["grounded"])
        adjudication.append(
            adjudication_record(
                iid,
                consensus_row,
                changed,
                "예행연습 합의 판정",
                "R1" if len(adjudication) % 2 == 0 else "R2",
                stamp(),
            )
        )

    truth = {
        "scenario": scenario.name,
        "n_main": len(main),
        "n_clean": len(clean_ids),
        "n_clean_planted_yes": len(planted_clean),
        "clean_planted_rate": (len(planted_clean) / len(clean_ids)) if clean_ids else None,
        "trap_detect_n": {r: len(detect[r]) for r in RATERS},
        "n_trap": len(trap_ids),
        "trap_sens_planted": {
            r: (len(detect[r]) / len(trap_ids)) if trap_ids else None for r in RATERS
        },
        "n_missing": {r: len(missing[r]) for r in RATERS},
        "n_resave_items": len(resave_ids),
        "resave_ids": sorted(resave_ids),
        "n_disagreement": len(disagreements),
        "n_adjudicated": len(adjudication),
        "n_strict_none": len(strict_ids),
        "final_contra": {
            r: {i: final[r][i]["stage2"]["contradiction"] for i in final[r]} for r in RATERS
        },
    }
    return Bundle(annotations, adjudication, truth)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialise(bundle: Bundle, store: Path) -> None:
    assert_sandbox(store)
    ann = store / "annotations"
    if ann.exists():
        shutil.rmtree(ann)
    for rater, rows in bundle.annotations.items():
        write_jsonl(ann / f"{rater}.jsonl", rows)
    write_jsonl(store / "adjudication.jsonl", bundle.adjudication)
    for name in LLM_NAMES:
        src = REAL_ANNO / name
        dest = store / name
        if not src.exists():
            raise SystemExit(f"LLM decisions missing: {src}")
        assert_sandbox(dest)
        shutil.copy2(src, dest)


def verify_last_wins(store: Path, key_path: Path, truth: dict[str, Any]) -> str | None:
    """For resaved items, the file's last line must match what load() returns."""
    df, _adj, _llm, _bank = load(store, key_path)
    if truth["scenario"] != "practice-test" and int((df.stratum == "practice").sum()) != 0:
        return "practice items remain in the analysis table"
    resave = set(truth.get("resave_ids") or [])
    if not resave:
        return None
    for rater in RATERS:
        rows = [
            json.loads(line)
            for line in (store / "annotations" / f"{rater}.jsonl").open(encoding="utf-8")
            if line.strip()
        ]
        first: dict[str, dict[str, Any]] = {}
        last: dict[str, dict[str, Any]] = {}
        for row in rows:
            last[row["item_id"]] = row
            first.setdefault(row["item_id"], row)
        flipped = [
            i
            for i in resave
            if i in first
            and first[i]["stage2"]["contradiction"] != last[i]["stage2"]["contradiction"]
        ]
        if not flipped:
            return f"{rater} resave did not change the contradiction label"
        got = df.set_index("item_id")
        for iid in flipped:
            if iid not in got.index:
                continue
            loaded = "yes" if int(got.loc[iid, f"{rater}_contra"]) == 1 else "no"
            if loaded != last[iid]["stage2"]["contradiction"]:
                return f"{rater} {iid} is not the last save"
            if loaded == first[iid]["stage2"]["contradiction"]:
                return f"{rater} {iid} used the first save"
    return None


def _as_bool(v: object) -> bool | None:
    try:
        if v is None or pd.isna(v):
            return None
    except TypeError:
        pass
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return bool(v)


def oracle(store: Path, key_path: Path) -> dict[str, Any]:
    """Recompute the truth with the same load and consensus as the analysis. A mismatched
    estimate is an analysis bug.
    """
    df, adj, _llm, _bank = load(store, key_path)
    df = consensus(df, adj)
    clean = df[df.stratum == "clean"].dropna(subset=["cons_contra"])
    k, n = int(clean.cons_contra.sum()), len(clean)
    trap = df[df.stratum == "trap"]
    sens = {}
    for r in RATERS:
        x = trap[f"{r}_contra"].dropna()
        sens[r] = (int(x.sum()) / len(x)) if len(x) else None
    cf = df[df.stratum.isin(["clean", "flagged"])].dropna(subset=["R1_grounded", "R2_grounded"])
    alpha = kripp_alpha_ordinal(cf.R1_grounded.to_numpy(int) - 1, cf.R2_grounded.to_numpy(int) - 1)
    hi = cp_upper(k, n) if n else None
    return {
        "p1_k": k,
        "p1_n": n,
        "p1_estimate": (k / n) if n else None,
        "p1_hi": hi,
        "p1_holds": (hi < P1_BOUND) if hi is not None else None,
        "trap_sens": sens,
        "p2_alpha": alpha,
        "p2_n": len(cf),
        "n_unresolved": int(clean.unresolved.sum()) if n else 0,
    }


def run_analysis(store: Path, key_path: Path, reports: Path) -> subprocess.CompletedProcess[str]:
    assert_sandbox(reports)
    reports.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/anno_analysis.py",
        "--store",
        str(store),
        "--key",
        str(key_path),
        "--out",
        str(reports),
    ]
    log.info("running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )


def _macro_body(name: str, text: str) -> str | None:
    needle = "\\newcommand{\\" + name + "}{"
    i = text.find(needle)
    if i < 0:
        return None
    j = text.find("}", i + len(needle))
    if j < 0:
        return None
    return text[i + len(needle) : j]


def _load_tables():
    path = ROOT / "sfac_persona_paper" / "scripts" / "make_tables_era.py"
    spec = importlib.util.spec_from_file_location("make_tables_era_dryrun", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tables module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_audit(reports: Path, tables: Path, tables_mod: Any) -> dict[str, Any]:
    """Call only `audit_era`. REPORTS and TABLES are pointed at temporary folders."""
    assert_sandbox(tables)
    tables.mkdir(parents=True, exist_ok=True)
    tables_mod.REPORTS = reports
    tables_mod.TABLES = tables
    try:
        tables_mod.audit_era()
    except Exception as exc:  # noqa: BLE001 — a manuscript-link failure is kept as a result too
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(),
        }
    text = (tables / "audit.tex").read_text(encoding="utf-8")
    leaks = [tok for tok in ("nan", "NaN", "pending", "None") if tok in text]
    rows = []
    for line in text.splitlines():
        if "&" not in line or "\\\\" not in line or "Registered claim" in line:
            continue
        parts = [p.strip() for p in line.split("&")]
        holds = parts[-1].replace("\\\\", "").strip()
        rows.append({"estimate": parts[1], "bound": parts[2], "n": parts[3], "holds": holds})
    empty = [
        i
        for i, r in enumerate(rows)
        if r["estimate"] in {"", "---", "nan", "NaN"} or r["holds"] not in {"yes", "no"}
    ]
    numbers = (REAL_TABLES / "numbers.tex").read_text(encoding="utf-8")
    macros = {name: _macro_body(name, numbers) for name in AUDIT_MACROS}
    in_tex = {name: ("\\" + name) in text for name in AUDIT_MACROS}
    return {
        "ok": True,
        "n_chars": len(text),
        "n_rows": len(rows),
        "leaks": leaks,
        "empty_rows": empty,
        "rows": rows,
        "macros_defined": macros,
        "macros_referenced": in_tex,
    }


def _close(a: float | None, b: float | None, tol: float = 1e-8) -> bool:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)):
        return False
    return abs(a - b) < tol


def compare(reports: Path, ora: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    csv_path = reports / "anno_results.csv"
    if not csv_path.exists():
        return {"ok": False, "error": "anno_results.csv is missing"}
    res = pd.read_csv(csv_path).set_index("id")
    out: dict[str, Any] = {"ok": True, "ids": {}}
    for pid, estimate_key, holds_key in (
        ("P1", "p1_estimate", "p1_holds"),
        ("S1a", None, None),
        ("S1b", None, None),
        ("P2", "p2_alpha", None),
    ):
        if pid not in res.index:
            out["ids"][pid] = None
            continue
        row = res.loc[pid]
        est = None if pd.isna(row.estimate) else float(row.estimate)
        hi = None if "hi" not in row or pd.isna(row.hi) else float(row.hi)
        lo = None if "lo" not in row or pd.isna(row.lo) else float(row.lo)
        holds = _as_bool(row.holds) if "holds" in row else None
        out["ids"][pid] = {"estimate": est, "hi": hi, "lo": lo, "n": row.n, "holds": holds}
    p1 = out["ids"].get("P1") or {}
    s1a = out["ids"].get("S1a") or {}
    s1b = out["ids"].get("S1b") or {}
    p2 = out["ids"].get("P2") or {}
    out["p1_matches_oracle"] = _close(p1.get("estimate"), ora["p1_estimate"])
    out["p1_holds_matches"] = p1.get("holds") == ora["p1_holds"]
    out["s1a_matches_oracle"] = _close(s1a.get("estimate"), ora["trap_sens"]["R1"])
    out["s1b_matches_oracle"] = _close(s1b.get("estimate"), ora["trap_sens"]["R2"])
    out["p2_matches_oracle"] = _close(p2.get("estimate"), ora["p2_alpha"], tol=1e-6)
    out["direction_p1"] = p1.get("holds")
    if audit.get("ok") and audit.get("rows"):
        csv_holds = []
        tex_holds = []
        for pid, tex_row in zip(AUDIT_ORDER, audit["rows"], strict=False):
            if pid not in res.index:
                continue
            csv_holds.append(_as_bool(res.loc[pid, "holds"]))
            tex_holds.append(tex_row["holds"] == "yes")
        out["audit_holds_match_csv"] = csv_holds == tex_holds
        out["audit_csv_holds"] = csv_holds
        out["audit_tex_holds"] = tex_holds
    return out


def _fmt(v: object) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def run_scenario(
    scenario: Scenario,
    key_path: Path,
    out: Path,
    seed: int,
    tables_mod: Any,
) -> dict[str, Any]:
    dest = out / scenario.name
    store, reports, tables = dest / "store", dest / "reports", dest / "tables"
    for p in (dest, store, reports, tables):
        assert_sandbox(p)
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + {"good": 0, "poor": 1, "messy": 2}[scenario.name])
    key = pd.read_csv(key_path, dtype=str).fillna("")
    bundle = generate(key, scenario, rng)
    (dest / "truth.json").write_text(
        json.dumps(
            {k: v for k, v in bundle.truth.items() if k != "final_contra"},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    materialise(bundle, store)
    last_err = verify_last_wins(store, key_path, bundle.truth)
    ora = oracle(store, key_path)
    log.info(
        "%s planted clean %.3f · consensus %.3f (upper %s, expected %s) · traps %s · "
        "resaves %d · adjudicated %d",
        scenario.name,
        bundle.truth["clean_planted_rate"],
        ora["p1_estimate"],
        _fmt(ora["p1_hi"]),
        ora["p1_holds"],
        {r: _fmt(ora["trap_sens"][r]) for r in RATERS},
        bundle.truth["n_resave_items"],
        bundle.truth["n_adjudicated"],
    )
    proc = run_analysis(store, key_path, reports)
    log_path = dest / "analysis.log"
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
    err_tail = "\n".join((proc.stderr or "").splitlines()[-30:])
    audit = (
        run_audit(reports, tables, tables_mod)
        if proc.returncode == 0
        else {"ok": False, "error": "analysis failed"}
    )
    cmp = compare(reports, ora, audit) if proc.returncode == 0 else {"ok": False}
    return {
        "scenario": scenario.name,
        "returncode": proc.returncode,
        "error_tail": err_tail if proc.returncode else "",
        "last_wins_error": last_err,
        "truth": {k: v for k, v in bundle.truth.items() if k != "final_contra"},
        "oracle": ora,
        "compare": cmp,
        "audit": {k: v for k, v in audit.items() if k != "rows"} | {"rows": audit.get("rows")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="human-audit analysis dry run (fake scores, temp dirs only)"
    )
    ap.add_argument("--scenario", choices=[*SCENARIOS, "all"], default="all")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "logs" / "anno_dryrun")
    ap.add_argument("--key", type=Path, default=REAL_ANNO / "answer_key.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else (ROOT / args.out).resolve()
    key = args.key if args.key.is_absolute() else (ROOT / args.key).resolve()
    assert_sandbox(out)
    if not key.exists():
        raise SystemExit(f"answer key missing: {key}")
    before = snapshot(WATCH)
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    tables_mod = _load_tables()
    results = [run_scenario(SCENARIOS[n], key, out, args.seed, tables_mod) for n in names]
    after = snapshot(WATCH)
    touched = [p for p, t in before.items() if after[p] != t]
    summary = {"watched_changed": touched, "scenarios": results}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if touched:
        log.error("real outputs changed: %s", touched)
        return 1
    log.info("dry run finished: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
