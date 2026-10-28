"""Sampler chain.

A Bayesian network that draws personas from grounding tables (conditional
distributions) only. Microdata is used only to *estimate* the conditional tables;
the sampler draws from the tables only (no resampling, ADR-003).

Design (following the grounding-error study)
  1. Root (sex, age_band, living_area) — public release uses resident-registration
     population R0; the error study uses the survey weighted distribution
  2. Demographic and household anchors — conditional on age band (+ sex, education, …)
  3. attend_any — parent of the attendance block
  4. 10 genres — weighted mutual-information Chow–Liu tree among attenders (pair
     correlations kept as a tree; cells are 2×2 so they always meet the floor)
  5. Attendance count, spend, companion, satisfaction — parents are attend_any and genre count
  6. The rest

Cell-size rule: if the unweighted n of a parent combination is < cell_floor, drop the
last parent and back off. Backoff counts are recorded.
"""

from __future__ import annotations

import itertools
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from sfac_persona.config import settings
from sfac_persona.schema import GENRE_ANCHORS, ROOT_ANCHORS

logger = logging.getLogger(__name__)

SEP = "|"


# ----------------------------------------------------------------------------
# Derived nodes (computed deterministically from sampled values)
# ----------------------------------------------------------------------------


N_GENRES_LEVELS = ("0", "1", "2", "3", "4", "5+")


def _n_genres_band(state: dict[str, Any]) -> str:
    n = sum(1 for g in GENRE_ANCHORS if state.get(g) == "예")
    return "5+" if n >= 5 else str(n)


DERIVED: dict[str, Callable[[dict[str, Any]], str]] = {"n_genres_band": _n_genres_band}


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the same derived columns to a microdata frame so conditional tables
    can be estimated.
    """
    out = frame.copy()
    n = (out[GENRE_ANCHORS] == "예").sum(axis=1)
    out["n_genres_band"] = np.where(n >= 5, "5+", n.astype(str))
    return out


# ----------------------------------------------------------------------------
# Chain specification
# ----------------------------------------------------------------------------


@dataclass
class Node:
    name: str
    parents: list[str] = field(default_factory=list)
    derived: str | None = None  # DERIVED key
    note: str = ""


def era_core_chain() -> list[Node]:
    """ADR-007 Tier 0–2 common chain (Tier 3 such as marital excluded)."""
    nodes: list[Node] = [
        Node("root", note="R0 or survey weights (sex, age_band, living_area)"),
        Node("education", ["age_band", "sex"]),
        Node("occupation", ["age_band", "sex", "education"]),
        Node("household_income_band", ["age_band", "occupation"]),
        Node("household_size_band", ["age_band", "sex"]),
        Node("seoul_years_band", ["age_band", "living_area"]),
        Node("leisure_sufficiency", ["age_band", "occupation"]),
        Node("leisure_companion", ["age_band", "sex"]),
        Node("attend_any", ["age_band", "education", "household_income_band"]),
    ]
    nodes.append(Node("n_genres_band", ["attend_any", "age_band"]))
    nodes.append(Node("__genre_tree__", ["attend_any", "age_band"]))
    nodes += [
        Node("attend_count_band", ["n_genres_band", "age_band"]),
        Node("paid_spend_band", ["attend_count_band", "household_income_band"]),
        Node("viewing_companion", ["attend_any", "leisure_companion"]),
    ]
    return nodes


def default_chain() -> list[Node]:
    """Parent choice follows conditional TVD in phase01_error_study_pairs.csv and
    the survey skip structure.
    """
    nodes: list[Node] = [
        Node("root", note="R0 or survey-weight joint (sex, age_band, living_area)"),
        # Demographics and household
        Node("education", ["age_band", "sex"]),
        Node("occupation", ["age_band", "sex", "education"]),
        Node("marital", ["age_band", "sex"]),
        Node(
            "child_cohabiting",
            ["marital", "age_band"],
            note="the survey skips the unmarried → marital is the parent",
        ),
        Node("household_income_band", ["age_band", "occupation"]),
        Node("subjective_class", ["household_income_band", "age_band"]),
        Node("seoul_years_band", ["age_band", "living_area"]),
        Node("arts_worker", ["age_band", "occupation"]),
        Node("reading", ["age_band", "education"]),
        Node("leisure_sufficiency", ["age_band", "occupation"]),
        Node("leisure_companion", ["age_band", "marital"]),
        # Attendance — parent of the block
        Node("attend_any", ["age_band", "education", "household_income_band"]),
    ]
    # Draw genre count from the table first (09-07): a Chow–Liu tree stores only pair
    # dependence and spreads the genre-count distribution (1–2 genres 47.6% → 38.3%,
    # 3–4 genres 20.1% → 30.0%). The tree only answers 'which genres go together';
    # this node sets the count.
    nodes.append(
        Node(
            "n_genres_band",
            ["attend_any", "age_band"],
            note="attender genre count 0/1/2/3/4/5+; the tree pattern is fit to this count",
        )
    )
    # The genre tree is fixed from data in build_spec (Chow–Liu). This only marks the slot.
    nodes.append(
        Node(
            "__genre_tree__", ["attend_any", "age_band"], note="Chow–Liu tree, filled by build_spec"
        )
    )
    nodes += [
        Node("attend_count_band", ["n_genres_band", "age_band"]),
        Node("paid_spend_band", ["attend_count_band", "household_income_band"]),
        Node("viewing_companion", ["attend_any", "leisure_companion"]),
        Node("viewing_satisfaction", ["attend_any", "n_genres_band"]),
        Node("intent_max", ["attend_any", "age_band"]),
        Node("participation_any", ["attend_any", "age_band"]),
        Node("activity_place", ["attend_any", "living_area"]),
        Node("seoul_culture_status", ["viewing_satisfaction", "age_band"]),
    ]
    return nodes


# ----------------------------------------------------------------------------
# Chow–Liu genre tree
# ----------------------------------------------------------------------------


def _weighted_mi(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    total = w.sum()
    mi = 0.0
    for a in (0, 1):
        pa = w[x == a].sum() / total
        for b in (0, 1):
            pab = w[(x == a) & (y == b)].sum() / total
            pb = w[y == b].sum() / total
            if pab > 0:
                mi += pab * np.log(pab / (pa * pb))
    return float(mi)


def chow_liu_genre_tree(frame: pd.DataFrame) -> list[tuple[str, str | None]]:
    """Maximum spanning tree of the 10 genres among attenders (attend_any=예).
    Returns (node, parent) pairs in topological order from the root.
    """
    sub = frame[frame["attend_any"] == "예"]
    w = sub["w"].to_numpy(dtype=float)
    x = {g: (sub[g] == "예").to_numpy(dtype=int) for g in GENRE_ANCHORS}
    edges = sorted(
        ((_weighted_mi(x[a], x[b], w), a, b) for a, b in itertools.combinations(GENRE_ANCHORS, 2)),
        reverse=True,
    )
    parent_of: dict[str, str] = {}
    comp = {g: g for g in GENRE_ANCHORS}

    def find(u: str) -> str:
        while comp[u] != u:
            comp[u] = comp[comp[u]]
            u = comp[u]
        return u

    adj: dict[str, list[str]] = {g: [] for g in GENRE_ANCHORS}
    for _, a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            comp[ra] = rb
            adj[a].append(b)
            adj[b].append(a)
    # Root: the genre with the highest attendance rate (movies)
    prev = {g: float(np.average(x[g], weights=w)) for g in GENRE_ANCHORS}
    root = max(prev, key=prev.get)
    order: list[tuple[str, str | None]] = [(root, None)]
    seen = {root}
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                parent_of[v] = u
                order.append((v, u))
                queue.append(v)
    logger.info("Chow-Liu genre tree root=%s edges=%s", root, [(v, p) for v, p in order if p])
    return order


# ----------------------------------------------------------------------------
# Conditional-table estimation (with backoff)
# ----------------------------------------------------------------------------


def _key(values: list[Any]) -> str:
    return SEP.join(str(v) for v in values)


def conditional_table(
    frame: pd.DataFrame, target: str, parents: list[str], floor: int
) -> dict[str, Any]:
    """levels[k] = table using only the first k parents. At sample time, try the longest
    key first and back off when n<floor.
    """
    sub = frame.dropna(subset=[target])
    levels: list[dict[str, dict[str, Any]]] = []
    for k in range(len(parents), -1, -1):
        ps = parents[:k]
        tbl: dict[str, dict[str, Any]] = {}
        if ps:
            g = sub.groupby(ps, observed=True)
            for keyvals, grp in g:
                keyvals = keyvals if isinstance(keyvals, tuple) else (keyvals,)
                if len(grp) < floor:
                    continue
                dist = grp.groupby(target, observed=True)["w"].sum()
                tbl[_key(list(keyvals))] = {
                    "values": [str(v) for v in dist.index],
                    "weights": [float(v) for v in (dist / dist.sum()).values],
                    "n": len(grp),
                }
        else:
            dist = sub.groupby(target, observed=True)["w"].sum()
            tbl[""] = {
                "values": [str(v) for v in dist.index],
                "weights": [float(v) for v in (dist / dist.sum()).values],
                "n": len(sub),
            }
        levels.append({"parents": ps, "table": tbl})
    return {"name": target, "parents": parents, "levels": levels}


def root_table(frame: pd.DataFrame, source: str, r0_path: Path | None = None) -> dict[str, Any]:
    if source == "population":
        path = r0_path or (settings.grounding_dir / "R0_population_root.parquet")
        r0 = pd.read_parquet(path)
        keys = [
            _key([a, b, c])
            for a, b, c in zip(
                r0.sex.astype(str), r0.age_band.astype(str), r0.living_area.astype(str)
            )
        ]
        weights = (r0.population / r0.population.sum()).tolist()
    elif source == "survey":
        g = frame.groupby(ROOT_ANCHORS, observed=True)["w"].sum()
        keys = [_key(list(k)) for k in g.index]
        weights = (g / g.sum()).tolist()
    else:
        raise ValueError(source)
    return {
        "name": "root",
        "fields": ROOT_ANCHORS,
        "source": source,
        "values": keys,
        "weights": weights,
    }


def build_spec(
    frame: pd.DataFrame,
    root_source: str = "population",
    floor: int | None = None,
    nodes: list[Node] | None = None,
    r0_path: Path | None = None,
) -> dict[str, Any]:
    """If nodes is given, use that structure instead of default_chain() (parent-search experiments)."""
    floor = settings.cell_floor if floor is None else floor
    frame = add_derived_columns(frame)
    nodes = default_chain() if nodes is None else nodes
    # skip nodes whose target column is absent (era core vs full)
    filtered: list[Node] = []
    for node in nodes:
        if node.name in ("root", "__genre_tree__", "n_genres_band") or node.derived:
            filtered.append(node)
            continue
        if node.name in frame.columns:
            filtered.append(node)
        else:
            logger.info("build_spec: skip missing anchor %s", node.name)
    nodes = filtered
    spec: dict[str, Any] = {
        "built_at": datetime.now(UTC).isoformat(),
        "root_source": root_source,
        "cell_floor": floor,
        "root": root_table(frame, root_source, r0_path=r0_path),
        "nodes": [],
        "genre_tree": [],
    }
    for node in nodes:
        if node.name == "root":
            continue
        if node.name == "__genre_tree__":
            tree = chow_liu_genre_tree(frame)
            spec["genre_tree"] = [{"genre": g, "parent": p} for g, p in tree]
            for g, p in tree:
                parents = ["attend_any", "age_band"] + ([p] if p else [])
                spec["nodes"].append(conditional_table(frame, g, parents, floor))
            continue
        if node.derived:
            spec["nodes"].append({"name": node.name, "derived": node.derived})
            continue
        # drop parents not in frame
        parents = [p for p in node.parents if p in frame.columns or p == "n_genres_band"]
        spec["nodes"].append(conditional_table(frame, node.name, parents, floor))
    return spec


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------


class ChainSampler:
    def __init__(self, spec: dict[str, Any]):
        self.spec = spec
        self.backoff_counts: dict[str, list[int]] = {}
        self.rejections = 0  # local redraw failed 100 times and the invariant broke — must stay 0
        self.genre_fallbacks = (
            0  # target genre-count pattern missed in 200 draws, then fit by probability order
        )
        self.enforce_local = (
            True  # False disables local invariant repair (bias-comparison runs only)
        )

    def _draw(self, rng: np.random.Generator, entry: dict[str, Any]) -> str:
        return entry["values"][
            rng.choice(len(entry["values"]), p=np.asarray(entry["weights"]) / sum(entry["weights"]))
        ]

    _COUNT_MAX: ClassVar[dict[str, int]] = {
        "0회": 0,
        "1~2회": 2,
        "3~5회": 5,
        "6~10회": 10,
        "11회 이상": 10**6,
    }

    @classmethod
    def _consistent(cls, state: dict[str, str]) -> bool:
        """Hard invariants from survey logic (gate G0). Final check."""
        n_genre = sum(1 for g in GENRE_ANCHORS if state.get(g) == "예")
        if (state.get("attend_any") == "예") != (n_genre > 0):
            return False
        return cls._COUNT_MAX.get(state.get("attend_count_band", ""), 10**6) >= n_genre

    def sample_one(
        self, rng: np.random.Generator, fixed_root: dict[str, str] | None = None
    ) -> dict[str, str]:
        """If fixed_root is given, pin the root to those values (cell holdout checks).

        Invariants are kept by redrawing **only the offending block** (second fix,
        09-07). The first version discarded the whole person and redrew; attenders
        were discarded more often, so the non-attender share drifted 23.9% → 26.5%.
        Redrawing only the block leaves the parents' marginals unchanged.
          R1  if attend_any == 예 and no genre is on, redraw the genre block (attend_any held fixed)
          R6  if the attend_count_band ceiling is < the genre count, or an attender has 0회,
              redraw the count band (parents held fixed)
        """
        state = self._sample_once(rng, fixed_root)
        if not self._consistent(state):
            self.rejections += 1  # only when local redraw failed 100 times — the table is defective
        return state

    def _p_on(self, node: dict[str, Any], state: dict[str, str]) -> float:
        """Probability this genre node is '예' under the current parents (with backoff)."""
        for level in node["levels"]:
            entry = level["table"].get(_key([state[p] for p in level["parents"]]))
            if entry is not None:
                w = dict(zip(entry["values"], entry["weights"]))
                return w.get("예", 0.0) / max(sum(entry["weights"]), 1e-12)
        return 0.0

    def _fit_genres_to_count(
        self, rng: np.random.Generator, genre_nodes: list[dict[str, Any]], state: dict[str, str]
    ) -> None:
        """Fit the genre block to the count set by n_genres_band. Redraw only the block
        until a matching pattern appears; if none appears in 200 draws, turn genres on
        or off in conditional-probability order.
        """
        band = state["n_genres_band"]
        lo, hi = (5, 10) if band == "5+" else (int(band), int(band))
        names = [g["name"] for g in genre_nodes]

        def count() -> int:
            return sum(1 for g in names if state[g] == "예")

        for _ in range(200):
            if lo <= count() <= hi:
                return
            for gn in genre_nodes:
                state[gn["name"]] = self._draw_node(rng, gn, state)
        self.genre_fallbacks += 1
        p = sorted(((self._p_on(gn, state), gn["name"]) for gn in genre_nodes), reverse=True)
        while count() < lo:
            state[next(n for _, n in p if state[n] != "예")] = "예"
        while count() > hi:
            state[next(n for _, n in reversed(p) if state[n] == "예")] = "아니오"

    def _draw_node(
        self, rng: np.random.Generator, node: dict[str, Any], state: dict[str, str]
    ) -> str:
        for level_idx, level in enumerate(node["levels"]):
            key = _key([state[p] for p in level["parents"]])
            entry = level["table"].get(key)
            if entry is not None:
                self.backoff_counts.setdefault(node["name"], [0] * len(node["levels"]))[
                    level_idx
                ] += 1
                return self._draw(rng, entry)
        raise RuntimeError(f"no table for {node['name']} even at marginal level")

    def _sample_once(
        self, rng: np.random.Generator, fixed_root: dict[str, str] | None = None
    ) -> dict[str, str]:
        root = self.spec["root"]
        if fixed_root is None:
            state: dict[str, str] = dict(zip(root["fields"], self._draw(rng, root).split(SEP)))
        else:
            missing = [f for f in root["fields"] if f not in fixed_root]
            if missing:
                raise KeyError(f"fixed_root is missing {missing}")
            state = {f: fixed_root[f] for f in root["fields"]}
        genre_nodes = [n for n in self.spec["nodes"] if n.get("name") in GENRE_ANCHORS]
        last_genre = genre_nodes[-1]["name"] if genre_nodes else None
        for node in self.spec["nodes"]:
            if "derived" in node:
                state[node["name"]] = DERIVED[node["derived"]](state)
                continue
            state[node["name"]] = self._draw_node(rng, node, state)
            if not self.enforce_local:
                continue
            if (
                node["name"] == "n_genres_band"
                and state.get("attend_any") == "예"
                and state["n_genres_band"] == "0"
            ):
                state["n_genres_band"] = (
                    "1"  # backoff can use the marginal and give an attender 0 — R1
                )
            if node["name"] == "n_genres_band" and state.get("attend_any") == "아니오":
                state["n_genres_band"] = "0"
            if node["name"] == last_genre and "n_genres_band" in state:
                # Only when genre count was drawn first (the ablated no_count_node variant
                # computes it later as a derived node)
                self._fit_genres_to_count(rng, genre_nodes, state)
            elif (
                node["name"] == last_genre
                and state.get("attend_any") == "예"
                and not any(state.get(g) == "예" for g in GENRE_ANCHORS)
            ):
                state[genre_nodes[0]["name"]] = "예"  # R1 floor (no genre-count node)
            if node["name"] == "attend_count_band" and state.get("attend_any") == "예":
                # R6/R3: count-band ceiling ≥ genre count; an attender is not 0회 —
                # redraw only the count band
                n_genre = max(1, sum(1 for g in GENRE_ANCHORS if state.get(g) == "예"))
                for _ in range(100):
                    if (
                        self._COUNT_MAX.get(state["attend_count_band"], 10**6) >= n_genre
                        and state["attend_count_band"] != "0회"
                    ):
                        break
                    state["attend_count_band"] = self._draw_node(rng, node, state)
                else:
                    state["attend_count_band"] = next(
                        b
                        for b, m in sorted(self._COUNT_MAX.items(), key=lambda kv: kv[1])
                        if m >= n_genre
                    )
        return state

    def sample(self, n: int, seed: int, fixed_root: dict[str, str] | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = [self.sample_one(rng, fixed_root) for _ in range(n)]
        return pd.DataFrame(rows)

    def backoff_report(self) -> pd.DataFrame:
        rows = []
        for name, counts in self.backoff_counts.items():
            total = sum(counts)
            rows.append(
                {
                    "node": name,
                    "full_parents_share": counts[0] / total if total else np.nan,
                    "marginal_share": counts[-1] / total if total else np.nan,
                }
            )
        return pd.DataFrame(rows)


def write_spec(spec: dict[str, Any], name: str = "sampler_specs.json") -> None:
    settings.grounding_dir.mkdir(parents=True, exist_ok=True)
    (settings.grounding_dir / name).write_text(
        json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_spec(name: str = "sampler_specs.json") -> dict[str, Any]:
    return json.loads((settings.grounding_dir / name).read_text(encoding="utf-8"))
