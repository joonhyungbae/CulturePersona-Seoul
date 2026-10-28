"""Attribute disclosure for real respondents — released tables and bank.

The 59%/28% in the ethics section is a probe that recovers anchors from synthetic
narratives. Here we measure the risk of guessing household-income band, any-attendance,
and genre-count band when an attacker knows only a real respondent's (sex, age band,
living area, education, occupation). Individual rows stay in memory; only aggregates are
written to disk.

Three predictors.

  marginal  Majority category of the released T1 marginal. The genre-count band is absent
            from T1, so the marginal of the released sampler spec is used.
  tables    From the released sampler spec, the longest conditional P(target | parents)
            whose parents are all known. A missing cell backs off to shorter parents.
            Keys not yet used are updated by naive Bayes on the released T1–T6.
            A suppressed missing cell is independence (likelihood ratio 1).
  bank      Logistic regression trained on the five keys of the released quantitative bank
            (the same classifier as era_probes). Narrative columns are not read.

Accuracy is the weighted hit rate of the point prediction. CAP is the correct attribution
probability (Taub et al. 2018). For the table predictor it is the weighted mean of the
probability the conditional assigns to the true target; for the bank it is the weighted
mean of the share of that target value among bank rows with the same five keys. If the key
is absent from the bank, fall back to the bank marginal share.

Run: uv run python scripts/era_attribute_disclosure.py
Outputs: outputs/reports/era_attribute_disclosure.csv · .md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings
from sfac_persona.grounding.chain import add_derived_columns
from sfac_persona.grounding.official import build_official_anchor_frame
from sfac_persona.schema import WEIGHTED_WAVES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("sfac_persona.grounding.official").setLevel(logging.WARNING)
log = logging.getLogger("attribute_disclosure")

SEED = 20260926
RELEASE_VERSION = "v1.0-rc1"
KEYS = ["sex", "age_band", "living_area", "education", "occupation"]
TARGETS = ["household_income_band", "attend_any", "n_genres_band"]
PREDICTORS = ["marginal", "tables", "bank"]
# When the spec does not use these keys as parents, update from the T6 joint.
PAIR_KEYS = ("education", "occupation")
UNIVARIATE = ("sex", "age_band", "living_area")
_LOG0 = -1e9

_TABLE_FILES = {
    "T1": "T1_release.parquet",
    "T2": "T2_release.parquet",
    "T3": "T3_release.parquet",
    "T4": "T4_release.parquet",
    "T5": "T5_release.parquet",
    "T6": "T6_pair_joints_release.parquet",
}


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("weights sum to 0")
    return float(np.dot(values, weights) / total)


def _join_keys(df: pd.DataFrame, cols: list[str] = KEYS) -> np.ndarray:
    arr = df[cols].astype(str).to_numpy()
    return np.array(["|".join(row) for row in arr], dtype=object)


def correct_attribution_probability(
    real_keys: np.ndarray,
    real_target: np.ndarray,
    source_keys: np.ndarray,
    source_target: np.ndarray,
    *,
    source_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted share of same-key source rows that hold the true target (per-row Taub CAP).

    If the key is absent from the source, fall back to that target's source marginal.
    If the key is present but no row has that target value, the share is 0.
    """
    if source_weight is None:
        source_weight = np.ones(len(source_keys), dtype=float)
    else:
        source_weight = np.asarray(source_weight, dtype=float)
    src = pd.DataFrame(
        {"k": np.asarray(source_keys), "y": np.asarray(source_target), "w": source_weight}
    )
    by_key = src.groupby("k", sort=False)["w"].sum()
    by_cell = src.groupby(["k", "y"], sort=False)["w"].sum()
    prop = (by_cell / by_key).rename("p").reset_index()
    marg = src.groupby("y", sort=False)["w"].sum()
    marg = marg / marg.sum()
    real = pd.DataFrame({"k": np.asarray(real_keys), "y": np.asarray(real_target)})
    merged = real.merge(prop, on=["k", "y"], how="left")
    known = real["k"].isin(by_key.index).to_numpy()
    filled = merged["p"].to_numpy(dtype=float)
    marginal = real["y"].map(marg).fillna(0.0).to_numpy(dtype=float)
    return np.where(known, np.nan_to_num(filled, nan=0.0), marginal)


def marginal_attribution_probability(
    real_target: np.ndarray,
    source_target: np.ndarray,
    *,
    source_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Probability the source marginal assigns to the true target, ignoring the key."""
    if source_weight is None:
        source_weight = np.ones(len(source_target), dtype=float)
    else:
        source_weight = np.asarray(source_weight, dtype=float)
    src = pd.DataFrame({"y": np.asarray(source_target), "w": source_weight})
    marg = src.groupby("y", sort=False)["w"].sum()
    marg = marg / marg.sum()
    return pd.Series(np.asarray(real_target)).map(marg).fillna(0.0).to_numpy(dtype=float)


def _softmax(logp: dict[str, float]) -> dict[str, float]:
    peak = max(logp.values())
    if peak <= -1e8:
        n = len(logp)
        return {k: 1.0 / n for k in logp}
    weights = {k: 0.0 if v - peak < -700 else math.exp(v - peak) for k, v in logp.items()}
    total = sum(weights.values())
    if total <= 0:
        n = len(logp)
        return {k: 1.0 / n for k in logp}
    return {k: v / total for k, v in weights.items()}


def _mode(dist: dict[str, float]) -> str:
    return min(dist, key=lambda k: (-dist[k], k))


class ReleasedTables:
    """Released T1–T6 and the released sampler spec. Does not read microdata."""

    def __init__(self, wave: int):
        era = settings.era_grounding_dir(wave)
        spec_path = era / "sampler_specs_structure_T.json"
        if not spec_path.exists():
            raise SystemExit(f"released sampler spec missing: {spec_path}")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.nodes = {
            n["name"]: n for n in spec["nodes"] if n.get("name") in TARGETS and "levels" in n
        }
        self.t1: dict[tuple[str, str], float] = {}
        self.t2: dict[tuple[str, str, str], float] = {}
        self.t3: dict[tuple[str, str, str], float] = {}
        self.t4: dict[tuple[str, str, str], float] = {}
        self.t5: dict[tuple[str, str, str, str], float] = {}
        self.joint: dict[tuple[str, str, str, str], float] = {}
        self._load_margins(era)
        self._load_conditionals(era)
        self._load_joints(era)
        self.prior: dict[str, dict[str, float]] = {}
        for target in TARGETS:
            if any(a == target for a, _v in self.t1):
                dist = {v: p for (a, v), p in self.t1.items() if a == target}
            elif target in self.nodes:
                dist, _parents = self._spec(
                    self.nodes[target], {k: "" for k in KEYS}, marginal_only=True
                )
            else:
                raise SystemExit(f"wave {wave}: no released marginal for {target}")
            total = sum(dist.values())
            self.prior[target] = {k: v / total for k, v in dist.items()}
        self._cache: dict[tuple, tuple[dict[str, float], bool]] = {}

    def _load_margins(self, era: Path) -> None:
        t1 = pd.read_parquet(era / _TABLE_FILES["T1"])
        keep = set(KEYS) | set(TARGETS)
        for r in t1.itertuples(index=False):
            if r.anchor in keep:
                self.t1[(str(r.anchor), str(r.value))] = float(r.share)

    def _load_conditionals(self, era: Path) -> None:
        t2 = pd.read_parquet(era / _TABLE_FILES["T2"])
        t3 = pd.read_parquet(era / _TABLE_FILES["T3"])
        t4 = pd.read_parquet(era / _TABLE_FILES["T4"])
        t5 = pd.read_parquet(era / _TABLE_FILES["T5"])
        for r in t2.itertuples(index=False):
            if r.anchor in TARGETS:
                self.t2[(str(r.anchor), str(r.age_band), str(r.value))] = float(r.share)
        for r in t3.itertuples(index=False):
            if r.anchor in TARGETS:
                self.t3[(str(r.anchor), str(r.sex), str(r.value))] = float(r.share)
        for r in t4.itertuples(index=False):
            if r.anchor in TARGETS:
                self.t4[(str(r.anchor), str(r.living_area), str(r.value))] = float(r.share)
        for r in t5.itertuples(index=False):
            if r.anchor in TARGETS:
                self.t5[(str(r.anchor), str(r.age_band), str(r.sex), str(r.value))] = float(r.share)

    def _load_joints(self, era: Path) -> None:
        t6 = pd.read_parquet(era / _TABLE_FILES["T6"])
        for r in t6.itertuples(index=False):
            a, va, b, vb = str(r.anchor_a), str(r.value_a), str(r.anchor_b), str(r.value_b)
            self.joint[(a, va, b, vb)] = float(r.joint)
            self.joint[(b, vb, a, va)] = float(r.joint)

    def _spec(
        self, node: dict, known: dict[str, str], *, marginal_only: bool = False
    ) -> tuple[dict[str, float], list[str]]:
        for level in node["levels"]:
            parents = list(level["parents"])
            if marginal_only and parents:
                continue
            if any(p not in known for p in parents):
                continue
            key = "|".join(known[p] for p in parents) if parents else ""
            entry = level["table"].get(key)
            if entry is None:
                continue
            weights = [float(v) for v in entry["weights"]]
            total = sum(weights)
            if total <= 0:
                continue
            dist = {str(v): w / total for v, w in zip(entry["values"], weights, strict=True)}
            return dist, parents
        raise RuntimeError(f"spec has no marginal cell for {node.get('name')}")

    def admissible_parents(self, target: str) -> list[str]:
        """Longest spec parents that fit inside the five known keys."""
        node = self.nodes.get(target)
        if node is None:
            return []
        known = set(KEYS)
        for level in node["levels"]:
            parents = list(level["parents"])
            if all(p in known for p in parents):
                return parents
        return []

    def _support(self, target: str, base: dict[str, float]) -> list[str]:
        return list(dict.fromkeys([*self.prior[target], *base]))

    def _add_lr(self, logp: dict[str, float], cls: str, lr: float) -> None:
        if lr == 1.0:
            return
        logp[cls] = _LOG0 if lr <= 0 else logp[cls] + math.log(lr)

    def _lr(self, target: str, feature: str, value: str, cls: str) -> float:
        pt = self.prior[target].get(cls, 0.0)
        if pt <= 0:
            return 1.0
        if feature in UNIVARIATE:
            table = {"sex": self.t3, "age_band": self.t2, "living_area": self.t4}[feature]
            share = table.get((target, value, cls))
            if share is None:
                return 1.0
            return share / pt
        if feature in PAIR_KEYS:
            joint = self.joint.get((feature, value, target, cls))
            pf = self.t1.get((feature, value), 0.0)
            if joint is None or pf <= 0:
                return 1.0
            return joint / (pt * pf)
        return 1.0

    def _add_t5(self, logp: dict[str, float], target: str, age: str, sex: str) -> bool:
        shares = {c: self.t5.get((target, age, sex, c)) for c in logp}
        if all(v is None for v in shares.values()):
            return False
        for cls, share in shares.items():
            if share is None:
                continue
            pt = self.prior[target].get(cls, 0.0)
            if pt <= 0:
                continue
            self._add_lr(logp, cls, share / pt)
        return True

    def posterior(self, target: str, known: dict[str, str]) -> tuple[dict[str, float], bool]:
        key = (target, *(known[k] for k in KEYS))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        node = self.nodes.get(target)
        if node is None:
            base, used, admissible = dict(self.prior[target]), [], []
        else:
            base, used = self._spec(node, known)
            admissible = self.admissible_parents(target)
        hit = used == admissible
        logp = {
            c: (math.log(base[c]) if base.get(c, 0.0) > 0 else _LOG0)
            for c in self._support(target, base)
        }
        remaining = [k for k in KEYS if k not in used]
        use_t5 = (
            "age_band" in remaining
            and "sex" in remaining
            and self._add_t5(logp, target, known["age_band"], known["sex"])
        )
        if use_t5:
            remaining = [k for k in remaining if k not in ("age_band", "sex")]
        for feature in remaining:
            for cls in logp:
                self._add_lr(logp, cls, self._lr(target, feature, known[feature], cls))
        out = (_softmax(logp), hit)
        self._cache[key] = out
        return out


def _complete(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.dropna(subset=[*KEYS, target, "w"]).copy()
    out = out[out["w"] > 0]
    for col in [*KEYS, target]:
        out[col] = out[col].astype(str)
    return out.reset_index(drop=True)


def _bank_path(wave: int) -> Path:
    return settings.data_root / "release" / RELEASE_VERSION / f"personas_{wave}.parquet"


def _classifier_accuracy(
    bank: pd.DataFrame, real: pd.DataFrame, target: str, w: np.ndarray
) -> float:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    yb = bank[target].to_numpy()
    clf.fit(enc.fit_transform(bank[KEYS]), yb)
    pred = clf.predict(enc.transform(real[KEYS]))
    return weighted_mean(pred == real[target].to_numpy(), w)


def _table_scores(
    model: ReleasedTables, real: pd.DataFrame, target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = real[target].to_numpy()
    pred = np.empty(len(real), dtype=object)
    p_true = np.empty(len(real), dtype=float)
    hit = np.empty(len(real), dtype=float)
    for key, part in real.groupby(KEYS, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        known = dict(zip(KEYS, (str(v) for v in key), strict=True))
        post, matched = model.posterior(target, known)
        choice = _mode(post)
        idx = part.index.to_numpy()
        pred[idx] = choice
        hit[idx] = float(matched)
        p_true[idx] = [post.get(v, 0.0) for v in y[idx]]
    return pred, p_true, hit


def evaluate_target(
    wave: int, target: str, real: pd.DataFrame, bank: pd.DataFrame, model: ReleasedTables
) -> list[dict]:
    real_t = _complete(real, target)
    bank_t = bank.dropna(subset=[*KEYS, target]).copy()
    for col in [*KEYS, target]:
        bank_t[col] = bank_t[col].astype(str)
    bank_t = bank_t.reset_index(drop=True)
    if real_t.empty or bank_t.empty:
        raise SystemExit(f"wave {wave} {target}: no complete rows")

    y = real_t[target].to_numpy()
    w = real_t["w"].to_numpy(dtype=float)
    coverage = float(np.mean([v in model.prior[target] for v in y]))
    if coverage < 0.99:
        log.warning(
            "wave %s %s fraction of target labels absent from the released marginal %.3f",
            wave,
            target,
            1 - coverage,
        )

    dist = model.prior[target]
    mode = _mode(dist)
    acc_m = weighted_mean(y == mode, w)
    cap_m = weighted_mean(np.array([dist.get(v, 0.0) for v in y]), w)

    pred_t, p_true, hit = _table_scores(model, real_t, target)
    acc_t = weighted_mean(pred_t == y, w)
    cap_t = weighted_mean(p_true, w)
    hit_rate = weighted_mean(hit, w)

    acc_b = _classifier_accuracy(bank_t, real_t, target, w)
    real_keys = _join_keys(real_t)
    bank_keys = _join_keys(bank_t)
    cap_rows = correct_attribution_probability(real_keys, y, bank_keys, bank_t[target].to_numpy())
    cap_b = weighted_mean(cap_rows, w)
    counts = pd.Series(bank_keys).value_counts()
    cell_n = pd.Series(real_keys).map(counts).fillna(0).to_numpy(dtype=float)
    matched = cell_n > 0
    match_rate = weighted_mean(matched.astype(float), w)
    mean_cell = weighted_mean(cell_n[matched], w[matched]) if matched.any() else 0.0

    parents = "+".join(model.admissible_parents(target)) or "(marginal)"
    common = {
        "wave": wave,
        "target": target,
        "n_real": len(real_t),
        "weight_sum": float(w.sum()),
        "n_bank": len(bank_t),
        "table_parents": parents,
        "table_cell_hit_rate": hit_rate,
        "bank_key_match_rate": match_rate,
        "bank_mean_cell_n": mean_cell,
        "bank_minus_tables_acc": acc_b - acc_t,
        "bank_minus_tables_cap": cap_b - cap_t,
    }
    scores = {"marginal": (acc_m, cap_m), "tables": (acc_t, cap_t), "bank": (acc_b, cap_b)}
    rows = []
    for name in PREDICTORS:
        acc, cap = scores[name]
        rows.append(
            {
                **common,
                "predictor": name,
                "accuracy": acc,
                "acc_lift": acc - acc_m,
                "cap": cap,
                "cap_lift": cap - cap_m,
            }
        )
    log.info(
        "wave %s %-24s acc m/t/b %.3f/%.3f/%.3f  cap m/t/b %.3f/%.3f/%.3f  d_cap %+.3f",
        wave,
        target,
        acc_m,
        acc_t,
        acc_b,
        cap_m,
        cap_t,
        cap_b,
        cap_b - cap_t,
    )
    return rows


def run_wave(wave: int) -> pd.DataFrame:
    path = _bank_path(wave)
    if not path.exists():
        raise SystemExit(f"released bank missing: {path}")
    log.info("wave %s real-respondent frame", wave)
    real = add_derived_columns(build_official_anchor_frame(wave))
    bank = pd.read_parquet(path, columns=[*KEYS, *TARGETS])
    model = ReleasedTables(wave)
    rows: list[dict] = []
    for target in TARGETS:
        rows.extend(evaluate_target(wave, target, real, bank, model))
    return pd.DataFrame(rows)


def _snap(v: float) -> float:
    if abs(v) < 1e-12:
        return 0.0
    if abs(v - 1.0) < 1e-12:
        return 1.0
    return v


def _fmt(frame: pd.DataFrame) -> str:
    show = frame.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(_snap)
    return show.to_markdown(index=False, floatfmt=".3f")


def _headline(res: pd.DataFrame) -> str:
    bits = []
    for target in TARGETS:
        sub = res[(res.wave == 2024) & (res.target == target)]
        d_cap = float(sub.bank_minus_tables_cap.iloc[0])
        d_acc = float(sub.bank_minus_tables_acc.iloc[0])
        bits.append(f"{target} CAP {d_cap:+.3f} (정확도 {d_acc:+.3f})")
    return "2024 공개 뱅크 − 공개 표: " + ", ".join(bits) + "."


def write_report(res: pd.DataFrame, reports: Path) -> None:
    for col in res.columns:
        if pd.api.types.is_float_dtype(res[col]):
            res[col] = res[col].map(_snap)
    res.to_csv(reports / "era_attribute_disclosure.csv", index=False)
    delta = (
        res[res.predictor == "marginal"][
            [
                "wave",
                "target",
                "n_real",
                "table_parents",
                "table_cell_hit_rate",
                "bank_key_match_rate",
                "bank_mean_cell_n",
                "bank_minus_tables_acc",
                "bank_minus_tables_cap",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )
    wide_rows = []
    for (wave, target), g in res.groupby(["wave", "target"], sort=False):
        rec = {"wave": int(wave), "target": target}
        for _, row in g.iterrows():
            rec[f"acc_{row.predictor}"] = row.accuracy
            rec[f"cap_{row.predictor}"] = row.cap
            rec[f"acc_lift_{row.predictor}"] = row.acc_lift
            rec[f"cap_lift_{row.predictor}"] = row.cap_lift
        wide_rows.append(rec)
    wide = pd.DataFrame(wide_rows)
    lines = [
        "# 실제 응답자 속성 노출",
        "",
        "공격자가 실제 응답자의 성·연령대·생활권·학력·직업을 알 때, 가구소득 구간·",
        "관람 여부·장르 수 구간을 공개 표와 공개 양적 뱅크로 맞힌 가중 집계다.",
        "개인 행은 저장하지 않았다. 근거 대상은 서울 거주 온라인 패널의 가중 응답이다.",
        "표의 조건부는 이 응답자로 만든 공개 집계(칸 하한 30)이고, 뱅크는 그 표에서 뽑은 합성 행이다.",
        "",
        _headline(res),
        "양수는 공개 뱅크가 공개 표보다 그 표적을 더 맞힌다는 뜻이다.",
        "",
        "## 뱅크 − 표",
        "",
        _fmt(delta),
        "",
        "## 예측기별 정확도·CAP",
        "",
        _fmt(wide),
        "",
        "## 정의",
        "",
        "- `marginal`: 공개 T1 다수 범주(정확도)와, T1 이 실제 표적에 부여한 확률(CAP).",
        "  장르 수 구간은 T1 에 없어 공개 샘플러 스펙의 주변분포를 쓴다.",
        "- `tables`: 스펙에서 부모가 전부 알려진 가장 긴 조건부. 소득은 연령+직업,",
        "  관람은 연령+학력(소득 부모는 공격자가 모른다), 장르 수는 부모가 관람 여부라",
        "  주변분포만 남는다. 남은 키는 T2–T5·T6 공개판으로 나이브 베이즈 갱신하고,",
        "  억제된 칸은 우도비 1. CAP 는 이 확률이 실제 표적에 부여한 값의 가중 평균.",
        "  `table_cell_hit_rate` 는 그 최장 칸이 스펙에 있어 백오프하지 않은 가중 비율.",
        "- `bank` 정확도: 공개 뱅크(`data/release/"
        + RELEASE_VERSION
        + "/personas_{wave}.parquet`)의",
        "  다섯 키만으로 학습한 로지스틱 회귀. 서사 열은 쓰지 않는다.",
        "- `bank` CAP: 같은 다섯 키 뱅크 행에서 실제 표적값이 차지하는 비율의 가중 평균.",
        "  키가 없으면 뱅크 주변비율. `bank_mean_cell_n` 은 키가 맞은 응답자의",
        "  가중 평균 칸 크기(뱅크 행 수).",
        "- `acc_lift`·`cap_lift` 는 같은 표적의 marginal 대비. 뱅크 행은 각 1, 응답자는 조사 가중.",
        "",
    ]
    (reports / "era_attribute_disclosure.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int, action="append", default=None)
    args = ap.parse_args()
    waves = args.wave or list(WEIGHTED_WAVES)
    frames = [run_wave(w) for w in waves]
    res = pd.concat(frames, ignore_index=True)
    res["target"] = pd.Categorical(res["target"], TARGETS, ordered=True)
    res["predictor"] = pd.Categorical(res["predictor"], PREDICTORS, ordered=True)
    res = res.sort_values(["wave", "target", "predictor"]).reset_index(drop=True)
    res["target"] = res["target"].astype(str)
    res["predictor"] = res["predictor"].astype(str)
    reports = settings.outputs_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_report(res, reports)
    log.info("%s", _headline(res))
    log.info("wrote %s", reports / "era_attribute_disclosure.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
