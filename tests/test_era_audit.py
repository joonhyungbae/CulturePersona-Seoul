"""Regression tests for the fixes from the 2026-09-11 audit (ADR-012).

The era holdout/PPI path had no tests, and four of the five defects here
would have been caught by simple assertions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


ehr = _load("era_holdout_run")
ppi = _load("era_holdout_ppi")


# ── D. Likert anchors matched to scale length ──────────────────────────────


@pytest.mark.parametrize("k", [4, 5, 7])
def test_likert_scale_covers_its_own_top(k: int) -> None:
    """The top of a k-point scale must be worded with "매우" ("very").

    The previous version truncated 7-point sentences, so on the 2020 1–5 scale
    5 became "다소 높음" ("somewhat high"), and on the 1–4 scale 4 became
    "보통" ("moderate") — SSR structurally could not match the top.
    """
    scale = ehr.LIKERT_REFS_BY_K[k]
    assert sorted(scale) == [str(i) for i in range(1, k + 1)]
    assert any("매우 그렇다" in s for s in scale[str(k)])
    assert any("전혀" in s for s in scale["1"])


def test_likert_scales_are_distinct_per_length() -> None:
    """If a 5-point '5' and a 7-point '5' share a sentence, the scales are not aligned."""
    assert ehr.LIKERT_REFS_BY_K[5]["5"] != ehr.LIKERT_REFS_BY_K[7]["5"]
    assert ehr.LIKERT_REFS_BY_K[4]["4"] != ehr.LIKERT_REFS_BY_K[7]["4"]


def test_interpolated_scale_spans_the_ends() -> None:
    out = ehr._interpolated_scale([1, 3, 5, 9])
    assert any("전혀" in s for s in out["1"])
    assert any("매우 그렇다" in s for s in out["9"])


# ── F/G. PPI ρ² ───────────────────────────────────────────────────────────


def test_rho2_is_one_for_a_perfect_predictor() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=400)
    Y = np.stack([(y == o).astype(float) for o in range(3)], axis=1)
    m = ppi.ppi_rho2(Y, Y.copy(), np.ones(len(Y)), n_boot=20)
    assert m["rho2"] == pytest.approx(1.0, abs=1e-9)
    assert m["ratio_pp"] == pytest.approx(0.0, abs=1e-9)


def test_rho2_null_level_matches_sample_size() -> None:
    n = 300
    rng = np.random.default_rng(1)
    Y = np.stack([(rng.integers(0, 3, n) == o).astype(float) for o in range(3)], axis=1)
    F = rng.random((n, 3))
    F /= F.sum(1, keepdims=True)
    m = ppi.ppi_rho2(Y, F, np.ones(n), n_boot=200)
    assert m["rho2_null"] == pytest.approx(1 / (n - 2))
    assert m["rho2_lo"] <= m["rho2_hi"]
    # With no signal, the CI lower bound must not exceed chance (one-sided selection).
    assert not m["above_null"]
    # ρ² meets a boundary at 0, so the CI **may not** contain the point estimate.
    # That is a property, not a defect, which is why the CI is not used as an error bar.
    assert m["rho2_hi"] < 0.05


def test_rect_difficulty_column_is_gone() -> None:
    """At the optimal λ, Var(Y−λF)/Var(Y) ≡ 1−ρ², so the column had no information."""
    rng = np.random.default_rng(2)
    Y = np.stack([(rng.integers(0, 2, 100) == o).astype(float) for o in range(2)], axis=1)
    m = ppi.ppi_rho2(Y, rng.random((100, 2)), np.ones(100), n_boot=5)
    assert "rect_difficulty" not in m


# ── C. Depth is measured with the same ruler ───────────────────────────────


def test_depth_proxy_is_a_pure_band_map() -> None:
    from sfac_persona.grounding.official import depth_proxy_from_band
    from sfac_persona.schema import ATTEND_COUNT_BAND_MIDPOINTS

    s = pd.Series(list(ATTEND_COUNT_BAND_MIDPOINTS) + ["없는구간"])
    got = depth_proxy_from_band(s)
    assert list(got[:-1]) == list(ATTEND_COUNT_BAND_MIDPOINTS.values())
    assert np.isnan(got.iloc[-1])


def test_binned_proxy_differs_from_raw_counts() -> None:
    """This gap is the discretization bias. Zero means this test's premise is broken."""
    from sfac_persona.grounding.official import depth_proxy_from_band

    band = pd.Series(["11회 이상"] * 10)
    raw = pd.Series([19.0] * 10)
    assert abs(depth_proxy_from_band(band).mean() - raw.mean()) > 1.0


# ── J. Positional join guard ───────────────────────────────────────────────


def test_attach_row_uid_rejects_length_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ehr, "load_general_master", lambda *a, **k: pd.DataFrame({"row_uid": ["a", "b"]})
    )
    with pytest.raises(SystemExit):
        ehr._attach_row_uid(pd.DataFrame({"w": [1.0]}), 2024)


# ── Helpers ────────────────────────────────────────────────────────────────


def test_tvd_is_symmetric_and_bounded() -> None:
    p = pd.Series({"a": 0.7, "b": 0.3})
    q = pd.Series({"a": 0.2, "c": 0.8})
    assert ehr._tvd(p, q) == pytest.approx(ehr._tvd(q, p))
    assert 0.0 <= ehr._tvd(p, q) <= 1.0
    assert ehr._tvd(p, p) == pytest.approx(0.0)


# ── Resume key (2026-09-12) ────────────────────────────────────────────────


def test_quant_dicts_returns_ids_and_sampling_depends_on_n(tmp_path: Path) -> None:
    """If `n` is below the full size, sampling happens, so an index cannot name a row.

    That is why position-based resume is forbidden: continuing an n=300 checkpoint
    at n=15000 makes the same index mean a different row.
    """
    import pandas as pd

    from sfac_persona.generate import _quant_dicts_from_parquet

    df = pd.DataFrame(
        {
            "persona_id": [f"p-{i:05d}" for i in range(50)],
            "age_band": ["20대"] * 50,
            "sex": ["남성"] * 50,
        }
    )
    path = tmp_path / "q.parquet"
    df.to_parquet(path, index=False)

    _, ids_all = _quant_dicts_from_parquet(path, 50, 1)
    _, ids_sub = _quant_dicts_from_parquet(path, 10, 1)
    assert ids_all[:10] == [f"p-{i:05d}" for i in range(10)]
    assert len(ids_sub) == 10
    assert ids_sub != ids_all[:10], "sample matching original order breaks this test's premise"


def test_narrative_records_keep_opaque_id_separate() -> None:
    """`persona_id` is an opaque UUID — overwriting it with a quant row id
    breaks the public contract.
    """
    import json as _json
    import uuid as _uuid

    bank = ROOT / "data" / "eval" / "era_holdout" / "w2024" / "eval_bank.jsonl"
    if not bank.exists():
        pytest.skip("eval bank missing")
    rec = _json.loads(bank.read_text(encoding="utf-8").splitlines()[0])
    _uuid.UUID(str(rec["persona_id"]))  # fails here unless it is a UUID
    if "quant_id" in rec:
        assert rec["quant_id"] != rec["persona_id"]
