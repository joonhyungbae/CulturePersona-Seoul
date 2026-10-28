"""How often the narratives state a marital or child situation that no anchor carries.

Marital status and co-resident children are not anchors: marital status is coded differently across the weighted
waves and the child item exists in 2024 only (docs/anchor_tiers.csv), so neither can enter every wave's bank. The
generation prompt tells the model to invent what the anchors do not give, without contradicting them. This script
measures how often that invention reaches family status, so the Limitations can state it with a number.

Design fixed before running (2026-10-12):
  data      every narrative persona of the released banks, 2020, 2022 and 2024 (personas.jsonl), all 12 fields
  measure   share of personas whose fields mention marriage, a spouse or a child, by a Korean lexicon:
            결혼, 기혼, 미혼, 이혼, 사별, 배우자, 남편, 아내, 부부, 자녀, 아들, 딸, 육아, 손주,
            and 아이 as a whole word or with a particle (아이가/아이와/아이들 …), excluding 아이돌/아이디어/아이템
  check     40 random matches printed for a manual precision check (not stored); per-wave shares and the share
            among single-person households reported
Addition after the manual precision check (2026-10-12): the printed matches showed that some fields only say the
  anchors leave family status open ("혼인 여부나 자녀 유무는 명시되지 않았다"), so the report also counts personas whose
  every family sentence is such a hedge (lexicon: 명시되지 않 / 명확하지 않 / 알 수 없 / 구체적이지 않 / 추정 / 관계없이 /
  여부나 / 유무에 따라 / 유무와), and the share in the family-context field, whose prompt asks for "혼인·동거 자녀·가구가
  문화생활에 미치는 영향" although no anchor carries marital status or children.
Every result is reported, whatever its value.

Run: uv run python scripts/era_family_mentions.py
Output: outputs/reports/era_family_mentions.csv (aggregates only)
"""

from __future__ import annotations

import ast
import json
import logging
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("era_family_mentions")

LEX = re.compile(
    r"결혼|기혼|미혼|이혼|사별|배우자|남편|아내|부부|자녀|아들|딸(?!기)|육아|손주"
    r"|아이(?=$|[\s,.들가와를는의랑도에])(?!돌|디어|템)"
)


HEDGE = re.compile(
    r"명시되지 않|명확하지 않|알 수 없|구체적이지 않|추정|관계없이|여부나|유무에 따라|유무와"
)


def _parse(v):
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return ast.literal_eval(v)


def main() -> int:
    rows, examples = [], []
    for wave in (2020, 2022, 2024):
        path = settings.synthetic_dir / f"era_{wave}_narrative" / "personas.jsonl"
        n = hit = single = single_hit = hedge_only = field_hit = 0
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            narr, quant = _parse(r["narrative"]), _parse(r["quant"])
            text = " ".join(str(v) for v in narr.values())
            m = LEX.search(text)
            n += 1
            is_single = str(quant.get("household_size_band", "")).startswith("1")
            single += is_single
            if LEX.search(str(narr.get("family_context", ""))):
                field_hit += 1
            if m:
                sents = [s for s in re.split(r"(?<=[.다])\s", text) if LEX.search(s)]
                if sents and all(HEDGE.search(s) for s in sents):
                    hedge_only += 1
                hit += 1
                single_hit += is_single
                if len(examples) < 4000:
                    examples.append(text[max(0, m.start() - 25) : m.end() + 25])
        rows.append(
            {
                "wave": wave,
                "n": n,
                "share_family": hit / n,
                "share_hedge_only": hedge_only / n,
                "share_stated": (hit - hedge_only) / n,
                "share_family_field": field_hit / n,
                "n_single": single,
                "share_family_single": single_hit / single if single else float("nan"),
            }
        )
        log.info(
            "wave %d: %d personas, %.3f mention family; single-person %.3f",
            wave,
            n,
            hit / n,
            rows[-1]["share_family_single"],
        )
    total_n = sum(r["n"] for r in rows)
    agg = {
        k: sum(r[k] * r["n"] for r in rows) / total_n
        for k in ("share_family", "share_hedge_only", "share_stated", "share_family_field")
    }
    rows.append({"wave": "all", "n": total_n, **agg})
    out = pd.DataFrame(rows)
    out.to_csv(settings.outputs_dir / "reports" / "era_family_mentions.csv", index=False)
    print(out.round(3).to_string(index=False))
    random.seed(20261012)
    for e in random.sample(examples, min(40, len(examples))):
        print("  ..", e.replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
