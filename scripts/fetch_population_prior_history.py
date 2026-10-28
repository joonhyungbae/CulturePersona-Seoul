"""Fetch past months of resident-registration population (Seoul district × sex × age band) — to resolve ADR-011.

`data/reference/culture_nowcast_population_prior.csv` held only months from
2023-01 on, so the R0 of the 2020 and 2022 era banks used the latest month's
population (ADR-011). Those months are fetched directly.

**Dong-level rows are not read.** MOIS responses mix in dong rows, but only
district rows (codes ending in `000000`) are kept and the rest are dropped
immediately (release geography stops at the district).

Check: `--verify-month` fetches a month already in the cache and compares it
with the cache. On a mismatch, stop.

Run:
  uv run python scripts/fetch_population_prior_history.py --verify-month 2025-12
  uv run python scripts/fetch_population_prior_history.py --months 2020-12 2022-12 --append
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json as _json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sfac_persona.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("fetch_population")

BASE = "https://jumin.mois.go.kr"
PAGE = f"{BASE}/ageStatMonth.do"
CODES = f"{BASE}/selectHangkikcdListAjax.do"
SEOUL = "1100000000"
SOURCE_DATASET = "행정안전부 주민등록 인구통계 (연령별 인구현황, 시군구)"

# MOIS 21 five-year bands → project age bands
AGE5 = [
    "0-4",
    "5-9",
    "10-14",
    "15-19",
    "20-24",
    "25-29",
    "30-34",
    "35-39",
    "40-44",
    "45-49",
    "50-54",
    "55-59",
    "60-64",
    "65-69",
    "70-74",
    "75-79",
    "80-84",
    "85-89",
    "90-94",
    "95-99",
    "100+",
]
BAND_OF = {
    "15-19": ["15-19"],
    "20-29": ["20-24", "25-29"],
    "30-39": ["30-34", "35-39"],
    "40-49": ["40-44", "45-49"],
    "50-59": ["50-54", "55-59"],
    "60-69": ["60-64", "65-69"],
    "70+": ["70-74", "75-79", "80-84", "85-89", "90-94", "95-99", "100+"],
}
# Row layout: code, name, [total total/bands/21], [male total/bands/21], [female total/bands/21] = 71
OFFSET = {"total": 2, "male": 25, "female": 48}


class _Session:
    """Minimal session that POSTs with cookies. stdlib, so `requests` stays out of the dependencies."""

    def __init__(self) -> None:
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        self.opener.addheaders = [("User-Agent", "Mozilla/5.0"), ("Referer", PAGE)]
        self.get(PAGE)

    def get(self, url: str) -> str:
        with self.opener.open(url, timeout=60) as r:
            return r.read().decode("utf-8", "replace")

    def _reopen(self) -> None:
        """Drop cookies and open a new session. A run of 502s often means the session has gone stale."""
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        self.opener.addheaders = [("User-Agent", "Mozilla/5.0"), ("Referer", PAGE)]
        self.get(PAGE)

    def post(self, url: str, data: dict[str, str], timeout: int = 180, tries: int = 8) -> str:
        """MOIS itself says a query takes 10–30 seconds. Wait generously and retry.

        On a run of 502s, open a new session. A government site tends to fail quietly when it is tired.
        """
        body = urllib.parse.urlencode(data).encode()
        last: Exception | None = None
        for attempt in range(tries):
            try:
                with self.opener.open(url, data=body, timeout=timeout) as r:
                    return r.read().decode("utf-8", "replace")
            except Exception as exc:  # noqa: BLE001 — the network drops often
                last = exc
                wait = min(10 * (attempt + 1), 60)
                log.warning("retry %d/%d (%s) — wait %ds", attempt + 1, tries, exc, wait)
                time.sleep(wait)
                if attempt >= 1:
                    try:
                        self._reopen()
                        log.info("  session reopened")
                    except Exception as exc2:  # noqa: BLE001
                        log.warning("  session reopen failed: %s", exc2)
        raise SystemExit(f"POST failed: {url} — {last}")


def _session() -> _Session:
    return _Session()


def seoul_districts(s: _Session) -> list[tuple[str, str]]:
    payload = _json.loads(s.post(CODES, {"sltOrgLvl1": SEOUL}, timeout=60))
    out = []
    for loc in payload.get("locations", []):
        code = str(loc.get("hangkikcd", ""))
        if loc.get("levels") == "2" and code.startswith("11") and loc.get("sikunkunm"):
            out.append((code, loc["sikunkunm"]))
    if len(out) != 25:
        raise SystemExit(f"Seoul district count is not 25: {len(out)}")
    return sorted(out)


def _cells(row_html: str) -> list[str]:
    return [
        re.sub(r"<[^>]+>", "", c).replace("&nbsp;", "").replace(",", "").strip()
        for c in re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
    ]


def fetch_district(s: _Session, code: str, year: str, month: str) -> list[str]:
    """Return **only the one district-level row** for that district.

    Dong rows are dropped (release geography stops at the district).
    """
    data = {
        "sltOrgType": "2",
        "sltOrgLvl1": SEOUL,
        "sltOrgLvl2": code,
        "gender": "gender",
        "sum": "sum",
        "sltUndefType": "",
        "searchYearStart": year,
        "searchMonthStart": month,
        "searchYearEnd": year,
        "searchMonthEnd": month,
        "sltOrderType": "1",
        "sltOrderValue": "ASC",
        "sltArgTypes": "5",
        "sltArgTypeA": "0",
        "sltArgTypeB": "100",
        "category": "month",
        "searchYearMonth": "month",
    }
    html = s.post(PAGE, data)
    body = html[html.find('id="contextTable"') :]
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.DOTALL):
        tds = _cells(row)
        if tds and tds[0] == code:  # district row only. Dong rows (other codes) are not read.
            if len(tds) != 71:
                raise SystemExit(
                    f"{code} {year}-{month}: column count is not 71 ({len(tds)}) — the page layout changed"
                )
            return tds
    raise SystemExit(f"{code} {year}-{month}: district row not found")


def to_rows(tds: list[str], name: str, code: str, ym: str, fetched_from: str) -> list[dict]:
    rows = []
    for sex, off in (("male", OFFSET["male"]), ("female", OFFSET["female"])):
        ages = tds[off + 2 : off + 2 + len(AGE5)]
        vals = {a: int(v or 0) for a, v in zip(AGE5, ages, strict=True)}
        for band, parts in BAND_OF.items():
            rows.append(
                {
                    "source_dataset": SOURCE_DATASET,
                    "source_url": f"{BASE}/ageStatMonth.do",
                    "reference_month": ym,
                    "district_code": int(code),
                    "district_name": name,
                    "sex": sex,
                    "age_band": band,
                    "population_count": sum(vals[p] for p in parts),
                    "fetched_from": fetched_from,
                }
            )
    return rows


def fetch_month(
    s: _Session, ym: str, districts: list[tuple[str, str]], tag: str, cache_dir: Path | None = None
) -> pd.DataFrame:
    """One month. Cached per district, so a rerun resumes after an interruption."""
    year, month = ym.split("-")
    cache = (cache_dir or Path(".")) / f"pop_{ym}.partial.csv"
    done: dict[str, pd.DataFrame] = {}
    if cache.exists():
        prev = pd.read_csv(cache)
        for code, g in prev.groupby("district_code"):
            done[str(code)] = g
        log.info("  reusing %d districts from cache (%s)", len(done), cache.name)
    rows: list[pd.DataFrame] = []
    for i, (code, name) in enumerate(districts, 1):
        if code in done:
            rows.append(done[code])
            continue
        tds = fetch_district(s, code, year, month)
        df = pd.DataFrame(to_rows(tds, name, code, ym, tag))
        rows.append(df)
        pd.concat(rows, ignore_index=True).to_csv(cache, index=False)
        log.info("  [%2d/25] %s %s", i, ym, name)
        time.sleep(1.5)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=[])
    ap.add_argument(
        "--verify-month", default=None, help="check the parser against a month already in the cache"
    )
    ap.add_argument(
        "--append", action="store_true", help="if the check passes, append to the reference CSV"
    )
    args = ap.parse_args()

    csv_path = settings.population_prior_csv
    cached = pd.read_csv(csv_path)
    s = _session()
    districts = seoul_districts(s)
    log.info("Seoul districts: %d", len(districts))

    if args.verify_month:
        ym = args.verify_month
        got = fetch_month(s, ym, districts, "verify", cache_dir=csv_path.parent)
        ref = cached[cached.reference_month == ym]
        if ref.empty:
            raise SystemExit(f"{ym} is not in the cache — cannot verify")
        key = ["district_code", "sex", "age_band"]
        # The cache also holds the Seoul total row (1100000000). R0 does not use
        # that row, so the comparison is only the 25 districts.
        m = got.merge(ref, on=key, suffixes=("_new", "_ref"))
        if len(m) != len(got):
            raise SystemExit(
                f"not all district rows are in the cache: got {len(got)} merged {len(m)}"
            )
        missing = set(map(tuple, ref[key].to_numpy())) - set(map(tuple, got[key].to_numpy()))
        extra_codes = {c for c, _, _ in missing}
        if extra_codes - {1100000000}:
            raise SystemExit(f"district codes only in the cache: {sorted(extra_codes)[:5]}")
        diff = m[m.population_count_new != m.population_count_ref]
        log.info(
            "verify %s: %d rows, %d mismatched · sum new=%d ref=%d",
            ym,
            len(m),
            len(diff),
            int(m.population_count_new.sum()),
            int(m.population_count_ref.sum()),
        )
        if len(diff):
            log.error("%s", diff.head(10).to_string())
            raise SystemExit("parser disagrees with the cache — past months are not fetched")
        log.info("parser check passed")

    if not args.months:
        return 0

    new = pd.concat(
        [
            fetch_month(s, ym, districts, "mois_live", cache_dir=csv_path.parent)
            for ym in args.months
        ],
        ignore_index=True,
    )
    log.info("fetched rows %d · months %s", len(new), sorted(new.reference_month.unique()))
    if not args.append:
        out = csv_path.parent / "population_prior_history_preview.csv"
        new.to_csv(out, index=False)
        log.info("preview only (--append not set) → %s", out)
        return 0

    keep = cached[~cached.reference_month.isin(set(new.reference_month))]
    merged = pd.concat([keep, new], ignore_index=True).sort_values(
        ["reference_month", "district_code", "sex", "age_band"]
    )
    backup = csv_path.with_suffix(".csv.bak")
    if not backup.exists():
        backup.write_bytes(csv_path.read_bytes())
        log.info("original backed up → %s", backup.name)
    merged.to_csv(csv_path, index=False)
    log.info(
        "updated %s: %d rows · %d months",
        csv_path.name,
        len(merged),
        merged.reference_month.nunique(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
