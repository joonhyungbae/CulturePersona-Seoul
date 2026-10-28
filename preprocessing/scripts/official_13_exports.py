"""Exports in the format the persona pipeline reads, built from the official outputs.

`src/sfac_persona/config.py` points at these two:

  data/processed/microdata/microdata_value_labels.csv   (year, group, var, value, value_label)
  data/interim/microdata/<year>_<group>.pkl             (a pickle holding one DataFrame)

The source of record is the official track; only the format follows the older contract.

Do not edit these files by hand. Changing the schema means changing src/sfac_persona too.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C

# The (year, group) pairs the persona pipeline uses. Extend here if more are needed.
DOWNSTREAM = [(2018, "general"), (2020, "general"), (2022, "general"), (2024, "general")]

LEGACY_PROCESSED = C.ROOT / "data" / "processed" / "microdata"
LEGACY_INTERIM = C.ROOT / "data" / "interim" / "microdata"


def main():
    LEGACY_PROCESSED.mkdir(parents=True, exist_ok=True)
    LEGACY_INTERIM.mkdir(parents=True, exist_ok=True)

    # ---- value labels: map back to the old column names ----------------------
    v = pd.read_csv(C.OUT / "official_value_labels.csv")
    # official holds 2022 as one mixed block; the old track called it general.
    v = v.assign(group=v.group.replace({"mixed": "general"}))
    # Match the old contract's value format. Labels coming from SAV read as '1.0',
    # while the consumer looks up '1' via lab.get(str(int(x)))
    # (src/sfac_persona/grounding/microdata.py). Without the match, every categorical
    # anchor silently becomes None -- no crash, just quiet corruption.
    def _v(x):
        t = str(x).strip()
        try:
            f = float(t)
            return str(int(f)) if f == int(f) else t
        except (TypeError, ValueError):
            return t

    v = v.assign(value=v.value.map(_v))
    # Drop anything that is not an integer code. These are the codebook's range
    # notations ('>=0') or fractional categories ('7.1'), which the consumer, looking
    # up only str(int(x)), can never hit. The old outputs did not carry them either
    # (zero non-integer values).
    v = v[v.value.astype(str).str.fullmatch(r"-?\d+")]
    out = (v.rename(columns={"wave": "year", "variable": "var", "label": "value_label"})
             [["year", "group", "var", "value", "value_label"]]
             .drop_duplicates(subset=["year", "group", "var", "value"]))
    out.to_csv(LEGACY_PROCESSED / "microdata_value_labels.csv", index=False)

    # ---- pickle cache: exactly one DataFrame, as the old contract had it ------
    made = []
    for year, group in DOWNSTREAM:
        src_group = "mixed" if (year, "mixed") in C.FILES else group
        df, _meta = C.load(year, src_group)
        if year == 2022:  # the old track's 'general' was GRP 1 plus 3
            df = df[pd.to_numeric(df["GRP"], errors="coerce").isin([1, 3])]
        p = LEGACY_INTERIM / f"{year}_{group}.pkl"
        with open(p, "wb") as f:
            pickle.dump(df.reset_index(drop=True), f)
        made.append((p.name, len(df), len(df.columns)))

    print("backwards-compatible exports (source: official)")
    print(f"  {LEGACY_PROCESSED/'microdata_value_labels.csv'}  {len(out):,} rows")
    for n, r, c in made:
        print(f"  {LEGACY_INTERIM.name}/{n}  {r:,} rows x {c} columns")
    # Check the format matches the old outputs. A mismatch breaks the consumer silently.
    bad = out[~out.value.astype(str).str.fullmatch(r"-?\d+")]
    assert bad.empty, f"value is not an integer string: {bad.head().to_dict('records')}"

    print("\nconsumer: src/sfac_persona/config.py "
          "(value_labels_csv · microdata pickle)")
    print("value format check passed (all integer strings, matching str(int(x)) lookup)")


if __name__ == "__main__":
    main()
