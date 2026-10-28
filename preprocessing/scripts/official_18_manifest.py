"""Output manifest and data dictionary -- what is in which file, and how they join.

Once the pipeline passed fifty files, the answer to "what is this file" was scattered
across script docstrings. Since the persona pipeline takes a copy of the finished
preprocessing, that description has to come out as an output in its own right.

This does for files what the registry (official_17) does for variables: **it holds a
declaration and checks it against reality.** A file on disk that is not declared stays
`declared=False` and the validator blocks it; a declaration with no file is
`exists=False`. So files cannot appear or disappear silently.

Hand-maintained inputs:
  config/output_manifest.csv   the per-file declaration (what a row is, what produces
                               it, its sensitivity)
  config/person_columns.csv    column descriptions for the person-level files -- these
                               get handed out, so the columns are documented too

Outputs:
  official_file_manifest.csv   file x (rows, columns, bytes, declared, in repository)
  official_column_profile.csv  file x column x (dtype, missing rate, distinct values,
                               example, description)
  official_join_map.csv        checks each declared join really meets, by match rate
  official_person_coverage.csv which (wave, group) each person-level file holds
  reports/official_data_dictionary.md   the four above, for a human reader

Why coverage is reported separately: person-level tables can hold different samples,
and **a file listing does not show that.** It is what caught the out-of-battery
harmonisation sweeping only one file per wave and leaving 10,011 people empty (resolved
2026-09-09).

`in_repo` is not guessed but decided by `git check-ignore`. The point is to make the
manifest recheck, every time, the premise that no person-level file enters the
repository (section 1-1).
"""
from __future__ import annotations

import re
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import official_00_common as C

ROOT = C.ROOT
REPORTS = C.REPORTS
md_table = C.md_table

# Layer -> the actual directories. The export layer spans two of them (the
# backwards-compatible exports).
LAYERS: dict[str, list[Path]] = {
    "processed": [C.OUT],
    "interim": [C.PERSON],
    "reference": [C.REFERENCE],
    "export": [ROOT / "data" / "processed" / "microdata",
               ROOT / "data" / "interim" / "microdata"],
}

# Large files are swept in chunks rather than loaded whole. The long person x item
# table is over 400 MB.
BIG_BYTES = 50 * 1024 * 1024
CHUNK = 500_000
# Stop counting distinct values above this -- they are not a value for any table.
UNIQ_CAP = 50_000

# What this script itself produces. Excluded from the disk scan and filled in by hand:
# at scan time they either do not exist yet (a cold start) or are stale, left from a
# previous run. bytes is left empty -- a manifest recording its own size changes that
# size again.
SELF_OUTPUTS = ("official_file_manifest.csv", "official_column_profile.csv",
                "official_join_map.csv", "official_person_coverage.csv")

# The declared joins. A match rate below 1 means a key has shifted -- which really
# happened, filtering on wave alone instead of (wave, source_group).
EDGES: list[tuple] = [
    ("interim", "official_person_master.parquet", ["row_uid"]),
    ("interim", "official_person_item.parquet", ["row_uid"]),
    ("interim", "official_person_item_imputed.parquet", ["row_uid"]),
    ("interim", "official_person_domain.parquet", ["row_uid"]),
    ("interim", "official_person_domain_imputed.parquet", ["row_uid"]),
    ("interim", "official_person_wavelocal.parquet", ["row_uid"]),
    ("interim", "official_person_text.parquet", ["row_uid"]),
]
EXTRA_EDGES: list[tuple] = [
    # from_layer, from_file, from_keys, to_layer, to_file, to_keys
    ("interim", "official_person_domain.parquet",
     ["wave", "source_group", "family", "canonical_item", "measure"],
     "processed", "official_domain_item_keys.csv",
     ["wave", "source_group", "family", "canonical_item", "measure"]),
    ("interim", "official_person_item.parquet", ["wave", "canonical_item"],
     "processed", "official_schema_coverage.csv", ["wave", "canonical_item"]),
    ("processed", "official_domain_declarations.csv", ["wave", "source_group", "variable"],
     "processed", "official_variable_registry.csv", ["wave", "source_group", "variable"]),
    ("processed", "official_variable_registry.csv", ["wave", "source_group", "variable"],
     "processed", "official_variable_dictionary.csv", ["wave", "group", "variable"]),
]


# ------------------------------------------------- checking the declarations

def to_regex(name: str) -> re.Pattern:
    """Turn a declaration pattern such as `<wave>_<group>.pkl` into a regex. An
    angle-bracketed name is one token."""
    parts = re.split(r"(<[^>]+>)", name)
    body = "".join(r"[^_./]+" if p.startswith("<") else re.escape(p) for p in parts)
    return re.compile(f"^{body}$")


def git_ignored(paths: list[Path]) -> dict[Path, bool]:
    """Decide repository membership with `git check-ignore`. Nothing is guessed."""
    if not paths:
        return {}
    rel = [os.path.relpath(p, ROOT) for p in paths]
    r = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                       input="\n".join(rel), capture_output=True, text=True)
    ign = {line.strip() for line in r.stdout.splitlines() if line.strip()}
    return {p: (s in ign) for p, s in zip(paths, rel)}


# ------------------------------------------------------------------ profiling

# The largest number of distinct values a column may have before its example is
# withheld. At or below this the column is schema vocabulary rather than anyone's
# answer (family, measure, unit, sex and the like).
VOCAB_MAX = 30


def profile_table(path: Path, safe: bool = False) -> tuple[int, list[dict]]:
    """Returns (row count, per-column profile). A CSV is swept in chunks, a Parquet one
    column at a time. Parquet reads column-wise, so even a large table holds only one
    column in memory at once.

    `safe` is used for person-level files. Taking the example from the first row puts
    one respondent's record into the repository as a row (it really did carry the
    person_id and attendance counts of 2014:mixed:0). Vocabulary columns instead take
    the first value in sort order, which detaches it from any row, and everything else
    is left empty."""
    acc: dict[str, dict] = {}
    n_rows = 0
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        cols = pq.ParquetFile(path).schema_arrow.names
        reader = (pd.read_parquet(path, columns=[c]) for c in cols)
    else:
        big = path.stat().st_size > BIG_BYTES
        reader = (pd.read_csv(path, chunksize=CHUNK, low_memory=False) if big
                  else [pd.read_csv(path, low_memory=False)])
    for chunk in reader:
        # Parquet arrives one column at a time, so count the rows once.
        if path.suffix != ".parquet" or not acc:
            n_rows += len(chunk)
        for col in chunk.columns:
            a = acc.setdefault(col, {"dtype": str(chunk[col].dtype), "n_missing": 0,
                                     "uniq": set(), "capped": False, "example": None})
            s = chunk[col]
            a["n_missing"] += int(s.isna().sum())
            if not a["capped"]:
                a["uniq"].update(s.dropna().unique().tolist())
                if len(a["uniq"]) > UNIQ_CAP:
                    a["capped"] = True
                    a["uniq"] = set()
            if a["example"] is None:
                nn = s.dropna()
                if len(nn):
                    a["example"] = nn.iloc[0]
    cols = []
    for col, a in acc.items():
        nu = None if a["capped"] else len(a["uniq"])
        ex = a["example"]
        if safe:
            ex = (sorted(a["uniq"], key=str)[0]
                  if nu is not None and 0 < nu <= VOCAB_MAX else None)
        cols.append({
            "column": col,
            "dtype": a["dtype"],
            "n_missing": a["n_missing"],
            "missing_pct": round(a["n_missing"] / n_rows * 100, 2) if n_rows else None,
            "n_unique": nu,
            "example": str(ex)[:60] if ex is not None else None,
        })
    return n_rows, cols


def profile_pickle(path: Path) -> tuple[int, list[dict]]:
    """A cache pkl is a (DataFrame, meta) tuple. No column profile is produced for it --
    it is raw data."""
    import pickle
    with open(path, "rb") as f:
        obj = pickle.load(f)
    df = obj[0] if isinstance(obj, tuple) else obj
    return (len(df), [{"column": "(raw columns)", "dtype": "-", "n_missing": None,
                       "missing_pct": None, "n_unique": len(df.columns),
                       "example": ", ".join(map(str, list(df.columns)[:4]))}])


# -------------------------------------------------------------- join checking

_KEYCACHE: dict[tuple, pd.Series] = {}


def key_counts(layer: str, fname: str, keys: list[str]) -> pd.Series:
    """Join key -> the number of rows carrying it. The string concatenation runs **over
    distinct combinations only**.

    Joining five columns across 5 million rows builds 5 million strings to end up with
    1,470 distinct ones. Folding with value_counts first and concatenating after gives
    the same answer at a tenth of the cost. (This spot once took seven minutes with a
    row-wise join; a vectorised concatenation fixed that, and this folds it once more.)
    """
    ck = (layer, fname, tuple(keys))
    if ck in _KEYCACHE:
        return _KEYCACHE[ck]
    path = next(d / fname for d in LAYERS[layer] if (d / fname).exists())
    df = (pd.read_parquet(path, columns=keys) if path.suffix == ".parquet"
          else pd.read_csv(path, usecols=keys, low_memory=False))
    vc = df.value_counts(subset=keys, sort=False, dropna=False)
    idx = vc.index.to_frame(index=False)
    out = idx[keys[0]].astype(str)
    for k in keys[1:]:
        out = out + "|" + idx[k].astype(str)
    vc = pd.Series(vc.values, index=out.values)
    _KEYCACHE[ck] = vc
    return vc


def check_edge(fl, ff, fk, tl, tf, tk) -> dict:
    lk = key_counts(fl, ff, fk)
    rk = key_counts(tl, tf, tk)
    lu, ru = set(lk.index), set(rk.index)
    matched = len(lu & ru)
    return {
        "from_file": ff, "from_keys": "+".join(fk),
        "to_file": tf, "to_keys": "+".join(tk),
        "n_from_rows": int(lk.sum()), "n_from_keys": len(lu), "n_to_keys": len(ru),
        "n_matched": matched,
        "match_rate": round(matched / len(lu), 6) if lu else None,
        "cardinality": f"{lk.max()}:{rk.max()}",
        "verdict": "ok" if lu <= ru else "unmatched_keys",
    }


# ------------------------------------------------------------------- coverage

def person_coverage(files: list[str]) -> pd.DataFrame:
    """Which (wave, group) each person-level file holds. row_uid is
    `{wave}:{group}:{i}`, so the key alone separates them without rereading the file."""
    def wg_of(idx) -> pd.Series:
        """Take just `{wave}:{group}` off a row_uid. Applied **to distinct keys only**:
        running rsplit over 5.7 million rows splits strings millions of times to obtain
        eleven values."""
        return pd.Series(idx).str.rsplit(":", n=1).str[0]

    base = key_counts("interim", "official_person_keys.parquet", ["row_uid"])
    frame_n = base.groupby(wg_of(base.index).values).sum()
    rows = []
    for f in files:
        vc = key_counts("interim", f, ["row_uid"])
        wg = wg_of(vc.index).values
        n_rows = vc.groupby(wg).sum()          # rows in that (wave, group)
        n_persons = vc.groupby(wg).size()      # people in that (wave, group)
        for wgk in frame_n.index:
            wave, group = wgk.split(":")
            rows.append({
                "file": f, "wave": int(wave), "source_group": group,
                "n_frame": int(frame_n[wgk]),
                "n_persons": int(n_persons.get(wgk, 0)),
                "n_rows": int(n_rows.get(wgk, 0)),
            })
    D = pd.DataFrame(rows)
    D["covered"] = D.n_persons > 0
    D["person_share"] = (D.n_persons / D.n_frame).round(4)
    return D.sort_values(["file", "wave", "source_group"], ignore_index=True)


# ------------------------------------------------------------------ rendering

def main():
    decl = pd.read_csv(ROOT / "config" / "output_manifest.csv")
    pcols = pd.read_csv(ROOT / "config" / "person_columns.csv")

    # ---- what is actually on disk ------------------------------------------
    found: list[dict] = []
    for layer, dirs in LAYERS.items():
        for d in dirs:
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.is_dir() or p.name.startswith(".") or p.name in SELF_OUTPUTS:
                    continue
                found.append({"layer": layer, "path": os.path.relpath(p, ROOT),
                              "file": p.name, "_p": p})

    ignored = git_ignored([f["_p"] for f in found])

    # Match against the declarations: a literal name first, then a pattern.
    dec_by_layer: dict[str, list] = {}
    for r in decl.itertuples(index=False):
        dec_by_layer.setdefault(r.layer, []).append((r, to_regex(r.file)))

    MAN, COLP = [], []
    for f in found:
        hit = None
        for r, rx in dec_by_layer.get(f["layer"], []):
            if r.file == f["file"] or rx.match(f["file"]):
                hit = r
                break
        sens = hit.sensitivity if hit is not None else None
        if f["file"].endswith((".csv", ".parquet")):
            n_rows, cols = profile_table(
                f["_p"], safe=sens in ("person_level", "free_text"))
        elif f["file"].endswith(".pkl"):
            n_rows, cols = profile_pickle(f["_p"])
        else:
            n_rows, cols = None, []
        MAN.append({
            "layer": f["layer"], "file": f["file"], "path": f["path"],
            "declared_as": hit.file if hit is not None else None,
            "declared": hit is not None,
            "grain": hit.grain if hit is not None else None,
            "produced_by": hit.produced_by if hit is not None else None,
            "sensitivity": sens,
            "description": hit.description if hit is not None else None,
            "rows": n_rows, "n_columns": len(cols) if cols else None,
            "bytes": f["_p"].stat().st_size,
            "in_repo": not ignored.get(f["_p"], False),
        })
        desc = dict(zip(pcols[pcols.file == f["file"]].column,
                        pcols[pcols.file == f["file"]].description))
        for c in cols:
            COLP.append({"layer": f["layer"], "file": f["file"], **c,
                         "description": desc.get(c["column"])})

    M = pd.DataFrame(MAN).sort_values(["layer", "file"], ignore_index=True)
    CP = pd.DataFrame(COLP)

    # Declared but not on disk. A pattern declaration may legitimately match nothing,
    # so only literals are checked.
    on_disk = {(r.layer, r.declared_as) for r in M.itertuples(index=False) if r.declared}
    missing = [r for r in decl.itertuples(index=False)
               if "<" not in r.file and r.file not in SELF_OUTPUTS
               and (r.layer, r.file) not in on_disk]
    for r in missing:
        M.loc[len(M)] = {"layer": r.layer, "file": r.file, "path": None,
                         "declared_as": r.file, "declared": True, "grain": r.grain,
                         "produced_by": r.produced_by, "sensitivity": r.sensitivity,
                         "description": r.description, "rows": None, "n_columns": None,
                         "bytes": None, "in_repo": None}
    M["exists"] = M.path.notna()

    # ---- join checks ---------------------------------------------------------
    edges = [(fl, ff, fk, "interim", "official_person_keys.parquet", ["row_uid"])
             for fl, ff, fk in EDGES] + EXTRA_EDGES
    J = pd.DataFrame([check_edge(*e) for e in edges])

    # ---- sample coverage of the person-level files ---------------------------
    PC = person_coverage([r.file for r in M.itertuples(index=False)
                          if r.layer == "interim" and r.exists
                          and r.sensitivity in ("person_level", "free_text")
                          and r.file != "official_person_keys.parquet"])

    # ---- this script's own outputs -----------------------------------------
    # Order matters. The column dictionary (CP) must take its own rows first so that
    # its row count is settled; only then can the manifest (M) record that count.
    frames = {"official_file_manifest.csv": M, "official_column_profile.csv": CP,
              "official_join_map.csv": J, "official_person_coverage.csv": PC}
    self_cols = {fn: list(fr.columns) for fn, fr in frames.items()}
    for fn, cols in self_cols.items():
        for c in cols:
            CP.loc[len(CP)] = {"layer": "processed", "file": fn, "column": c,
                               "dtype": "-", "n_missing": None, "missing_pct": None,
                               "n_unique": None, "example": None,
                               "description": None}
    n_self = {"official_file_manifest.csv": len(M) + len(SELF_OUTPUTS),
              "official_column_profile.csv": len(CP),
              "official_join_map.csv": len(J),
              "official_person_coverage.csv": len(PC)}
    for fn in SELF_OUTPUTS:
        d = decl[(decl.layer == "processed") & (decl.file == fn)].iloc[0]
        M.loc[len(M)] = {"layer": "processed", "file": fn,
                         "path": os.path.relpath(C.OUT / fn, ROOT),
                         "declared_as": fn, "declared": True, "grain": d.grain,
                         "produced_by": d.produced_by, "sensitivity": d.sensitivity,
                         "description": d.description, "rows": n_self[fn],
                         "n_columns": len(self_cols[fn]), "bytes": None,
                         "in_repo": True, "exists": True}

    C.OUT.mkdir(parents=True, exist_ok=True)
    M.to_csv(C.OUT / "official_file_manifest.csv", index=False)
    CP.to_csv(C.OUT / "official_column_profile.csv", index=False)
    J.to_csv(C.OUT / "official_join_map.csv", index=False)
    PC.to_csv(C.OUT / "official_person_coverage.csv", index=False)

    render(M, CP, J, PC)

    n_und = int((~M.declared).sum())
    gap = PC[(PC.file == "official_person_domain.parquet") & ~PC.covered].n_frame.sum()
    print(f"{len(M)} files ({n_und} undeclared · {int((~M.exists).sum())} missing) · "
          f"{len(CP)} columns · {len(J)} joins (lowest match rate "
          f"{J.match_rate.min():.4f}) · {int(gap):,} people with nothing outside the "
          "battery")


def render(M: pd.DataFrame, CP: pd.DataFrame, J: pd.DataFrame, PC: pd.DataFrame):
    def human(b):
        if pd.isna(b):
            return ""
        b = float(b)
        for u in ("B", "KB", "MB", "GB"):
            if b < 1024 or u == "GB":
                return f"{b:.0f}{u}" if u == "B" else f"{b:.1f}{u}"
            b /= 1024

    person = M[(M.layer == "interim") & M.sensitivity.isin(["person_level", "free_text"])
               & M.exists].sort_values("file")
    agg = M[(M.layer == "processed") & M.exists].sort_values("file")
    ref = M[M.layer.isin(["reference", "export"]) & M.exists].sort_values(["layer", "file"])
    cache = M[(M.layer == "interim") & (M.sensitivity == "raw_cache") & M.exists]

    person_tbl = person.assign(**{"rows": person.rows.map("{:,.0f}".format),
                                  "columns": person.n_columns.astype(int),
                                  "size": person.bytes.map(human)})
    agg_tbl = agg.assign(**{"rows": agg.rows.map("{:,.0f}".format)})
    ref_tbl = ref.assign(**{"rows": ref.rows.map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")})

    piv = PC.pivot_table(index=["wave", "source_group"], columns="file",
                         values="n_persons", aggfunc="first").fillna(0).astype(int)
    piv.columns = [c.replace("official_person_", "").replace(".csv", "") for c in piv.columns]
    frame_n = PC.groupby(["wave", "source_group"]).n_frame.first()
    cov_md = md_table(piv.assign(**{"sample": frame_n}).reset_index()
                      .rename(columns={"wave": "wave", "source_group": "group"}))
    _rd = J[(J.from_file == "official_variable_registry.csv")]
    n_reg = int(_rd.n_from_keys.iloc[0]) if len(_rd) else 0
    n_nocb = n_reg - int(_rd.n_matched.iloc[0]) if len(_rd) else 0
    miss = PC[(PC.file == "official_person_domain.parquet") & ~PC.covered]
    gap_n = int(miss.n_frame.sum())
    if len(miss):
        gap_who = " · ".join(f"{r.wave} {r.source_group} {r.n_frame:,}"
                             for r in miss.itertuples(index=False))
        gap_para = (f"**{gap_n:,} people** have no out-of-battery variables at all "
                    f"({gap_who}). They still have their keys and the attendance "
                    "battery.")
    else:
        gap_para = ("**No gap** -- the out-of-battery harmonisation sweeps all "
                    f"{len(C.groups())} raw files. Only the open text is confined to "
                    "the waves that asked it.")

    _txt = person[person.file.str.startswith("official_person_text")]
    txt_file = _txt.file.iloc[0] if len(_txt) else "official_person_text"
    txt_rows = int(_txt.rows.iloc[0]) if len(_txt) else 0

    blocks = []
    for f in person.file:
        sub = CP[CP.file == f]
        t = sub.assign(**{"missing %": sub.missing_pct.map(lambda v: "" if pd.isna(v) else f"{v:.1f}"),
                          "distinct": sub.n_unique.map(lambda v: "" if pd.isna(v) else f"{v:,.0f}")})
        blocks.append(f"#### `{f}`\n\n" + md_table(
            t.rename(columns={"column": "column", "dtype": "dtype", "example": "example",
                              "description": "description"}),
            ["column", "dtype", "missing %", "distinct", "example", "description"]))

    doc = f"""# Data dictionary -- outputs of the official raw data track

Generated automatically · {date.today().isoformat()} · `scripts/official_18_manifest.py`

> **This document is generated. Do not edit it by hand.**
> To change a description, edit `config/output_manifest.csv` (per file) or
> `config/person_columns.csv` (columns of the person-level files).

## 0. What to know first

1. **The join key is `row_uid`, and nothing else.** Its form is
   `{{wave}}:{{source_group}}:{{raw row number}}` and it is unique across every wave.
   `person_id` has 890 duplicates in 2020 and must not be used to join.
2. **Filter on `(wave, source_group)`, both of them.** A wave can have several group
   files, so filtering on `wave` alone shifts rows.
3. **The `example` column appears only for vocabulary columns.** Carrying a first-row
   value from a person-level file would put one respondent's record into the
   repository. Only vocabulary columns, with at most {VOCAB_MAX} distinct values, carry
   their first value in sort order; everything else is left empty.
4. **Person-level files live only in `data/interim/official/`.** Internal research use
   is unrestricted (the foundation approved it on 2026-09-09), but the repository and
   its `origin` are an external boundary. The repository column in the tables below is
   what `git check-ignore` decided.
5. **A wave with no weight has a missing one.** `weight` is empty for 2014 through
   2018. Do not fill it with 1; state in every output that those waves are unweighted.

## 1. The join graph

Each declared join checked against the actual values. The match rate is the share of
left-hand keys found on the right; anything below 1 means a key has shifted.

{md_table(J.assign(**{"match rate": J.match_rate.map("{:.4f}".format)}).rename(columns={
    "from_file": "from", "from_keys": "keys", "to_file": "to",
    "n_from_keys": "from keys", "n_matched": "matched", "verdict": "verdict"}),
    ["from", "keys", "to", "from keys", "matched", "match rate", "verdict"])}

Every person-level join is 1.0: any person-level table attaches to the spine on
`row_uid` alone. Only the last row falls short -- {n_nocb} of the {n_reg:,} registry
entries are not in a codebook. Those are the open-text suffixes (`*_OP`), the original
recodes (`Raw_*`) and the identifiers that codebooks do not carry, plus the battery
variables caught by declaration rather than codebook. That is not a defect: it means
**the variable dictionary spans the codebooks while the registry spans the data
files**.

## 2. Person-level files -- candidates for handing out

This is the bundle the persona pipeline takes a copy of. The open text (`{txt_file}`) is
excluded by default, as it carries a re-identification risk.

{md_table(person_tbl.rename(columns={"file": "file", "grain": "what a row is",
                                     "produced_by": "produced by"}),
          ["file", "what a row is", "rows", "columns", "size", "produced by"])}

{md_table(person.assign(**{"repository": person.in_repo.map({True: "included", False: "excluded (gitignored)"})})
          .rename(columns={"file": "file", "sensitivity": "sensitivity", "description": "description"}),
          ["file", "sensitivity", "repository", "description"])}

### 2-1. Sample coverage -- which file holds which (wave, group)

A file listing does not show **which person-level table is missing which sample**. So
it is measured separately: a blank means that table never swept that (wave, group).

{cov_md}

{gap_para}

### 2-2. Column dictionary

{(chr(10) * 2).join(blocks)}

## 3. Aggregate outputs -- what stays in the repository

{md_table(agg_tbl.rename(columns={"file": "file", "grain": "what a row is",
                                  "produced_by": "produced by", "description": "description"}),
          ["file", "what a row is", "rows", "produced by", "description"])}

## 4. Reference material and the backwards-compatible exports

`data/reference/` holds frozen copies of material from outside the microdata and is
never edited by hand. The `export` layer is the set of paths `official_13_exports.py`
rebuilds in the format the persona pipeline reads.

{md_table(ref_tbl.rename(columns={"layer": "layer", "file": "file", "grain": "what a row is",
                                  "produced_by": "produced by", "description": "description"}),
          ["layer", "file", "what a row is", "rows", "produced by", "description"])}

The {len(cache)} raw loader caches at
`data/interim/official/<wave>_<group>.pkl` are `(DataFrame, meta)` tuples. Deleting
them is safe: the next run rereads the sources and rebuilds them.

## 5. Check before handing anything out

This section applies **to releasing outside the repository**. Internal research use is
unrestricted.

- **Direct identifiers sit in the raw data as they are.** The 2020 `NAME` (1,052 real
  names) and `HP` (1,052 mobile numbers) are carried in the open-text table and the
  wave_local table respectively. Neither is an analysis variable.
- **The open-text responses** (`{txt_file}`, {txt_rows:,} rows) carry a high
  re-identification risk. The 2024 `SQ2_NAME` is a free-entry address at neighbourhood
  level.
- **The administrative neighbourhood code** exists only in 2015, 2016 and 2024, and the
  sample per wave is thin. Do not release anything at neighbourhood level.
- **Without shipping `universe` alongside**, an out-of-battery item will be misread as a
  population share (2020 `D1A_*` is 812 people, not everyone).
- **Ship the imputed version together with the original.** An imputed table without
  `is_imputed` cannot be undone.
"""
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "official_data_dictionary.md").write_text(doc, encoding="utf-8")
    print(f"reports/official_data_dictionary.md written ({len(doc):,} characters)")


if __name__ == "__main__":
    main()
