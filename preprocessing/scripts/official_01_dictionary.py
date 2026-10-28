"""Variable dictionary and value labels for seven waves, from the official codebooks
and the SPSS metadata.

The codebook format differs by wave. Four shapes are folded into one.

  2014, 2015, 2016  question | variable | description | choices | note
                    (the choices cell holds the value labels as free text)
  2018              section | question | description | variable | value | value label
                    (merged cells, forward-filled)
  2020              two DATAMAP sheets: variable information (variable, label) and
                    variable values (variable, value, label)
  2022, 2024        SPSS metadata (the question wording sits in the variable label)

Outputs:
  official_variable_dictionary.csv  wave x group x variable x label x source x present
  official_value_labels.csv         wave x group x variable x value x label
"""
from __future__ import annotations

import re
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import official_00_common as C

# Expand variables the codebook writes as a range, such as "B1_2_1#1~3", one by one.
_RANGE_PAT = re.compile(r"^(.*)(\d+)(\D*)~(\d+)(\D*)$")


def expand_range(var: str) -> list[str]:
    m = _RANGE_PAT.match(var)
    if not m:
        return [var]
    prefix, a, suf1, b, suf2 = m.groups()
    if suf1 != suf2:
        return [var]
    a, b = int(a), int(b)
    if not (0 <= a <= b <= 99):
        return [var]
    return [f"{prefix}{i}{suf1}" for i in range(a, b + 1)]


def norm(v: str) -> str:
    """Separator conventions drift across waves (2015 B2A1 vs B2_A1, 2016 A1_1 vs
    A1#1)."""
    return re.sub(r"[_#\s]", "", str(v)).upper()


# Name differences that separator normalisation still does not resolve.
# (wave, codebook name) -> data column name. Record the evidence: check the value
# range and the distribution before adding an entry.
ALIASES = {
    # Codebook 'SQ1_1_GU' (district of residence, 25 choices) = data 'SQ1_GU'.
    # Both groups checked as 1-25, 25 distinct values, no missing (2026-09-08).
    (2018, "SQ1_1_GU"): "SQ1_GU",
}


# Take apart the choices cell, which comes as "1 'male' 2 'female'" or "1. male
# 2. female" in the instrument's own wording.
_VAL_PAT = re.compile(r"(\d+)\s*[.'\)]?\s*'([^']*)'|(\d+)\s*[.\)]\s*([^\d'][^\n]*?)(?=\s+\d+\s*[.\)]|$)")


def parse_choices(text) -> list[tuple[str, str]]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    s = str(text).replace("\n", " ").strip()
    if not s:
        return []
    out = []
    for m in _VAL_PAT.finditer(s):
        if m.group(1) is not None:
            out.append((m.group(1), m.group(2).strip()))
        elif m.group(3) is not None:
            lab = m.group(4).strip().rstrip("'").strip()
            if lab:
                out.append((m.group(3), lab))
    return out


def _five_col(df: pd.DataFrame, wave: int, group: str, source: str):
    """The 2014/2015/2016 shape: question, variable, description, choices, note. Find
    the header row and read what follows it."""
    hdr = None
    for i in range(min(8, len(df))):
        row = [str(v) for v in df.iloc[i].tolist()]
        if any("변수" in v for v in row) and any("보기" in v for v in row):
            hdr = i
            break
    if hdr is None:
        return [], []
    body = df.iloc[hdr + 1 :]
    variables, values = [], []
    cur_q = None
    for _, r in body.iterrows():
        q, var, desc, choices = (r.iloc[0], r.iloc[1], r.iloc[2], r.iloc[3] if len(r) > 3 else None)
        q = None if pd.isna(q) or str(q).strip() in ("", "-") else str(q).strip()
        if q:
            cur_q = q
        var = None if pd.isna(var) else str(var).strip()
        if not var:
            continue
        desc = "" if pd.isna(desc) else str(desc).strip().replace("\n", " ")
        # one cell sometimes holds two variables, as in "DEPART/ PART"
        raw = [x.strip() for x in re.split(r"[/\n]", var) if x.strip()]
        for v in [e for x in raw for e in expand_range(x)]:
            variables.append(
                dict(wave=wave, group=group, variable=v, question=cur_q, label=desc, source=source)
            )
            for val, lab in parse_choices(choices):
                values.append(dict(wave=wave, group=group, variable=v, value=val, label=lab, source=source))
    return variables, values


def _2018(df: pd.DataFrame, wave: int, group: str, source: str):
    """Section, question, description, variable, value, value label. Merged cells are
    forward-filled."""
    d = df.iloc[1:].copy()
    d.columns = ["구분", "문항", "변수설명", "변수명", "변수값", "변수값설명"] + list(d.columns[6:])
    for c in ["문항", "변수설명", "변수명"]:
        d[c] = d[c].replace({"": None}).ffill()
    variables, values = [], []
    seen = set()
    for _, r in d.iterrows():
        var = r["변수명"]
        if var is None or pd.isna(var) or not str(var).strip():
            continue
        desc = "" if pd.isna(r["변수설명"]) else str(r["변수설명"]).strip()
        q = None if pd.isna(r["문항"]) else str(r["문항"]).strip()
        val, lab = r["변수값"], r["변수값설명"]
        for var_i in expand_range(str(var).strip()):
            if var_i not in seen:
                seen.add(var_i)
                variables.append(
                    dict(wave=wave, group=group, variable=var_i, question=q, label=desc, source=source)
                )
            if not pd.isna(val) and not pd.isna(lab) and str(lab).strip():
                values.append(
                    dict(wave=wave, group=group, variable=var_i, value=str(val).strip(),
                         label=str(lab).strip(), source=source)
                )
    return variables, values


def _2020(wave: int, group: str):
    src = "DATAMAP_2020"
    info = C.read_xlsx(C.DIRS[2020] / "0. DATAMAP_2020 서울시민 문화향유 실태조사.xlsx", "변수 정보", header=False)
    vals = C.read_xlsx(C.DIRS[2020] / "0. DATAMAP_2020 서울시민 문화향유 실태조사.xlsx", "변수값", header=False)
    variables = []
    for _, r in info.iloc[2:].iterrows():
        v = r.iloc[0]
        if pd.isna(v) or not str(v).strip():
            continue
        lab = "" if pd.isna(r.iloc[1]) else str(r.iloc[1]).strip()
        q = lab.split(".")[0] if "." in lab else None
        variables.append(dict(wave=wave, group=group, variable=str(v).strip(), question=q, label=lab, source=src))
    values, cur = [], None
    for _, r in vals.iloc[2:].iterrows():
        v, val, lab = r.iloc[0], r.iloc[1], r.iloc[2]
        if not pd.isna(v) and str(v).strip():
            cur = str(v).strip()
        if cur is None or pd.isna(val) or pd.isna(lab):
            continue
        values.append(dict(wave=wave, group=group, variable=cur, value=str(val).strip(),
                           label=str(lab).strip(), source=src))
    return variables, values


def _from_sav(wave: int, group: str, meta, source: str):
    variables = [
        dict(wave=wave, group=group, variable=c, question=c.split("_")[0],
             label=(meta.column_names_to_labels.get(c) or ""), source=source)
        for c in meta.column_names
    ]
    values = [
        dict(wave=wave, group=group, variable=v, value=str(k), label=str(lab), source=source)
        for v, m in meta.variable_value_labels.items()
        for k, lab in m.items()
    ]
    return variables, values


def main():
    V, L = [], []
    plans = [
        (2014, "mixed", "codebook_2014"),
        (2016, "mixed", "codebook_2016"),
    ]
    for wave, group, src in plans:
        cb = C.load_codebook(wave, group)
        v, l = _five_col(cb, wave, group, src)
        V += v
        L += l

    # The 2015 codebook is split across sheets: one shared sheet plus one per frame
    cb_path = C.FILES[(2015, "mixed")]["codebook"][0]
    import openpyxl

    wb = openpyxl.load_workbook(cb_path, read_only=True, data_only=True)
    sheets = wb.sheetnames
    wb.close()
    for sn in sheets:
        v, l = _five_col(C.read_xlsx(cb_path, sn, header=False), 2015, "mixed", f"codebook_2015:{sn}")
        V += v
        L += l

    for group in ("general", "interest"):
        cb = C.load_codebook(2018, group)
        v, l = _2018(cb, 2018, group, f"codebook_2018_{group}")
        V += v
        L += l
        v, l = _2020(2020, group)
        V += v
        L += l

    for wave, group in [(2022, "mixed"), (2024, "general"), (2024, "interest"), (2024, "disabled")]:
        _, meta = C.load(wave, group)
        v, l = _from_sav(wave, group, meta, f"sav_{wave}_{group}")
        V += v
        L += l

    vdf = pd.DataFrame(V).drop_duplicates(subset=["wave", "group", "variable", "source"])
    ldf = pd.DataFrame(L).drop_duplicates(subset=["wave", "group", "variable", "value", "source"])

    # Check every codebook variable against the columns actually in the data
    present = []
    for (wave, group), _ in vdf.groupby(["wave", "group"]):
        df, _m = C.load(wave, group)
        by_norm = {norm(c): c for c in df.columns}
        sub = vdf[(vdf.wave == wave) & (vdf.group == group)].copy()
        sub["data_column"] = sub.variable.map(
            lambda v: (ALIASES.get((wave, v)) if ALIASES.get((wave, v)) in df.columns
                       else by_norm.get(norm(v))))
        sub["in_data"] = sub.data_column.notna()
        present.append(sub)
    vdf = pd.concat(present, ignore_index=True)

    C.OUT.mkdir(parents=True, exist_ok=True)
    vdf.to_csv(C.OUT / "official_variable_dictionary.csv", index=False)
    ldf.to_csv(C.OUT / "official_value_labels.csv", index=False)

    print(f"{len(vdf)} variable rows · {len(ldf)} value-label rows")
    print("\ncodebook variables against the data, by wave and group:")
    g = vdf.groupby(["wave", "group"]).agg(codebook=("variable", "nunique"), matched=("in_data", "sum"))
    for (w, grp), r in g.iterrows():
        df, _m = C.load(w, grp)
        cov = r.matched / len(df.columns) * 100
        print(f"  {w} {grp:9s} codebook {r.codebook:4d} · data columns "
              f"{len(df.columns):4d} · matched {r.matched:4d} ({cov:5.1f}%)")


if __name__ == "__main__":
    main()
