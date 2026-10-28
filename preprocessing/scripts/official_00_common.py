"""Official raw data track, shared parts -- paths, readers, cache, vocabulary.

Reads the foundation's official delivery of 2026-09-08 (`data/official_raw_data/`),
which holds person-level data for all seven waves from 2014 to 2024. The directory
contains versions that were not released under the disclosure request, so it is
gitignored.

The earlier `microdata_*` track read a shared copy of the 2018-2024 disclosure-request
versions. This track replaces it and, for the same wave, uses the richer version:
  · 2018 takes the `.xls` the codebook documents (256 columns, including derived
    recodes) as the source of record. The disclosure `.xlsx` (223 columns) carries the
    DQ and F blocks but no A/B recodes, so the two are merged.
  · 2022 and 2024 take the SAV as the source of record. Its contents match the
    disclosure xlsx exactly (rows and columns both checked), but the SPSS metadata
    additionally carries the question wording and the value labels.
  · 2020 has no SAV, so the disclosure-request xlsx is used.

The group vocabulary:
  general    the general public (the 2016 general panel, Seoul residents from 2018)
  interest   the culture-interested group (foundation and institutional members, the
             Seoul integrated membership, art museum members)
  disabled   disabled respondents
  multicult  multicultural families
  mixed      one file holding several frames, to be split on a frame variable
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "data" / "official_raw_data"
CACHE = ROOT / "data" / "interim" / "official"
# Person-level outputs stay out of the repository (gitignored). Not a permissions
# question: the repository and its origin are an external boundary, and it is 1.4 GB.
PERSON = ROOT / "data" / "interim" / "official"
# Only aggregate and metadata outputs are kept in the repository.
OUT = ROOT / "data" / "processed" / "official"
# Reference material from outside the microdata (frozen published figures, the
# living-area crosswalk). Tracked, in the repository's top-level data/reference/.
REFERENCE = ROOT.parent / "data" / "reference"
REPORTS = ROOT / "reports"

DIRS = {
    2014: OFFICIAL / "2014 서울시민문화향유실태조사 raw data(코드북포함)",
    2015: OFFICIAL / "2015 서울시민문화향유실태조사 raw data(코드북포함)",
    2016: OFFICIAL / "2016 서울시민문화향유실태조사 raw data(코드북포함)",
    2018: OFFICIAL / "2018 서울시민문화향유실태조사 raw data(코드북포함)",
    2020: OFFICIAL / "2020 서울시민문화향유실태조사 raw data(코드북포함)",
    2022: OFFICIAL / "2022 서울시민문화향유실태조사 raw data(코드북포함)",
    2024: OFFICIAL / "2024 서울시민문화향유실태조사 raw data(코드북포함)",
}


# --------------------------------------------------------------- low-level IO

def read_xls(path: Path, sheet: int = 0) -> pd.DataFrame:
    """Read an .xls with xlrd. Empty strings become NA."""
    import xlrd

    wb = xlrd.open_workbook(path)
    sh = wb.sheet_by_index(sheet)
    hdr = [str(sh.cell_value(0, c)).strip() for c in range(sh.ncols)]
    rows = [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(1, sh.nrows)]
    df = pd.DataFrame(rows, columns=hdr)
    return df.replace({"": pd.NA, " ": pd.NA})


def read_xlsx(path: Path, sheet=0, header: bool = True) -> pd.DataFrame:
    """Read an .xlsx with openpyxl."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return pd.DataFrame()
    if header:
        cols = [str(c).strip() if c is not None else f"_c{i}" for i, c in enumerate(rows[0])]
        return pd.DataFrame(rows[1:], columns=cols).replace({"": pd.NA, " ": pd.NA})
    return pd.DataFrame(rows)


def read_sav(path: Path):
    """Read a SAV as (DataFrame, meta). The meta carries question wording and value
    labels."""
    import pyreadstat

    return pyreadstat.read_sav(str(path))


# ------------------------------------------------------------- the file register
# (year, group) -> {file, kind, sheet, frame_var}
FILES = {
    (2014, "mixed"): {
        "file": DIRS[2014] / "2014 서울시민 문화예술 향유 실태조사_DATA.xls",
        "kind": "xls",
        "codebook": (DIRS[2014] / "2014 서울시민 문화예술 향유 실태조사_코드북.xls", "xls", 0),
    },
    (2015, "mixed"): {
        "file": DIRS[2015] / "2015 서울시민 문화예술 향유 실태조사_DATA.xls",
        "kind": "xls",
        "frame_var": "DEPART",
        "codebook": (
            DIRS[2015] / "2015 서울문화재단인지도및서울시민문화예술향유실태조사 코드북 자료(0803).xlsx",
            "xlsx",
            "공통",
        ),
    },
    (2016, "mixed"): {
        "file": DIRS[2016] / "2016 서울문화재단인지도및서울시민문화예술향유실태조사_RAW DATA.xlsx",
        "kind": "xlsx",
        "sheet": "raw data",
        "frame_var": "TYPE",
        "codebook": (
            DIRS[2016] / "2016 서울문화재단인지도및서울시민문화예술향유실태조사 코드북 자료.xlsx",
            "xlsx",
            "CODE BOOK",
        ),
    },
    (2018, "general"): {
        "file": DIRS[2018] / "2018 서울시민 문화향유 실태조사(일반시민)_DATA.xls",
        "kind": "xls",
        "supplement": DIRS[2018] / "2018_DATA_서울시민(정보공개).xlsx",
        "codebook": (
            DIRS[2018] / "2018 서울시민 문화향유 실태조사(서울시민)_코드북.xlsx",
            "xlsx",
            "코드북",
        ),
    },
    (2018, "interest"): {
        "file": DIRS[2018] / "2018 서울시민 문화향유 실태조사(기관회원)_DATA.xls",
        "kind": "xls",
        "supplement": DIRS[2018] / "2018_DATA_기관회원(정보공개).xlsx",
        "codebook": (
            DIRS[2018] / "2018 서울시민 문화향유 실태조사(기관별)_코드북.xlsx",
            "xlsx",
            "코드북",
        ),
    },
    (2020, "general"): {
        "file": DIRS[2020] / "1. DATA_2020 서울시민 문화향유 실태조사_일반시민_F(정보공개청구).xlsx",
        "kind": "xlsx",
        "sheet": 0,
        "codebook": (DIRS[2020] / "0. DATAMAP_2020 서울시민 문화향유 실태조사.xlsx", "xlsx", 0),
    },
    (2020, "interest"): {
        "file": DIRS[2020] / "2. DATA_2020 서울시민 문화향유 실태조사_문화관심집단_F(정보공개청구).xlsx",
        "kind": "xlsx",
        "sheet": "DATA",
        "codebook": (DIRS[2020] / "0. DATAMAP_2020 서울시민 문화향유 실태조사.xlsx", "xlsx", 0),
    },
    (2022, "mixed"): {
        "file": DIRS[2022] / "데이터_2022년 서울시민 문화향유 실태조사_최종 (1).SAV",
        "kind": "sav",
        "frame_var": "GRP",
        "codebook": (
            DIRS[2022] / "코드북_2022년 서울시민 문화향유 실태조사_(정보공개청구).xlsx",
            "xlsx",
            0,
        ),
    },
    (2024, "general"): {
        "file": DIRS[2024] / "(RawDATA_전달) 2024 서울시민 문화향유 실태조사_일반시민(가중).SAV",
        "kind": "sav",
    },
    (2024, "interest"): {
        "file": DIRS[2024] / "(RawDATA_전달) 2024 서울시민 문화향유 실태조사_문화관심_V2.SAV",
        "kind": "sav",
    },
    (2024, "disabled"): {
        "file": DIRS[2024] / "(RawDATA_전달) 2024 서울시민 문화향유 실태조사_장애인.SAV",
        "kind": "sav",
    },
}

# The value that splits frames within one file -> (group, frame label)
# For 2015 the codebook gives DEPART as 1-8 districts, 9 foundation members and
# 10 the existing panel. The delivered file holds only 9 and 10 -- the district
# frames never arrived. Code 10 is the cultural policy panel, not a general-public
# sample, so both codes are interest.
FRAME_MAPS = {
    2015: {9: ("interest", "재단회원"), 10: ("interest", "기존패널")},
    2016: {
        1: ("general", "일반패널"),
        2: ("interest", "서울시 통합회원"),
        3: ("interest", "서울시립미술관"),
        4: ("interest", "서울문화재단"),
        5: ("interest", "자치구"),
    },
    2022: {
        1: ("general", "패널 조사"),
        2: ("interest", "관심집단 조사"),
        3: ("general", "면접 조사"),
        4: ("disabled", "장애인"),
        5: ("multicult", "다문화가정"),
    },
}


# ------------------------------------------------------------------ file listing

def groups() -> list[tuple[int, str]]:
    """The eleven (wave, group) pairs of raw data. The file register is the only
    source -- anywhere this list is retyped by hand will eventually disagree."""
    return sorted(FILES)


def default_group(year: int) -> str:
    """The wave's representative file. Where one file mixes frames, that is mixed."""
    return "mixed" if (year, "mixed") in FILES else "general"


# ------------------------------------------------------------ person-level table IO
# Person-level tables are Parquet. The table is long, so the same values (row_uid,
# family, item, measure) repeat millions of times and dictionary encoding folds them
# away: 566 MB becomes 10.7 MB, reading is 34 times faster, and reading a single
# column 42 times. Being able to read column-wise removes every place that used to
# parse a whole CSV just to take a few columns through usecols.
PERSON_SUFFIX = ".parquet"


def person_path(name: str) -> Path:
    return PERSON / f"{name}{PERSON_SUFFIX}"


def write_person(df, name: str) -> None:
    PERSON.mkdir(parents=True, exist_ok=True)
    df.to_parquet(person_path(name), index=False, compression="zstd")


def read_person(name: str, columns=None):
    return pd.read_parquet(person_path(name), columns=columns)


# --------------------------------------------------------------- table rendering

def md_table(df, cols=None) -> str:
    """DataFrame to a markdown table. Shared by the reports and the data dictionary."""
    df = df[cols] if cols else df
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


# ------------------------------------------------------------------ cached loader

def load(year: int, group: str = None, refresh: bool = False):
    """Load the (year, group) source as a DataFrame. For SAV the meta is pickled too."""
    if group is None:
        group = default_group(year)
    key = (year, group)
    spec = FILES[key]
    p = CACHE / f"{year}_{group}.pkl"
    if p.exists() and not refresh:
        with open(p, "rb") as f:
            return pickle.load(f)

    kind = spec["kind"]
    if kind == "xls":
        df, meta = read_xls(spec["file"]), None
        # The two 2018 versions carry different variables: the source-of-record .xls
        # has the derived recodes, the disclosure .xlsx has the DQ (demographic) and F
        # blocks. The IDs match exactly (all 2,545 rows, and SQ2, SQ3 and SQ1_GU agree
        # on every value), so the missing columns are simply appended.
        sup = spec.get("supplement")
        if sup is not None:
            sheet = "DATA_서울시민" if "서울시민" in sup.name else "DATA_기관회원"
            extra = read_xlsx(sup, sheet)
            # The IDs must not be compared as strings. The .xls reads 1.0 and the
            # .xlsx reads 1, so "1.0" != "1" and the reindex turns every row missing.
            # That bug left all 20 columns of the 2018 DQ block empty.
            eid = pd.to_numeric(extra["ID"], errors="coerce")
            did = pd.to_numeric(df["ID"], errors="coerce")
            if not eid.equals(did):
                extra = extra.set_index(eid).reindex(did).reset_index(drop=True)
                if extra.drop(columns=["ID"], errors="ignore").notna().sum().sum() == 0:
                    raise AssertionError(
                        "the 2018 supplement failed to align -- check the ID correspondence")
            new = [c for c in extra.columns if c not in df.columns]
            df = pd.concat([df, extra[new].reset_index(drop=True)], axis=1)
    elif kind == "xlsx":
        df, meta = read_xlsx(spec["file"], spec.get("sheet", 0)), None
    elif kind == "sav":
        df, meta = read_sav(spec["file"])
    else:
        raise ValueError(kind)

    CACHE.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump((df, meta), f)
    return df, meta


def load_codebook(year: int, group: str = None) -> pd.DataFrame:
    if group is None:
        group = default_group(year)
    cb = FILES[(year, group)].get("codebook")
    if cb is None:
        return pd.DataFrame()
    path, kind, sheet = cb
    return read_xls(path, sheet) if kind == "xls" else read_xlsx(path, sheet, header=False)
