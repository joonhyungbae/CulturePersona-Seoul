"""KCAS track, shared parts -- paths, the ZIP reader, person-level storage.

The source of record is a directory under `data/raw/` whose name is NFD-normalised.
Hard-coding an NFC Korean path in a shell script fails; only this module resolves the
path.

Kept entirely separate from the official track. Do not touch `scripts/official_*.py`.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
PERSON = ROOT / "data" / "interim" / "kcas"
OUT = ROOT / "data" / "processed" / "kcas"
PERSON_SUFFIX = ".parquet"

WAVES = [2014, 2016, 2018, 2020, 2021, 2022, 2023, 2024, 2025]

# The nine items common to every wave, which go into Spec.items. 2016 has an extra
# history_philosophy item; it is excluded.
ITEMS = [
    "literature_event", "exhibition", "western_classical", "traditional",
    "theater", "musical", "dance_performance", "movie", "popular_performance",
]

# Waves that left non-attendance as null -- filled with 0 before dichotomising.
NULL_NONATTEND = {2020, 2021, 2025}


def raw_dir() -> Path:
    """Find the NFD-normalised directory by what it contains, not by its name.

    Other zips may sit under `data/raw/`, so the directory is identified by the one
    thing only the KCAS delivery has -- zip members named `<wave>*.csv` for every wave.
    """
    raw = ROOT / "data" / "raw"
    for p in sorted(raw.iterdir()):
        if not p.is_dir():
            continue
        years = set()
        for zp in p.glob("*.zip"):
            with zipfile.ZipFile(zp) as zf:
                for name in zf.namelist():
                    if name.endswith(".csv") and name[:4].isdigit():
                        years.add(int(name[:4]))
        if set(WAVES) <= years:
            return p
    raise FileNotFoundError(
        "no directory under data/raw/ holds KCAS wave CSVs for " + str(WAVES))


def zip_paths() -> list[Path]:
    return sorted(raw_dir().glob("*.zip"))


def csv_member(wave: int) -> tuple[Path, str]:
    """Wave -> (zip path, member name)."""
    for zp in zip_paths():
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if name.endswith(".csv") and name.startswith(str(wave)):
                    return zp, name
    raise FileNotFoundError(f"no CSV member for {wave}")


def load_wave(wave: int, usecols=None) -> pd.DataFrame:
    """Read the cp949 CSV from inside the ZIP. Nothing is extracted to disk."""
    zp, name = csv_member(wave)
    with zipfile.ZipFile(zp) as zf, zf.open(name) as f:
        return pd.read_csv(f, encoding="cp949", usecols=usecols, low_memory=False)


def wave_columns(wave: int) -> list[str]:
    zp, name = csv_member(wave)
    with zipfile.ZipFile(zp) as zf, zf.open(name) as f:
        return list(pd.read_csv(f, encoding="cp949", nrows=0).columns)


def declarations() -> pd.DataFrame:
    return pd.read_csv(CONFIG / "kcas_declarations.csv")


def person_path(name: str = "kcas_person") -> Path:
    return PERSON / f"{name}{PERSON_SUFFIX}"


def write_person(df: pd.DataFrame, name: str = "kcas_person") -> None:
    """Person-level data stays in interim. zstd Parquet, as in the official track."""
    PERSON.mkdir(parents=True, exist_ok=True)
    df.to_parquet(person_path(name), index=False, compression="zstd")


def read_person(name: str = "kcas_person", columns=None) -> pd.DataFrame:
    return pd.read_parquet(person_path(name), columns=columns)


def write_out(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.csv"
    df.to_csv(path, index=False)
    return path
