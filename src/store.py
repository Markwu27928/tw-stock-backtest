"""資料快取：每日一個小檔（不會再改動），方便 git 版本控管、repo 不會快速膨脹。

data/price/YYYY/YYYYMMDD.csv.gz   每日收盤行情
data/t86/YYYY/YYYYMMDD.csv.gz     三大法人買賣超
data/div/YYYY.csv.gz              除權息
data/rev/YYYY-MM.csv.gz           月營收（公開資訊觀測站）
data/rev_finmind/CODE.csv.gz      月營收（FinMind 備援）
data/state.json                   假日、無資料清單、抓取紀錄
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from . import config


def _p(*parts) -> Path:
    return config.DATA_DIR.joinpath(*parts)


def daily_path(kind: str, d: date) -> Path:
    return _p(kind, f"{d:%Y}", f"{d:%Y%m%d}.csv.gz")


def save_daily(kind: str, d: date, df: pd.DataFrame) -> None:
    p = daily_path(kind, d)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, compression="gzip")


def has_daily(kind: str, d: date) -> bool:
    return daily_path(kind, d).exists()


def load_daily(kind: str) -> pd.DataFrame:
    files = sorted(_p(kind).glob("*/*.csv.gz"))
    parts = []
    for f in files:
        df = pd.read_csv(f, dtype={"code": str})
        df["date"] = pd.Timestamp(f.name[:8])
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def rev_path(ym: str) -> Path:
    return _p("rev", f"{ym}.csv.gz")


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def load_revenue() -> pd.DataFrame:
    """回傳 ym, code, rev, rev_ly（官方檔優先，FinMind 補缺）。"""
    parts = []
    for f in sorted(_p("rev").glob("*.csv.gz")):
        df = pd.read_csv(f, dtype={"code": str})
        df["ym"] = f.name[:7]
        parts.append(df[["ym", "code", "rev", "rev_ly"]])
    off = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ym", "code", "rev", "rev_ly"])

    fm_parts = [pd.read_csv(f, dtype={"code": str}) for f in sorted(_p("rev_finmind").glob("*.csv.gz"))]
    if fm_parts:
        fm = pd.concat(fm_parts, ignore_index=True)
        fm["period"] = pd.PeriodIndex(fm["ym"], freq="M")
        ly = fm[["code", "period", "rev"]].copy()
        ly["period"] = ly["period"] + 12
        fm = fm.merge(ly.rename(columns={"rev": "rev_ly"}), on=["code", "period"], how="left")
        fm = fm[["ym", "code", "rev", "rev_ly"]]
        key = set(zip(off["ym"], off["code"]))
        fm = fm[[(a, b) not in key for a, b in zip(fm["ym"], fm["code"])]]
        off = pd.concat([off, fm], ignore_index=True)
    return off


def load_dividends() -> pd.DataFrame:
    parts = [pd.read_csv(f, dtype={"code": str}) for f in sorted(_p("div").glob("*.csv.gz"))]
    if not parts:
        return pd.DataFrame(columns=["date", "code", "before", "ref"])
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "code"])
    df["date"] = pd.to_datetime(df["date"])
    return df


class State:
    PATH_NAME = "state.json"

    def __init__(self, d: dict):
        self.d = d
        for k, v in {"holidays": [], "t86_na": [], "rev_na": [], "div": {},
                     "finmind_done": [], "rev_source": "mops"}.items():
            self.d.setdefault(k, v)
        self._hol = set(self.d["holidays"])
        self._t86na = set(self.d["t86_na"])

    @classmethod
    def load(cls) -> "State":
        p = _p(cls.PATH_NAME)
        return cls(json.loads(p.read_text("utf-8")) if p.exists() else {})

    def save(self) -> None:
        self.d["holidays"] = sorted(self._hol)
        self.d["t86_na"] = sorted(self._t86na)
        p = _p(self.PATH_NAME)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.d, ensure_ascii=False, indent=1), "utf-8")

    def is_holiday(self, d: date) -> bool:
        return d.isoformat() in self._hol

    def add_holiday(self, d: date) -> None:
        self._hol.add(d.isoformat())

    def t86_na(self, d: date) -> bool:
        return d.isoformat() in self._t86na

    def add_t86_na(self, d: date) -> None:
        self._t86na.add(d.isoformat())
