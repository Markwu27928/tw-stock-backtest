"""把快取資料整理成面板，並在每個決策日計算特徵。

重點：決策日 d 的特徵只用 d 當天（含）以前的資料；
月營收只用「d 所在月份的上個月」，該月營收在 d 之前（10 日）已公告。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config, store

FEATURES = ["rev_yoy", "rev_mom", "yoy_3m", "yoy_accel", "yoy_up2", "yoy_streak",
            "ret_21", "mom_6_1", "vol_20", "high_252", "trust_20", "foreign_20", "log_value_20"]


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    close: pd.DataFrame        # 原始收盤價（沒成交為 NaN）
    tri: pd.DataFrame          # 還原權息報酬指數
    ret: pd.DataFrame          # 還原後日報酬
    value: pd.DataFrame
    volume: pd.DataFrame
    trust: pd.DataFrame
    foreign: pd.DataFrame
    history: pd.DataFrame      # 已上市交易日數
    names: pd.Series
    rev: pd.DataFrame          # index=Period(M), columns=code
    rev_ly: pd.DataFrame


def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot_table(index="date", columns="code", values=col, aggfunc="last")


def build_panel(until: pd.Timestamp | None = None) -> Panel:
    px = store.load_daily("price")
    if px.empty:
        raise RuntimeError("還沒有任何股價資料")
    if until is not None:
        px = px[px["date"] <= until]
    px = px.drop_duplicates(["date", "code"], keep="last")
    close = _pivot(px, "close").sort_index()
    cols, idx = close.columns, close.index
    value = _pivot(px, "value").reindex(index=idx, columns=cols).fillna(0.0)
    volume = _pivot(px, "volume").reindex(index=idx, columns=cols).fillna(0.0)
    names = px.groupby("code")["name"].last()

    t86 = store.load_daily("t86")
    if t86.empty:
        trust = foreign = pd.DataFrame(0.0, index=idx, columns=cols)
    else:
        t86 = t86.drop_duplicates(["date", "code"], keep="last")
        trust = _pivot(t86, "trust_net").reindex(index=idx, columns=cols).fillna(0.0)
        foreign = _pivot(t86, "foreign_net").reindex(index=idx, columns=cols).fillna(0.0)

    # 除權息還原：除權息日的報酬乘上 (除權息前收盤 / 參考價)
    factor = pd.DataFrame(1.0, index=idx, columns=cols)
    div = store.load_dividends()
    if not div.empty:
        div = div[div["date"].isin(idx) & div["code"].isin(cols) & (div["ref"] > 0)]
        for r in div.itertuples():
            factor.at[r.date, r.code] = r.before / r.ref
    cf = close.ffill()
    ret = (cf / cf.shift(1)) * factor - 1
    # 漲跌幅 10%；超過多半是減資、分割等未處理事件，截斷避免假報酬
    ret = ret.clip(-0.11, 0.11)
    tri = (1 + ret.fillna(0)).cumprod().where(cf.notna())
    history = cf.notna().cumsum()

    rv = store.load_revenue()
    if rv.empty:
        rev = rev_ly = pd.DataFrame()
    else:
        rv = rv.drop_duplicates(["ym", "code"], keep="first").copy()
        rv["period"] = pd.PeriodIndex(rv["ym"], freq="M")
        rev = rv.pivot(index="period", columns="code", values="rev").sort_index()
        rev_ly = rv.pivot(index="period", columns="code", values="rev_ly").reindex_like(rev)
    return Panel(idx, close, tri, ret, value, volume, trust, foreign, history, names, rev, rev_ly)


def decision_dates(p: Panel, start: str | None = None) -> list[pd.Timestamp]:
    """每月第一個 >= 11 日的交易日。"""
    s = pd.Series(p.dates, index=p.dates)
    s = s[s.dt.day >= config.REBALANCE_DAY]
    out = s.groupby(s.dt.to_period("M")).min().tolist()
    if start:
        out = [d for d in out if d >= pd.Timestamp(start + "-01")]
    return out


def _rev_row(p: Panel, period: pd.Period, frame: pd.DataFrame) -> pd.Series:
    if frame.empty or period not in frame.index:
        return pd.Series(dtype=float)
    return frame.loc[period]


def revenue_features(p: Panel, d: pd.Timestamp) -> pd.DataFrame:
    m1 = d.to_period("M") - 1
    revs = [_rev_row(p, m1 - k, p.rev) for k in range(13)]
    lys = [_rev_row(p, m1 - k, p.rev_ly) for k in range(13)]
    if revs[0].empty:
        return pd.DataFrame(columns=FEATURES[:6], dtype=float)
    codes = revs[0].index

    def yoy(k):
        r = revs[k].reindex(codes)
        l = lys[k].reindex(codes)
        return (r / l.where(l > 0) - 1).clip(-1, 5)

    y = [yoy(k) for k in range(13)]
    r0, r1 = revs[0].reindex(codes), revs[1].reindex(codes)
    sum_r = sum(revs[k].reindex(codes) for k in range(3))
    sum_l = sum(lys[k].reindex(codes) for k in range(3))
    streak = pd.Series(0, index=codes, dtype=float)
    alive = pd.Series(True, index=codes)
    for k in range(12):
        alive &= (y[k] > 0).fillna(False)
        streak += alive.astype(float)
    return pd.DataFrame({
        "rev_yoy": y[0],
        "rev_mom": (r0 / r1.where(r1 > 0) - 1).clip(-1, 5),
        "yoy_3m": (sum_r / sum_l.where(sum_l > 0) - 1).clip(-1, 5),
        "yoy_accel": y[0] - y[1],
        "yoy_up2": ((y[0] > y[1]) & (y[1] > y[2])).astype(float),
        "yoy_streak": streak,
    })


def features_at(p: Panel, d: pd.Timestamp) -> pd.DataFrame:
    """決策日 d 的全市場特徵（已套用選股母體條件）。"""
    i = p.dates.get_loc(d)
    if i < 21:
        return pd.DataFrame(columns=FEATURES, dtype=float)
    tri = p.tri
    lo20 = max(0, i - 19)
    vol20 = p.volume.iloc[lo20:i + 1].sum()
    f = pd.DataFrame({
        "close": p.close.iloc[i],
        "ret_21": tri.iloc[i] / tri.iloc[i - 21] - 1,
        "mom_6_1": tri.iloc[i - 21] / tri.iloc[i - 126] - 1 if i >= 126 else np.nan,
        "vol_20": p.ret.iloc[lo20:i + 1].std(),
        "high_252": tri.iloc[i] / tri.iloc[max(0, i - 251):i + 1].max(),
        "value_20": p.value.iloc[lo20:i + 1].mean(),
        "trust_20": p.trust.iloc[lo20:i + 1].sum() / vol20.where(vol20 > 0),
        "foreign_20": p.foreign.iloc[lo20:i + 1].sum() / vol20.where(vol20 > 0),
        "history": p.history.iloc[i],
    })
    f["log_value_20"] = np.log10(f["value_20"].where(f["value_20"] > 0))
    f = f.join(revenue_features(p, d), how="left")
    codes = f.index.to_series()
    mask = (
        codes.str.fullmatch(r"[1-9]\d{3}") & ~codes.str.startswith("91")
        & (f["close"] >= config.MIN_PRICE)
        & (f["value_20"] >= config.MIN_VALUE_20)
        & (f["history"] >= config.MIN_HISTORY_DAYS)
        & f["rev_yoy"].notna() & f["ret_21"].notna()
    )
    return f[mask.fillna(False)]


def forward_returns(p: Panel, entry: pd.Timestamp, exit_: pd.Timestamp, codes) -> pd.Series:
    t = p.tri
    return (t.loc[exit_, codes] / t.loc[entry, codes] - 1).astype(float)


def bench_return(p: Panel, entry: pd.Timestamp, exit_: pd.Timestamp) -> float:
    b = config.BENCH
    if b not in p.tri.columns:
        return np.nan
    return float(p.tri.at[exit_, b] / p.tri.at[entry, b] - 1)
