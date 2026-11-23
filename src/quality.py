"""資料品質檢查：每次抓完資料後執行，結果顯示在報告頁與 Actions 頁面。

等級：
- error   ：會讓回測結果錯誤，應該先處理
- warning ：可能有問題，需要人看一下
- info    ：提醒，通常不需處理
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from . import config, store


def _f(level: str, msg: str) -> dict:
    return {"level": level, "msg": msg}


def check(panel, update_info: dict | None = None, today: date | None = None) -> list[dict]:
    out: list[dict] = []
    info = update_info or {}
    today = today or date.today()

    # 1. 月營收必須是數值（今天曾因 FinMind 空檔變成 object 而讓回測當掉）
    rv = store.load_revenue()
    if not rv.empty:
        for c in ("rev", "rev_ly"):
            if rv[c].dtype.kind not in "fi":
                out.append(_f("error", f"月營收欄位 {c} 不是數值（{rv[c].dtype}），回測可能失敗"))
    fm = sorted(store._p("rev_finmind").glob("*.csv.gz"))
    empty = [f.name[:-7] for f in fm if len(pd.read_csv(f, nrows=1)) == 0]
    if empty:
        out.append(_f("info", f"FinMind 有 {len(empty)} 檔沒有月營收（多為存託憑證，已排除）：{'、'.join(empty[:5])}"))

    # 2. 資料是否最新：最新交易日距今超過 5 天（扣掉週末約 3 個交易日）
    last = panel.dates[-1].date()
    gap = (today - last).days
    if gap > 5:
        out.append(_f("warning", f"最新行情停在 {last}，已經 {gap} 天沒有更新"))

    # 3. 回補進度與連續性
    done, total = info.get("days_done"), info.get("days_total")
    if done is not None and total and done < total:
        out.append(_f("warning", f"歷史行情尚未補完：{done}/{total} 天，連續完整至 {info.get('contiguous_until')}"))

    # 4. 基準 0050 必須有最新價格
    b = config.BENCH
    if b not in panel.close.columns or pd.isna(panel.close[b].iloc[-1]):
        out.append(_f("error", f"基準 {b} 在 {last} 沒有收盤價"))

    # 5. 近 60 個交易日被截斷的極端報酬（多為減資、分割，報酬已截斷在 ±11%）
    stocks = [c for c in panel.ret.columns if len(c) == 4 and c.isdigit() and not c.startswith("0")]
    recent = panel.ret[stocks].iloc[-60:]
    clipped = (recent.abs() >= 0.1099).sum()
    clipped = clipped[clipped > 0].sort_values(ascending=False)
    if len(clipped):
        names = "、".join(f"{c}{panel.names.get(c, '')}" for c in clipped.index[:5])
        out.append(_f("info", f"近 60 個交易日有 {int(clipped.sum())} 筆單日漲跌觸及 ±11% 截斷（{names}），"
                              "若是減資或分割，報酬可能不準"))

    # 6. 最新月營收是否已到（每月 11 日起應有上月資料）
    want = pd.Period(today, freq="M") - (1 if today.day >= config.REBALANCE_DAY else 2)
    if not rv.empty and str(want) not in set(rv["ym"]):
        out.append(_f("warning", f"還沒有 {want} 的月營收，本月選股可能用到舊資料"))

    # 7. 母體大小突然變化（資料缺漏的徵兆）
    n_today = int(panel.close.iloc[-1].notna().sum())
    n_prev = int(panel.close.iloc[-21:-1].notna().sum(axis=1).median()) if len(panel.dates) > 21 else n_today
    if n_prev and n_today < n_prev * 0.9:
        out.append(_f("warning", f"最新交易日只有 {n_today} 檔有收盤價，近期一般約 {n_prev} 檔，可能資料不完整"))
    return out
