"""增量抓資料：有時間上限，跑不完下次接著抓。

順序：除權息 → 月營收 → 最近 20 個交易日 → 由舊到新回補歷史。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, sources, store

TPE = ZoneInfo("Asia/Taipei")


def taipei_now() -> datetime:
    fake = os.getenv("BT_FAKE_NOW")       # 測試用
    return datetime.fromisoformat(fake).replace(tzinfo=TPE) if fake else datetime.now(TPE)


def weekdays(start: date, end: date) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def month_range(start: pd.Period, end: pd.Period) -> list[pd.Period]:
    return list(pd.period_range(start, end, freq="M"))


def latest_final_revenue_month(today: date) -> pd.Period:
    """營收於次月 10 日前公告；11 日起才視為上月資料齊全。"""
    p = pd.Period(today, freq="M")
    return p - 1 if today.day >= config.REBALANCE_DAY else p - 2


class Budget:
    def __init__(self, minutes: float):
        self.end = time.time() + minutes * 60
        self.errors = 0
        self.blocked = False

    def ok(self) -> bool:
        return not self.blocked and time.time() < self.end

    def fail(self, e: Exception, log: list) -> None:
        self.errors += 1
        log.append(f"錯誤：{e}")
        if self.errors >= config.MAX_CONSECUTIVE_ERRORS:
            self.blocked = True
            log.append("連續錯誤過多，本次停止抓取（可能被證交所暫時封鎖）")

    def success(self) -> None:
        self.errors = 0


def finmind_hint(e: Exception) -> str:
    s = str(e)
    if "HTTP 402" in s:
        return "FinMind 本小時額度已用完，下次執行會接續"
    if "HTTP 401" in s or "HTTP 403" in s:
        return "FinMind token 無效或已過期，請到 GitHub Secrets 更新 FINMIND_TOKEN"
    return "FinMind 暫時無法連線"


def update_dividends(st, today, budget, log):
    start = date.fromisoformat(config.DATA_START)
    for y in range(start.year, today.year + 1):
        if not budget.ok():
            return
        done = st.d["div"].get(str(y))
        if y < today.year and done and done >= f"{y + 1}-01-10":
            continue
        if y == today.year and done == today.isoformat():
            continue
        try:
            df = sources.fetch_dividends(max(start, date(y, 1, 1)), min(today, date(y, 12, 31)))
            budget.success()
        except sources.FetchError as e:
            budget.fail(e, log)
            continue
        store.save_csv(store._p("div", f"{y}.csv.gz"), df)
        st.d["div"][str(y)] = today.isoformat()
        log.append(f"除權息 {y}：{len(df)} 筆")


def update_revenue(st, today, budget, log, finmind_token=None):
    start = pd.Period(config.DATA_START, freq="M") - 4
    last = latest_final_revenue_month(today)
    mops_fail = 0
    for p in month_range(start, last):
        if not budget.ok() or st.d["rev_source"] == "finmind":
            break
        ym = str(p)
        if store.rev_path(ym).exists() or ym in st.d["rev_na"]:
            continue
        try:
            df = sources.fetch_revenue_mops(p.year, p.month)
            budget.success()
        except sources.FetchError as e:
            budget.fail(e, log)
            mops_fail += 1
            if mops_fail >= 3 and not any(store._p("rev").glob("*.csv.gz")):
                st.d["rev_source"] = "finmind"
                log.append("公開資訊觀測站連不上，月營收改用 FinMind 逐檔回補")
            continue
        if df is None or df.empty:
            st.d["rev_na"].append(ym)
            log.append(f"月營收 {ym}：查無資料")
        else:
            store.save_csv(store.rev_path(ym), df)
            log.append(f"月營收 {ym}：{len(df)} 家")

    # 最新月份若觀測站沒有，用證交所 OpenAPI 補
    if budget.ok() and not store.rev_path(str(last)).exists():
        try:
            ym, df = sources.fetch_revenue_openapi()
            budget.success()
            if ym and not df.empty and not store.rev_path(ym).exists() and pd.Period(ym) <= last:
                store.save_csv(store.rev_path(ym), df)
                log.append(f"月營收 {ym}（OpenAPI）：{len(df)} 家")
        except sources.FetchError as e:
            budget.fail(e, log)

    if st.d["rev_source"] == "finmind":
        codes = _known_codes()
        done = set(st.d["finmind_done"])
        fm_errors = 0
        for c in codes:
            if not budget.ok():
                break
            if c in done:
                continue
            try:
                df = sources.fetch_revenue_finmind(c, str(start.start_time.date()), finmind_token)
                fm_errors = 0
            except sources.FetchError as e:
                # FinMind 的錯誤只停掉 FinMind 這段，不計入證交所的連續錯誤，行情回補照常進行
                fm_errors += 1
                if fm_errors >= config.MAX_CONSECUTIVE_ERRORS:
                    log.append(f"錯誤：{e}")
                    log.append(f"月營收逐檔回補暫停：{finmind_hint(e)}（已完成 {len(done)}/{len(codes)} 檔）")
                    break
                continue
            store.save_csv(store._p("rev_finmind", f"{c}.csv.gz"), df)
            st.d["finmind_done"].append(c)
            done.add(c)


def _known_codes() -> list[str]:
    files = sorted(store._p("price").glob("*/*.csv.gz"))[-5:]
    codes = set()
    for f in files:
        codes |= set(pd.read_csv(f, dtype={"code": str})["code"])
    return sorted(c for c in codes if len(c) == 4 and c.isdigit() and not c.startswith("0"))


def update_daily(st, now, budget, log):
    today = now.date()
    days = weekdays(date.fromisoformat(config.DATA_START), today)
    if now.hour < 15 and days and days[-1] == today:   # 今天收盤資料還沒出來
        days = days[:-1]
    recent = list(reversed(days[-20:]))
    order = recent + [d for d in days[:-20]]
    n_price = n_t86 = 0
    for d in order:
        if not budget.ok():
            break
        if st.is_holiday(d):
            continue
        if not store.has_daily("price", d):
            try:
                df = sources.fetch_prices(d)
                budget.success()
            except sources.FetchError as e:
                budget.fail(e, log)
                continue
            if df is None or df.empty:
                if d < today:
                    st.add_holiday(d)
                continue
            store.save_daily("price", d, df)
            n_price += 1
        if not store.has_daily("t86", d) and not st.t86_na(d) and budget.ok():
            try:
                df = sources.fetch_t86(d)
                budget.success()
            except sources.FetchError as e:
                budget.fail(e, log)
                continue
            if df is None or df.empty:
                if (today - d).days > 10:
                    st.add_t86_na(d)
                continue
            store.save_daily("t86", d, df)
            n_t86 += 1
        if (n_price + n_t86) % 40 == 0:
            st.save()
    log.append(f"每日行情新增 {n_price} 天、法人新增 {n_t86} 天")
    return n_price, n_t86


def coverage(st, now) -> dict:
    """從資料起點開始，連續完整的最後一天（回測只用到這裡）。"""
    today = now.date()
    days = weekdays(date.fromisoformat(config.DATA_START), today)
    if now.hour < 15 and days and days[-1] == today:
        days = days[:-1]
    contiguous = None
    for d in days:
        ok = st.is_holiday(d) or (store.has_daily("price", d) and (store.has_daily("t86", d) or st.t86_na(d)))
        if not ok:
            break
        contiguous = d
    return {
        "contiguous_until": contiguous.isoformat() if contiguous else None,
        "days_done": sum(1 for d in days if st.is_holiday(d) or store.has_daily("price", d)),
        "days_total": len(days),
    }


def run(max_minutes: float, finmind_token: str | None = None) -> dict:
    now = taipei_now()
    st = store.State.load()
    budget = Budget(max_minutes)
    log: list[str] = []
    new_price = new_t86 = 0
    try:
        update_dividends(st, now.date(), budget, log)
        update_revenue(st, now.date(), budget, log, finmind_token)
        new_price, new_t86 = update_daily(st, now, budget, log)
    finally:
        st.save()
    cov = coverage(st, now)
    summary = {"run_at": now.isoformat(timespec="minutes"), "log": log[-30:],
               "new_price_days": new_price, "new_t86_days": new_t86,
               "stopped_early": budget.blocked,
               "warnings": [l for l in log if l.startswith(("連續錯誤過多", "月營收逐檔回補暫停"))],
               **cov}
    (config.DATA_DIR / "update_log.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=45)
    a = ap.parse_args()
    s = run(a.max_minutes, os.getenv("FINMIND_TOKEN"))
    print(json.dumps(s, ensure_ascii=False, indent=1))
