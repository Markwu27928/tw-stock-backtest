"""盤勢預測用的資料來源。解析函式只吃字串 / dict，網路存取集中在 fetch_*。

- 證交所 MI_5MINS_INDEX：每 5 秒的加權指數（盤後提供，用來回補歷史與收盤）
- 證交所 MIS：盤中即時報價（加權指數 t00、個股）
- 證交所 BFI82U：三大法人買賣金額
- 期交所：台指期（TX）日盤／夜盤行情
- Yahoo Finance：美股（S&P 500、費城半導體、台積電 ADR）
"""
from __future__ import annotations

import io
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from . import sources
from .sources import FetchError, to_num

TWSE = "https://www.twse.com.tw/rwd/zh"
MIS = "https://mis.twse.com.tw/stock"
TAIFEX_DL = "https://www.taifex.com.tw/cht/3/futDataDown"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
US_SYMBOLS = {"spx": "^GSPC", "sox": "^SOX", "tsm": "TSM"}


# ---------------------------------------------------------------- 加權指數每 5 秒
def parse_5min_index(js: dict) -> pd.DataFrame | None:
    """回傳 time（'HH:MM:SS'）、idx 兩欄；沒資料回 None。"""
    if not js or js.get("stat") != "OK" or not js.get("data"):
        return None
    fields = js.get("fields") or []
    col = next((i for i, f in enumerate(fields) if "加權" in str(f)), 1)
    rows = [(str(r[0]).strip(), to_num(r[col])) for r in js["data"] if len(r) > col]
    df = pd.DataFrame(rows, columns=["time", "idx"]).dropna()
    return df if len(df) else None


def summarize_5min(df: pd.DataFrame, at: str = "09:40:00") -> dict:
    """一天的關鍵點：前日收盤（09:00:00 那筆）、開盤、09:40、盤中高低、收盤。"""
    df = df.sort_values("time")
    prev = float(df["idx"].iloc[0])                       # 09:00:00 尚未成交，等於前一日收盤
    after = df[df["time"] > "09:00:00"]
    open_ = float(after["idx"].iloc[0]) if len(after) else np.nan
    upto = df[df["time"] <= at]
    at_v = float(upto["idx"].iloc[-1]) if len(upto) else np.nan
    early = after[after["time"] <= at]
    return {"prev": prev, "open": open_, "at": at_v,
            "hi_at": float(early["idx"].max()) if len(early) else np.nan,
            "lo_at": float(early["idx"].min()) if len(early) else np.nan,
            "close": float(df["idx"].iloc[-1])}


# 證交所改版過網址，依序嘗試；第一個成功的記下來之後直接用
FIVE_MIN_URLS = [f"{TWSE}/TAIEX/MI_5MINS_INDEX",
                 "https://www.twse.com.tw/exchangeReport/MI_5MINS_INDEX",
                 f"{TWSE}/afterTrading/MI_5MINS_INDEX"]
_good = {}


def fetch_5min_index(d: date) -> pd.DataFrame | None:
    urls = [_good["5min"]] if "5min" in _good else FIVE_MIN_URLS
    last = None
    for u in urls:
        try:
            js = sources.get_json(u, {"date": d.strftime("%Y%m%d"), "response": "json"})
        except FetchError as e:
            last = e
            continue
        _good["5min"] = u
        return parse_5min_index(js)
    raise last or FetchError("MI_5MINS_INDEX：沒有可用網址")


# ---------------------------------------------------------------- MIS 即時報價
def _mis_price(q: dict) -> float:
    """最新成交價；這一筆沒成交時用最佳買賣價中間價，再不行用昨收。"""
    z = to_num(q.get("z"))
    if np.isfinite(z):
        return z
    b = to_num(str(q.get("b", "")).split("_")[0])
    a = to_num(str(q.get("a", "")).split("_")[0])
    if np.isfinite(b) and np.isfinite(a):
        return (a + b) / 2
    return to_num(q.get("y"))


def parse_mis(js: dict) -> dict[str, dict]:
    out = {}
    for q in (js or {}).get("msgArray") or []:
        c = str(q.get("c", "")).strip()
        if not c:
            continue
        out[c] = {"name": q.get("n", ""), "price": _mis_price(q), "prev": to_num(q.get("y")),
                  "open": to_num(q.get("o")), "high": to_num(q.get("h")), "low": to_num(q.get("l")),
                  "date": str(q.get("d", "")), "time": str(q.get("t", ""))}
    return out


_mis = requests.Session()


def fetch_mis(codes: list[str]) -> dict[str, dict]:
    """codes：'t00'（加權指數）或股票代號；上市用 tse_，抓不到再試上櫃 otc_。"""
    if not _mis.cookies:
        try:
            _mis.get(f"{MIS}/index.jsp", headers=sources.HEADERS, timeout=20)
        except requests.RequestException:
            pass
    out: dict[str, dict] = {}
    for i in range(0, len(codes), 40):
        chunk = codes[i:i + 40]
        ex = "|".join(f"tse_{c}.tw" for c in chunk)
        try:
            r = _mis.get(f"{MIS}/api/getStockInfo.jsp",
                         params={"ex_ch": ex, "json": "1", "delay": "0", "_": int(time.time() * 1000)},
                         headers=dict(sources.HEADERS, Referer=f"{MIS}/index.jsp"), timeout=20)
            out.update(parse_mis(r.json()))
        except (requests.RequestException, ValueError) as e:
            raise FetchError(f"MIS: {e}") from e
        time.sleep(1.0)
    return out


# ---------------------------------------------------------------- 三大法人買賣金額
def parse_bfi82u(js: dict) -> dict | None:
    if not js or js.get("stat") != "OK" or not js.get("data"):
        return None
    # 依名稱開頭分類：「外資及陸資(不含外資自營商)」含有「自營」二字，不能用「包含」判斷
    out = {}
    for r in js["data"]:
        name, net = str(r[0]).strip(), to_num(r[-1])
        if name.startswith("外資"):                  # 外資及陸資 + 外資自營商
            out["foreign"] = out.get("foreign", 0) + net
        elif name.startswith("投信"):
            out["trust"] = net
        elif name.startswith("自營商"):              # 自行買賣 + 避險
            out["dealer"] = out.get("dealer", 0) + net
        elif name.startswith("合計"):
            out["total"] = net
    return out or None


def fetch_bfi82u(d: date) -> dict | None:
    return parse_bfi82u(sources.get_json(f"{TWSE}/fund/BFI82U",
                                         {"dayDate": d.strftime("%Y%m%d"), "type": "day", "response": "json"}))


# ---------------------------------------------------------------- 期交所台指期
def parse_taifex_csv(text: str) -> pd.DataFrame:
    """台指期近月（非週選）每個交易日、每個時段一列：date, session('day'/'night'), close。"""
    df = pd.read_csv(io.StringIO(text), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    need = {"交易日期", "到期月份(週別)", "收盤價"}
    if not need <= set(df.columns):
        raise FetchError(f"期交所 CSV 欄位不符：{list(df.columns)[:8]}")
    df["契約"] = df.get("契約", "TX").astype(str).str.strip()
    df = df[df["契約"] == "TX"].copy()
    df["exp"] = df["到期月份(週別)"].astype(str).str.strip()
    df = df[df["exp"].str.fullmatch(r"\d{6}")]
    sess = df.get("交易時段", pd.Series("一般", index=df.index)).astype(str).str.strip()
    df["session"] = np.where(sess.str.contains("盤後"), "night", "day")
    df["date"] = pd.to_datetime(df["交易日期"].str.strip(), format="%Y/%m/%d").dt.strftime("%Y-%m-%d")
    df["close"] = df["收盤價"].map(to_num)
    df = df.dropna(subset=["close"])
    near = df.sort_values("exp").groupby(["date", "session"], as_index=False).first()
    return near[["date", "session", "close"]]


def fetch_taifex_month(s: date, e: date, tries: int = 3) -> pd.DataFrame:
    """期交所一次最多查一個月；失敗重試。"""
    last = None
    for k in range(tries):
        time.sleep(1.5 + 3 * k)
        try:
            r = requests.post(TAIFEX_DL, data={"down_type": "1", "commodity_id": "TX",
                                               "queryStartDate": s.strftime("%Y/%m/%d"),
                                               "queryEndDate": e.strftime("%Y/%m/%d")},
                              headers=dict(sources.HEADERS, Referer="https://www.taifex.com.tw/cht/3/futDailyMarketView"),
                              timeout=60)
        except requests.RequestException as ex:
            last = f"期交所：{ex}"
            continue
        if r.status_code != 200:
            last = f"期交所：HTTP {r.status_code}"
            continue
        text = r.content.decode("cp950", errors="replace")
        if "交易日期" not in text:            # 每個月都有交易日，拿不到 CSV 就是被擋或網址錯
            last = f"期交所回應不是 CSV：{text[:120]!r}"
            continue
        return parse_taifex_csv(text)
    raise FetchError(last or "期交所：未知錯誤")


def month_ranges(start: date, end: date) -> list[tuple[date, date]]:
    out, s = [], start
    while s <= end:
        e = min(end, (pd.Timestamp(s) + pd.offsets.MonthEnd(0)).date())
        out.append((s, e))
        s = e + timedelta(days=1)
    return out


# FinMind 台指期（期交所抓不到時的備援）
FINMIND = "https://api.finmindtrade.com/api/v4/data"


def parse_finmind_futures(js: dict) -> pd.DataFrame:
    rows = (js or {}).get("data") or []
    if not rows:
        return pd.DataFrame(columns=["date", "session", "close"])
    df = pd.DataFrame(rows)
    df["contract_date"] = df["contract_date"].astype(str).str.strip()
    df = df[df["contract_date"].str.fullmatch(r"\d{6}")]
    sess = df["trading_session"].astype(str) if "trading_session" in df else pd.Series("position", index=df.index)
    df["session"] = np.where(sess.str.contains("after"), "night", "day")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[df["close"] > 0]
    near = df.sort_values("contract_date").groupby(["date", "session"], as_index=False).first()
    return near[["date", "session", "close"]]


def fetch_finmind_futures(start: date, end: date, token: str | None) -> pd.DataFrame:
    out = []
    for y in range(start.year, end.year + 1):
        s, e = max(start, date(y, 1, 1)), min(end, date(y, 12, 31))
        time.sleep(1.0)
        try:
            r = requests.get(FINMIND, params={"dataset": "TaiwanFuturesDaily", "data_id": "TX",
                                              "start_date": s.isoformat(), "end_date": e.isoformat()},
                             headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=60)
        except requests.RequestException as ex:
            raise FetchError(f"FinMind 期貨：{ex}") from ex
        if r.status_code != 200:
            raise FetchError(f"FinMind 期貨：HTTP {r.status_code}")
        out.append(parse_finmind_futures(r.json()))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "session", "close"])


# ---------------------------------------------------------------- 美股
def parse_yahoo(js: dict) -> pd.Series:
    """回傳以美國交易日（YYYY-MM-DD）為索引的收盤價。"""
    try:
        res = js["chart"]["result"][0]
        ts = res["timestamp"]
        close = res["indicators"]["quote"][0]["close"]
        off = res["meta"].get("gmtoffset", -18000)
    except (KeyError, IndexError, TypeError) as e:
        raise FetchError(f"Yahoo 格式不符：{e}") from e
    idx = [datetime.fromtimestamp(t + off, timezone.utc).strftime("%Y-%m-%d") for t in ts]
    s = pd.Series(close, index=idx, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


def fetch_us(range_: str = "15y") -> pd.DataFrame:
    cols = {}
    for k, sym in US_SYMBOLS.items():
        try:
            r = requests.get(YAHOO.format(sym=sym), params={"range": range_, "interval": "1d"},
                             headers=sources.HEADERS, timeout=30)
            cols[k] = parse_yahoo(r.json())
        except (requests.RequestException, ValueError) as e:
            raise FetchError(f"Yahoo {sym}: {e}") from e
        time.sleep(1.0)
    return pd.DataFrame(cols).sort_index()
