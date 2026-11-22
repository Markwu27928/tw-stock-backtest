"""資料來源：證交所（股價、三大法人、除權息）、公開資訊觀測站（月營收）、FinMind（月營收備援）。

所有解析函式只吃 dict / 字串，方便離線測試；網路存取集中在 http_get。
"""
from __future__ import annotations

import io
import random
import re
import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from . import config

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9",
}

_session = requests.Session()
_last_call = {"t": 0.0}


class FetchError(Exception):
    """連線或格式錯誤（不是「當天沒資料」）。"""


def http_get(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response:
    """節流後的 GET。測試時可 monkeypatch 這個函式。"""
    wait = config.REQUEST_SLEEP + random.uniform(0, 1.0) - (time.time() - _last_call["t"])
    if wait > 0:
        time.sleep(wait)
    try:
        r = _session.get(url, params=params, headers=HEADERS, timeout=timeout)
    except requests.RequestException as e:
        raise FetchError(f"{url}: {e}") from e
    finally:
        _last_call["t"] = time.time()
    if r.status_code != 200:
        raise FetchError(f"{url}: HTTP {r.status_code}")
    return r


def get_json(url: str, params: dict | None = None) -> dict:
    r = http_get(url, params)
    try:
        return r.json()
    except ValueError as e:  # 被擋時常回 HTML 錯誤頁
        raise FetchError(f"{url}: 回傳不是 JSON（可能被暫時封鎖）") from e


# ---------------------------------------------------------------- 共用工具
def to_num(x) -> float:
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace(",", "").strip()
    s = re.sub(r"<[^>]+>", "", s)          # 有些欄位含 HTML 標籤
    if s in ("", "--", "---", "-", "X", "N/A"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def roc_to_date(s: str) -> str | None:
    """'113年07月01日' 或 '113/07/01' → '2024-07-01'"""
    m = re.match(r"\s*(\d{2,3})\D+(\d{1,2})\D+(\d{1,2})", str(s))
    if not m:
        return None
    y, mth, d = int(m.group(1)) + 1911, int(m.group(2)), int(m.group(3))
    return f"{y:04d}-{mth:02d}-{d:02d}"


def _tables(js: dict):
    """證交所 JSON 有新版 tables[] 與舊版 fields/data、fieldsN/dataN 兩種格式。"""
    for t in js.get("tables") or []:
        if t.get("fields") and t.get("data") is not None:
            yield t["fields"], t["data"]
    if js.get("fields") and js.get("data") is not None:
        yield js["fields"], js["data"]
    for k in js:
        m = re.fullmatch(r"fields(\d+)", k)
        if m and js.get(f"data{m.group(1)}") is not None:
            yield js[k], js[f"data{m.group(1)}"]


def _has_data(js: dict) -> bool:
    return str(js.get("stat", "")).upper() == "OK"


# ---------------------------------------------------------------- 每日收盤行情
MI_INDEX_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"


def parse_mi_index(js: dict) -> pd.DataFrame | None:
    """回傳 code,name,close,volume,value；當天無資料（假日）回傳 None。"""
    if not _has_data(js):
        return None
    for fields, data in _tables(js):
        if "證券代號" in fields and "收盤價" in fields and data:
            idx = {f: i for i, f in enumerate(fields)}
            rows = []
            for r in data:
                rows.append({
                    "code": str(r[idx["證券代號"]]).strip(),
                    "name": str(r[idx.get("證券名稱", idx["證券代號"])]).strip(),
                    "close": to_num(r[idx["收盤價"]]),
                    "volume": to_num(r[idx["成交股數"]]) if "成交股數" in idx else np.nan,
                    "value": to_num(r[idx["成交金額"]]) if "成交金額" in idx else np.nan,
                })
            return pd.DataFrame(rows)
    return None


def fetch_prices(d: date) -> pd.DataFrame | None:
    js = get_json(MI_INDEX_URL, {"date": d.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"})
    return parse_mi_index(js)


# ---------------------------------------------------------------- 三大法人買賣超
T86_URL = "https://www.twse.com.tw/rwd/zh/fund/T86"


def parse_t86(js: dict) -> pd.DataFrame | None:
    """回傳 code, foreign_net, trust_net（股數）。"""
    if not _has_data(js):
        return None
    for fields, data in _tables(js):
        if "證券代號" not in fields:
            continue
        idx = {f: i for i, f in enumerate(fields)}
        f_col = next((f for f in fields if f.startswith("外陸資買賣超") or f.startswith("外資買賣超")), None)
        t_col = next((f for f in fields if f.startswith("投信買賣超")), None)
        if f_col is None or t_col is None:
            continue
        return pd.DataFrame({
            "code": [str(r[idx["證券代號"]]).strip() for r in data],
            "foreign_net": [to_num(r[idx[f_col]]) for r in data],
            "trust_net": [to_num(r[idx[t_col]]) for r in data],
        })
    return None


def fetch_t86(d: date) -> pd.DataFrame | None:
    js = get_json(T86_URL, {"date": d.strftime("%Y%m%d"), "selectType": "ALLBUT0999", "response": "json"})
    return parse_t86(js)


# ---------------------------------------------------------------- 除權息
TWT49U_URL = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U"


def parse_twt49u(js: dict) -> pd.DataFrame:
    """回傳 date, code, before, ref（除權息前收盤價、除權息參考價）。"""
    cols = ["date", "code", "before", "ref"]
    if not _has_data(js):
        return pd.DataFrame(columns=cols)
    for fields, data in _tables(js):
        if "股票代號" in fields and "除權息參考價" in fields:
            idx = {f: i for i, f in enumerate(fields)}
            out = pd.DataFrame({
                "date": [roc_to_date(r[idx["資料日期"]]) for r in data],
                "code": [str(r[idx["股票代號"]]).strip() for r in data],
                "before": [to_num(r[idx["除權息前收盤價"]]) for r in data],
                "ref": [to_num(r[idx["除權息參考價"]]) for r in data],
            })
            return out.dropna()
    return pd.DataFrame(columns=cols)


def fetch_dividends(start: date, end: date) -> pd.DataFrame:
    js = get_json(TWT49U_URL, {"startDate": start.strftime("%Y%m%d"),
                               "endDate": end.strftime("%Y%m%d"), "response": "json"})
    return parse_twt49u(js)


# ---------------------------------------------------------------- 月營收
MOPS_URLS = [
    "https://mopsov.twse.com.tw/nas/t21/sii/t21sc03_{roc}_{m}_0.html",
    "https://mops.twse.com.tw/nas/t21/sii/t21sc03_{roc}_{m}_0.html",
]


def _flat(col) -> str:
    if isinstance(col, tuple):
        return "".join(str(c) for c in col if not str(c).startswith("Unnamed"))
    return str(col)


def parse_mops_revenue(html: str) -> pd.DataFrame:
    """公開資訊觀測站上市公司月營收彙總表 → code, rev, rev_prev, rev_ly。"""
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return pd.DataFrame(columns=["code", "rev", "rev_prev", "rev_ly"])
    parts = []
    for t in tables:
        cols = [_flat(c) for c in t.columns]
        def find(*keys, exclude=()):
            for i, c in enumerate(cols):
                if all(k in c for k in keys) and not any(x in c for x in exclude):
                    return i
            return None
        ic = find("公司代號")
        ir = find("當月營收", exclude=("累計", "去年"))
        ip = find("上月營收")
        il = find("去年當月營收")
        if None in (ic, ir, il):
            continue
        df = pd.DataFrame({
            "code": t.iloc[:, ic].astype(str).str.strip(),
            "rev": t.iloc[:, ir].map(to_num),
            "rev_prev": t.iloc[:, ip].map(to_num) if ip is not None else np.nan,
            "rev_ly": t.iloc[:, il].map(to_num),
        })
        parts.append(df[df["code"].str.fullmatch(r"\d{4}")])
    if not parts:
        return pd.DataFrame(columns=["code", "rev", "rev_prev", "rev_ly"])
    return pd.concat(parts).drop_duplicates("code").reset_index(drop=True)


def fetch_revenue_mops(year: int, month: int) -> pd.DataFrame | None:
    """抓 year/month 的上市公司營收。全部網址都失敗才丟 FetchError；查無資料回 None。"""
    last_err = None
    for u in MOPS_URLS:
        url = u.format(roc=year - 1911, m=month)
        try:
            r = http_get(url)
        except FetchError as e:
            last_err = e
            continue
        html = r.content.decode("cp950", errors="ignore")
        df = parse_mops_revenue(html)
        if len(df):
            return df
    if last_err:
        raise last_err
    return None


OPENAPI_REV_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"


def parse_openapi_revenue(rows: list) -> tuple[str | None, pd.DataFrame]:
    """證交所 OpenAPI 只有最新一個月。回傳 ('YYYY-MM', df)。"""
    if not rows:
        return None, pd.DataFrame()
    def pick(row, *keys):
        for k, v in row.items():
            if all(x in k for x in keys):
                return v
        return None
    ym_raw = str(pick(rows[0], "資料年月") or "")
    ym = None
    if re.fullmatch(r"\d{5}", ym_raw):
        ym = f"{int(ym_raw[:3]) + 1911:04d}-{ym_raw[3:]}"
    df = pd.DataFrame({
        "code": [str(pick(r, "公司代號")).strip() for r in rows],
        "rev": [to_num(pick(r, "當月營收")) for r in rows],
        "rev_prev": [to_num(pick(r, "上月營收")) for r in rows],
        "rev_ly": [to_num(pick(r, "去年當月營收")) for r in rows],
    })
    return ym, df[df["code"].str.fullmatch(r"\d{4}")].reset_index(drop=True)


def fetch_revenue_openapi() -> tuple[str | None, pd.DataFrame]:
    r = http_get(OPENAPI_REV_URL)
    try:
        return parse_openapi_revenue(r.json())
    except ValueError as e:
        raise FetchError("OpenAPI 回傳不是 JSON") from e


FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


def parse_finmind_revenue(js: dict) -> pd.DataFrame:
    """FinMind TaiwanStockMonthRevenue → ym, code, rev（rev_ly 由歷史自行對齊）。"""
    rows = js.get("data") or []
    if not rows:
        return pd.DataFrame(columns=["ym", "code", "rev"])
    df = pd.DataFrame(rows)
    df["ym"] = df["revenue_year"].astype(int).astype(str) + "-" + df["revenue_month"].astype(int).map("{:02d}".format)
    return df.rename(columns={"stock_id": "code", "revenue": "rev"})[["ym", "code", "rev"]]


def fetch_revenue_finmind(code: str, start: str, token: str | None) -> pd.DataFrame:
    params = {"dataset": "TaiwanStockMonthRevenue", "data_id": code, "start_date": start}
    if token:
        params["token"] = token
    return parse_finmind_revenue(get_json(FINMIND_URL, params))
