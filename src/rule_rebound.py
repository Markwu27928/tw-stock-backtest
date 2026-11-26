"""使用者的「大盤超跌找反彈」規則（2016 年起回測後採用的版本）。

進場：加權指數 RSI(14) < 30 且收盤碰到或跌破布林下軌（20 日、2 個標準差），
      條件「新出現」的那天收盤買進；同一標的持有中不重複進場。
出場：分兩半各自追蹤
  - MACD：標的自己的 MACD 柱狀體轉正後，連續 2 天縮小 → 賣出這一半
  - 20 日線：標的收盤站上 20 日均線後，再跌破 → 賣出這一半
  兩半都賣完才算結束；最多持有 120 個交易日。
標的：0050 與觀察清單。

盤中（12:40）把即時價格當成「今天的收盤」暫時判斷；正式紀錄以收盤價為準。
所有狀態每次都從歷史資料重新推算，不另外存可能出錯的中間狀態。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSETS = ["0050", "2330", "2317", "2454", "1717"]
CAP = 120
GO_LIVE = "2026-11-27"       # 這天以後進場的算「上線後」的實際紀錄


# ---------------------------------------------------------------- 指標
def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    u = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    v = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + u / v)


def boll_lower(s: pd.Series, n: int = 20) -> pd.Series:
    return s.rolling(n, min_periods=n).mean() - 2 * s.rolling(n, min_periods=n).std()


def macd_hist(s: pd.Series) -> pd.Series:
    dif = s.ewm(span=12, adjust=False, min_periods=12).mean() - s.ewm(span=26, adjust=False, min_periods=26).mean()
    return dif - dif.ewm(span=9, adjust=False, min_periods=9).mean()


def entry_signal(index: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"index": index, "rsi": rsi(index), "lower": boll_lower(index)})
    out["cond"] = (out["rsi"] < 30) & (out["index"] <= out["lower"])
    out["new"] = out["cond"] & ~out["cond"].shift(1, fill_value=False)
    return out


# ---------------------------------------------------------------- 單一標的推算
def _legs(px: np.ndarray, h: np.ndarray, ma: np.ndarray, i: int):
    """從第 i 天收盤進場，回傳兩半的出場索引（None＝還沒出場）與目前狀態。"""
    n = len(px)
    armed_m = h[i] > 0
    armed_a = px[i] > ma[i]
    ex_m = ex_a = None
    for j in range(i + 1, min(n, i + CAP + 1)):
        if ex_m is None:
            if h[j] > 0:
                armed_m = True
            # 連續 2 天縮小要從進場當天起算，不能把進場前的日子算進去
            if armed_m and j - 2 >= i and h[j - 2] > 0 and h[j] < h[j - 1] < h[j - 2]:
                ex_m = j
        if ex_a is None:
            if px[j] > ma[j]:
                armed_a = True
            elif armed_a and px[j] < ma[j]:
                ex_a = j
        if ex_m is not None and ex_a is not None:
            break
    if i + CAP < n:
        ex_m = ex_m if ex_m is not None else i + CAP
        ex_a = ex_a if ex_a is not None else i + CAP
    return ex_m, ex_a, armed_m, armed_a


def replay(index: pd.Series, prices: pd.DataFrame) -> dict:
    """index：加權指數收盤（日期索引，最後一天可以是盤中暫定值）；prices：各標的收盤，同樣的日期索引。"""
    sig = entry_signal(index)
    dates = list(index.index)
    trades, status = [], {}
    for a in prices.columns:
        s = prices[a].ffill()
        px = s.values
        h = macd_hist(s).values
        ma = s.rolling(20, min_periods=20).mean().values
        busy = -1
        for i in np.flatnonzero(sig["new"].values):
            if i <= busy or not np.isfinite(px[i]):
                continue
            ex_m, ex_a, armed_m, armed_a = _legs(px, h, ma, i)
            leg = lambda j: None if j is None else {"date": str(dates[j])[:10], "price": float(px[j]),
                                                     "ret": float(px[j] / px[i] - 1)}
            t = {"asset": a, "entry_date": str(dates[i])[:10], "entry_price": float(px[i]),
                 "macd_exit": leg(ex_m), "ma20_exit": leg(ex_a),
                 "live": str(dates[i])[:10] >= GO_LIVE}
            done = [j for j in (ex_m, ex_a) if j is not None]
            if len(done) == 2:
                t["ret"] = float((px[ex_m] + px[ex_a]) / 2 / px[i] - 1)     # 兩半平均
                t["days"] = int(max(ex_m, ex_a) - i)
                busy = max(ex_m, ex_a)
            else:
                busy = len(px)                                          # 還在持有，後面的訊號不重複進場
                last = len(px) - 1
                t["open"] = True
                t["now_price"] = float(px[last])
                t["now_ret"] = float(px[last] / px[i] - 1)
                t["days"] = int(last - i)
                t["macd"] = {"hist": [float(x) for x in h[max(0, last - 2):last + 1]], "armed": bool(armed_m),
                             "sold": ex_m is not None}
                t["ma20"] = {"ma": float(ma[last]), "gap": float(px[last] / ma[last] - 1), "armed": bool(armed_a),
                             "sold": ex_a is not None}
            trades.append(t)
        status[a] = next((t for t in trades if t["asset"] == a and t.get("open")), None)
    last = sig.iloc[-1]
    return {"date": str(dates[-1])[:10], "rsi": float(last["rsi"]), "lower": float(last["lower"]),
            "index": float(last["index"]), "cond": bool(last["cond"]), "new": bool(last["new"]),
            "prev_cond": bool(sig["cond"].iloc[-2]) if len(sig) > 1 else False,
            "trades": trades, "open": {a: t for a, t in status.items() if t}}


def stats(trades: list[dict], cost_etf: float, cost_stk: float) -> dict:
    """已結束交易（兩半平均）的統計；回測與上線後分開。"""
    out = {}
    for label, sel in (("回測", lambda t: not t["live"]), ("上線後", lambda t: t["live"])):
        rs = [t["ret"] - (cost_etf if t["asset"] == "0050" else cost_stk) for t in trades if "ret" in t and sel(t)]
        out[label] = {"n": len(rs), "win": float(np.mean([r > 0 for r in rs])) if rs else None,
                      "avg": float(np.mean(rs)) if rs else None}
    return out
