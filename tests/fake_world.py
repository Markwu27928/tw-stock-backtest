"""合成台股市場：模擬證交所 / 觀測站 API 回應，讓整條流程能離線跑。

植入的訊號：月營收年增率高的股票，公布後下一個月有正向漂移；投信買超也有小幅正向效果。
"""
from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

HOLIDAYS = {date(2015, 2, 16), date(2015, 2, 17), date(2015, 2, 18), date(2015, 2, 19), date(2015, 2, 20),
            date(2016, 2, 8), date(2016, 2, 9), date(2016, 2, 10), date(2016, 2, 11), date(2016, 2, 12),
            date(2017, 1, 27), date(2017, 1, 30), date(2017, 1, 31), date(2018, 2, 15), date(2018, 2, 16)}


class World:
    def __init__(self, start="2015-01-05", end="2018-12-31", n=260, signal=0.035, seed=7, trust_effect=0.0008):
        rng = np.random.default_rng(seed)
        days = [d.date() for d in pd.bdate_range(start, end) if d.date() not in HOLIDAYS]
        self.days = days
        codes = [str(1101 + i * 7) for i in range(n)]
        codes = [c for c in codes if not c.startswith("91")][:n]
        self.codes = codes
        months = pd.period_range(pd.Period(start, "M") - 16, pd.Period(end, "M"), freq="M")

        # 營收：每檔一個持續的成長率過程
        g = np.zeros((len(months), len(codes)))
        for t in range(1, len(months)):
            g[t] = 0.85 * g[t - 1] + rng.normal(0, 0.12, len(codes))
        base = rng.uniform(1e5, 5e6, len(codes))
        rev = np.zeros_like(g)
        for t in range(len(months)):
            ly = rev[t - 12] if t >= 12 else base * (1 + rng.normal(0, 0.05, len(codes)))
            rev[t] = np.maximum(ly * (1 + g[t]) * (1 + rng.normal(0, 0.03, len(codes))), 1000)
        self.rev = pd.DataFrame(rev, index=months, columns=codes)
        self.rev_ly = self.rev.shift(12).fillna(self.rev / 1.05)
        yoy = (self.rev / self.rev_ly - 1)

        # 每日報酬：大盤 + 營收漂移（營收公布後才生效）+ 雜訊
        T = len(days)
        mkt = rng.normal(0.0004, 0.01, T)
        z = yoy.sub(yoy.mean(axis=1), axis=0).div(yoy.std(axis=1), axis=0).clip(-3, 3)
        drift = np.zeros((T, len(codes)))
        for i, d in enumerate(days):
            per = pd.Period(d, "M") - (1 if d.day >= 11 else 2)   # 已公告的最新月份
            drift[i] = signal * z.loc[per].values / 21
        trust_sig = rng.normal(0, 1, (T, len(codes)))
        ret = mkt[:, None] * rng.uniform(0.6, 1.4, len(codes)) + drift + rng.normal(0, 0.018, (T, len(codes)))
        ret[1:] += trust_effect * trust_sig[:-1]
        ret = np.clip(ret, -0.095, 0.095)
        px = 30 * np.exp(rng.normal(0, 0.6, len(codes))) * np.cumprod(1 + ret, axis=0)

        # 除息：每年 7/15 後第一個交易日，股利 3%
        self.div_rows = []
        for y in sorted({d.year for d in days}):
            exd = next((d for d in days if d >= date(y, 7, 15)), None)
            if exd is None:
                continue
            i = days.index(exd)
            before = px[i - 1].copy()
            ref = before * 0.97
            px[i:] *= 0.97
            for k, c in enumerate(codes):
                self.div_rows.append((exd, c, before[k], ref[k]))
        # 0050：大盤
        b = 70 * np.cumprod(1 + mkt)
        self.close = pd.DataFrame(np.round(px, 2), index=days, columns=codes)
        self.close["0050"] = np.round(b, 2)
        self.volume = pd.DataFrame(rng.integers(5e5, 8e6, (T, len(codes) + 1)), index=days, columns=self.close.columns)
        self.trust = pd.DataFrame((trust_sig * 2e5).round(), index=days, columns=codes)
        # 下市：最後 5 檔在 2017-06 之後消失
        self.delisted = set(codes[-5:])
        self.delist_date = date(2017, 6, 1)
        self.calls = []

    # ---------------- 模擬 API
    def _mi_index(self, d):
        if d not in self.close.index:
            return {"stat": "很抱歉，沒有符合條件的資料!"}
        rows = []
        for c in self.close.columns:
            if c in self.delisted and d >= self.delist_date:
                continue
            cl = self.close.at[d, c]
            v = int(self.volume.at[d, c])
            rows.append([c, f"名稱{c}", f"{v:,}", "100", f"{int(v * cl):,}", "1", "1", "1", f"{cl:,.2f}",
                         "+", "0.1", "1", "1", "1", "1", "10"])
        # 偶爾沒成交
        if rows:
            rows[3][8] = "--"
        fields = ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價",
                  "漲跌(+/-)", "漲跌價差", "最後揭示買價", "最後揭示買量", "最後揭示賣價", "最後揭示賣量", "本益比"]
        return {"stat": "OK", "tables": [
            {"title": "價格指數", "fields": ["指數", "收盤指數"], "data": [["發行量加權股價指數", "10,000"]]},
            {"title": f"{d}每日收盤行情", "fields": fields, "data": rows}]}

    def _t86(self, d):
        if d not in self.close.index:
            return {"stat": "很抱歉，沒有符合條件的資料!"}
        old = d < date(2017, 12, 18)
        fcol = "外資買賣超股數" if old else "外陸資買賣超股數(不含外資自營商)"
        fields = ["證券代號", "證券名稱", "外陸資買進股數(不含外資自營商)", fcol, "外資自營商買賣超股數",
                  "投信買進股數", "投信買賣超股數", "自營商買賣超股數", "三大法人買賣超股數"]
        rows = [[c, f"名稱{c}", "0", "1,000", "-5", "0", f"{int(self.trust.at[d, c]):,}", "0", "0"]
                for c in self.codes if not (c in self.delisted and d >= self.delist_date)]
        return {"stat": "OK", "fields": fields, "data": rows}

    def _twt49u(self, s, e):
        rows = [[f"{d.year - 1911}年{d.month:02d}月{d.day:02d}日", c, f"名稱{c}", f"{b:.2f}", f"{r:.2f}", "1", "息"]
                + ["0"] * 8 for d, c, b, r in self.div_rows if s <= d <= e]
        fields = ["資料日期", "股票代號", "股票名稱", "除權息前收盤價", "除權息參考價", "權值+息值", "權/息", "漲停價格",
                  "跌停價格", "開盤競價基準", "減除股利參考價", "詳細資料", "最近一次申報資料 季別/日期",
                  "最近一次申報每股 (單位)淨值", "最近一次申報每股 (單位)盈餘"]
        return {"stat": "OK", "fields": fields, "data": rows}

    def _mops_html(self, y, m):
        per = pd.Period(f"{y}-{m:02d}", "M")
        if per not in self.rev.index:
            return "<html><body>查無資料</body></html>"
        half = len(self.codes) // 2
        out = ["<html><head><title>營業收入統計表</title></head><body>"]
        for ind, cs in (("水泥工業", self.codes[:half]), ("電子工業", self.codes[half:])):
            out.append(f"<table><tr><th>產業別：{ind}</th></tr></table>")
            out.append("<table border=1><tr><th rowspan=2>公司代號</th><th rowspan=2>公司名稱</th>"
                       "<th colspan=5>營業收入</th><th colspan=3>累計營業收入</th><th rowspan=2>備註</th></tr>"
                       "<tr><th>當月營收</th><th>上月營收</th><th>去年當月營收</th><th>上月比較增減(%)</th>"
                       "<th>去年同月增減(%)</th><th>當月累計營收</th><th>去年累計營收</th><th>前期比較增減(%)</th></tr>")
            for c in cs:
                r = self.rev.at[per, c]
                p = self.rev.at[per - 1, c] if per - 1 in self.rev.index else r
                l = self.rev_ly.at[per, c]
                out.append(f"<tr><td>{c}</td><td>名稱{c}</td><td>{r:,.0f}</td><td>{p:,.0f}</td><td>{l:,.0f}</td>"
                           f"<td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>-</td></tr>")
            out.append("<tr><td>合計</td><td></td><td>1</td><td>1</td><td>1</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>")
            out.append("</table>")
        out.append("</body></html>")
        return "".join(out)

    def http_get(self, url, params=None, timeout=30):
        import json as _json
        params = params or {}
        self.calls.append(url)
        if "MI_INDEX" in url:
            body = _json.dumps(self._mi_index(_d(params["date"]))).encode()
        elif "T86" in url:
            body = _json.dumps(self._t86(_d(params["date"]))).encode()
        elif "TWT49U" in url:
            body = _json.dumps(self._twt49u(_d(params["startDate"]), _d(params["endDate"]))).encode()
        elif "t21sc03" in url:
            name = url.rsplit("/", 1)[1]
            _, roc, m, _ = name.replace(".html", "").split("_")
            body = self._mops_html(int(roc) + 1911, int(m)).encode("cp950")
        elif "openapi" in url:
            body = b"[]"
        else:
            raise AssertionError(url)
        return SimpleNamespace(status_code=200, content=body,
                               json=lambda b=body: _json.loads(b.decode()), text=body.decode("cp950", "ignore"))


def _d(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
