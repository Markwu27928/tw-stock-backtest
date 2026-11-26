"""超跌反彈規則：訊號、兩半出場、持有中不重複進場、12:40 任務流程。"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src import config, market, mkt_sources as S, rule_rebound as RR, store, today_page


def _path(n=260, seed=1):
    """平穩 → 急跌（觸發）→ 反彈站上均線 → 再回落；再一次急跌。"""
    rng = np.random.default_rng(seed)
    seg = [np.full(80, 0.0005), np.full(12, -0.03), np.full(25, 0.012), np.full(20, -0.006),
           np.full(40, 0.001), np.full(12, -0.03), np.full(n - 189, 0.004)]
    r = np.concatenate(seg) + rng.normal(0, 0.002, n)
    return pd.Series(10000 * np.cumprod(1 + r), index=pd.bdate_range("2024-01-01", periods=n))


def test_entry_signal_needs_both_conditions_and_fires_once():
    s = _path()
    sig = RR.entry_signal(s)
    assert sig["new"].sum() >= 1
    for d in sig.index[sig["new"]]:
        assert sig.at[d, "rsi"] < 30 and sig.at[d, "index"] <= sig.at[d, "lower"]
    # 連續成立的日子只有第一天算新訊號
    assert not (sig["new"] & sig["new"].shift(1, fill_value=False)).any()


def test_replay_exits_and_no_pyramiding():
    s = _path()
    res = RR.replay(s, pd.DataFrame({"0050": s.values}, index=s.index))
    trades = [t for t in res["trades"] if t["asset"] == "0050"]
    assert trades
    t = trades[0]
    assert t["macd_exit"] and t["ma20_exit"] and "ret" in t
    assert t["macd_exit"]["date"] > t["entry_date"] and t["ma20_exit"]["date"] > t["entry_date"]
    # 兩半平均
    assert abs(t["ret"] - (t["macd_exit"]["ret"] + t["ma20_exit"]["ret"]) / 2) < 1e-12
    # 同一標的的交易不重疊
    for a, b in zip(trades, trades[1:]):
        last_exit = max(a["macd_exit"]["date"], a["ma20_exit"]["date"])
        assert b["entry_date"] > last_exit


def test_open_position_reports_exit_distances():
    s = _path()
    cut = s.iloc[:100]                                  # 急跌後不久就停：部位還在持有
    res = RR.replay(cut, pd.DataFrame({"0050": cut.values}, index=cut.index))
    op = res["open"]["0050"]
    assert op["open"] and not op["macd"]["sold"] and not op["ma20"]["sold"]
    assert len(op["macd"]["hist"]) == 3 and op["ma20"]["gap"] < 0


def test_midday_task_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path / "docs")
    s = _path(260).iloc[:100]                          # 急跌後不久：12:40 時部位持有中
    hist, today = s.iloc[:-1], s.index[-1]
    intr = pd.DataFrame({"date": hist.index.strftime("%Y-%m-%d"), "prev": hist.shift(1).bfill().values,
                         "open": hist.values, "at": hist.values, "hi_at": hist.values, "lo_at": hist.values,
                         "close": hist.values})
    intr.to_csv(market.mdir() / "intraday.csv", index=False)
    for d, v in hist.items():
        store.save_daily("price", d.date(), pd.DataFrame({"code": RR.ASSETS, "name": "x", "close": v / 200,
                                                          "volume": 1.0, "value": 1.0}))
    ymd = today.strftime("%Y%m%d")
    q = {"t00": {"price": float(s.iloc[-1]), "date": ymd, "time": "12:40:00"},
         **{a: {"price": float(s.iloc[-1]) / 200, "date": ymd} for a in RR.ASSETS}}
    monkeypatch.setattr(S, "fetch_mis", lambda codes: {c: q[c] for c in codes if c in q})
    monkeypatch.setenv("BT_FAKE_NOW", today.strftime("%Y-%m-%d") + "T12:40")
    market.task_midday()
    m = market.load_day(today.strftime("%Y-%m-%d"))["midday"]
    assert m["status"] == "ok" and m["final"] is False and 0 < m["rsi"] < 100
    assert set(m["open"]) <= set(RR.ASSETS)
    html = today_page.render()
    assert "超跌反彈規則" in html and "NaN" not in html
