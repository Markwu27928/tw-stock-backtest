"""盤勢功能的離線測試：解析器、模型（不偷看未來）、任務流程。"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import config, live_log, market, mkt_models as M, mkt_sources as S, today_page


# ---------------------------------------------------------------- 解析器
def test_parse_5min_and_summary():
    js = {"stat": "OK", "fields": ["時間", "發行量加權股價指數", "未含金融保險股指數"],
          "data": [["09:00:00", "20,000.00", "1"], ["09:00:05", "20,100.00", "1"], ["09:20:00", "19,950.00", "1"],
                   ["09:40:00", "20,050.00", "1"], ["13:30:00", "20,200.00", "1"]]}
    s = S.summarize_5min(S.parse_5min_index(js))
    assert s == {"prev": 20000.0, "open": 20100.0, "at": 20050.0, "hi_at": 20100.0, "lo_at": 19950.0, "close": 20200.0}
    assert S.parse_5min_index({"stat": "很抱歉，沒有符合條件的資料"}) is None


def test_parse_mis_falls_back_to_bid_ask():
    js = {"msgArray": [{"c": "t00", "n": "發行量加權股價指數", "z": "20123.45", "y": "20000", "o": "20050",
                        "h": "20200", "l": "19990", "d": "20261124", "t": "09:40:05"},
                       {"c": "2330", "n": "台積電", "z": "-", "b": "2000_1995_", "a": "2005_2010_", "y": "1990"}]}
    q = S.parse_mis(js)
    assert q["t00"]["price"] == 20123.45 and q["t00"]["date"] == "20261124"
    assert q["2330"]["price"] == 2002.5


def test_parse_bfi82u():
    js = {"stat": "OK", "data": [["自營商(自行買賣)", "1", "2", "-1,000"], ["自營商(避險)", "1", "2", "500"],
                                 ["投信", "1", "2", "2,000"], ["外資及陸資(不含外資自營商)", "1", "2", "-30,000"],
                                 ["外資自營商", "0", "0", "0"], ["合計", "1", "2", "-28,500"]]}
    f = S.parse_bfi82u(js)
    assert f == {"dealer": -500.0, "trust": 2000.0, "foreign": -30000.0, "total": -28500.0}


def test_parse_taifex_picks_near_month_not_weekly():
    csv = ("交易日期,契約,到期月份(週別),開盤價,最高價,最低價,收盤價,交易時段\n"
           "2026/11/23,TX,202612,1,1,1,20000,一般\n2026/11/23,TX,202611W4,1,1,1,19990,一般\n"
           "2026/11/23,TX,202701,1,1,1,20100,一般\n2026/11/24,TX,202612,1,1,1,20150,盤後\n"
           "2026/11/23,MTX,202612,1,1,1,1,一般\n")
    df = S.parse_taifex_csv(csv)
    assert df.to_dict("records") == [{"date": "2026-11-23", "session": "day", "close": 20000.0},
                                     {"date": "2026-11-24", "session": "night", "close": 20150.0}]


def test_parse_finmind_futures():
    js = {"data": [{"date": "2026-11-23", "futures_id": "TX", "contract_date": "202612", "close": 20000, "trading_session": "position"},
                   {"date": "2026-11-23", "futures_id": "TX", "contract_date": "202612/202701", "close": 5, "trading_session": "position"},
                   {"date": "2026-11-23", "futures_id": "TX", "contract_date": "202701", "close": 20100, "trading_session": "position"},
                   {"date": "2026-11-24", "futures_id": "TX", "contract_date": "202612", "close": 20150, "trading_session": "after_market"}]}
    df = S.parse_finmind_futures(js)
    assert df.to_dict("records") == [{"date": "2026-11-23", "session": "day", "close": 20000.0},
                                     {"date": "2026-11-24", "session": "night", "close": 20150.0}]


def test_parse_yahoo():
    js = {"chart": {"result": [{"meta": {"gmtoffset": -18000}, "timestamp": [1763643000, 1763729400],
                                "indicators": {"quote": [{"close": [6600.5, None]}]}}]}}
    s = S.parse_yahoo(js)
    assert list(s.values) == [6600.5] and len(s.index[0]) == 10


# ---------------------------------------------------------------- 模型
def _fake_intraday(n=900, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-06-01", periods=n).strftime("%Y-%m-%d")
    prev = 10000.0
    rows = []
    for d in days:
        gap = rng.normal(0, 0.004)
        r_at = gap + rng.normal(0, 0.004)
        ret = 0.8 * r_at + rng.normal(0, 0.004)          # 收盤和 09:40 高度相關
        rows.append({"date": d, "prev": prev, "open": prev * (1 + gap), "at": prev * (1 + r_at),
                     "hi_at": prev * (1 + max(gap, r_at) + 0.001), "lo_at": prev * (1 + min(gap, r_at) - 0.001),
                     "close": prev * (1 + ret)})
        prev = prev * (1 + ret)
    return pd.DataFrame(rows)


def test_intraday_model_beats_naive_up():
    ds = M.intraday_dataset(_fake_intraday(1500))
    st = M.evaluate(ds, M.walk_forward(ds, M.INTRA_FEATS, M.INTRA_RULES))
    assert st["sign_at"]["acc"] > st["up"]["acc"] + 0.1
    assert M.choose(st) in ("sign_at", "logit")


def test_walk_forward_has_no_lookahead():
    ds = M.intraday_dataset(_fake_intraday(1500))
    base = M.walk_forward(ds, M.INTRA_FEATS, M.INTRA_RULES)
    cut = "2020-06-01"
    alt = ds.copy()
    fut = alt.index >= cut
    alt.loc[fut, "y"] = 1 - alt.loc[fut, "y"]           # 把未來的答案全部反過來
    alt.loc[fut, "ret"] = -alt.loc[fut, "ret"]
    p2 = M.walk_forward(alt, M.INTRA_FEATS, M.INTRA_RULES)
    before = base.index < cut
    assert base.loc[before, "logit"].notna().sum() > 200          # 確實有模型預測可比
    assert np.allclose(base.loc[before, "logit"].fillna(-1), p2.loc[before, "logit"].fillna(-1))


@pytest.mark.parametrize("true_lag", [0, 1])
def test_night_alignment_detects_label_convention(true_lag):
    intr = _fake_intraday(400)
    days = list(intr["date"])
    rows = []
    for i, d in enumerate(days):
        rows.append({"date": d, "session": "day", "close": intr.loc[i, "prev"] * 1.0})
        if i + 1 < len(days):
            nxt = intr.loc[i + 1]
            night_close = intr.loc[i, "prev"] * (nxt["open"] / nxt["prev"])     # 夜盤反映隔天跳空
            rows.append({"date": days[i + 1] if true_lag == 0 else d, "session": "night", "close": night_close})
    tx = pd.DataFrame(rows)
    # 日盤收盤用當天收盤
    tx.loc[tx["session"] == "day", "close"] = intr["close"].values
    assert M.night_alignment(tx, intr) == true_lag


def test_close_forecast_interval_covers_about_80pct():
    ds = M.intraday_dataset(_fake_intraday(1500))
    bt = M.close_forecast_backtest(ds)
    assert 0.7 < bt["coverage_80"] < 0.9 and bt["n"] > 300


# ---------------------------------------------------------------- 任務流程
@pytest.fixture()
def mkt_env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path / "docs")
    intr = _fake_intraday(700)
    md = market.mdir()
    intr.to_csv(md / "intraday.csv", index=False)
    days = list(intr["date"])
    tx = [{"date": d, "session": "day", "close": c} for d, c in zip(days, intr["close"])]
    tx += [{"date": days[i + 1], "session": "night", "close": intr.loc[i, "close"] * (1 + 0.5 * (intr.loc[i + 1, "open"] / intr.loc[i + 1, "prev"] - 1))}
           for i in range(len(days) - 1)]
    pd.DataFrame(tx).to_csv(md / "taifex.csv", index=False)
    rng = np.random.default_rng(3)
    us = pd.DataFrame({"date": days, "spx": 5000 * np.cumprod(1 + rng.normal(0, .01, len(days))),
                       "sox": 4000 * np.cumprod(1 + rng.normal(0, .02, len(days))),
                       "tsm": 150 * np.cumprod(1 + rng.normal(0, .02, len(days)))})
    us.to_csv(md / "us.csv", index=False)
    for name in ("update_taifex", "update_us", "catch_up", "update_intraday"):
        monkeypatch.setattr(market, name, lambda *a, **k: 0)
    return intr


def test_preopen_task_end_to_end(mkt_env, monkeypatch):
    last = mkt_env["date"].iloc[-1]
    today = (pd.Timestamp(last) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    # 今天的夜盤（標在今天）
    tx = pd.read_csv(market.mdir() / "taifex.csv")
    tx.loc[len(tx)] = {"date": today, "session": "night", "close": float(mkt_env["close"].iloc[-1]) * 1.01}
    tx.to_csv(market.mdir() / "taifex.csv", index=False)
    monkeypatch.setenv("BT_FAKE_NOW", today + "T08:20")
    market.task_preopen()
    rec = market.load_day(today)
    p = rec["preopen"]
    assert p["status"] == "ok" and 0 <= p["prob_up"] <= 1 and p["method"] in M.NAMES
    assert abs(p["inputs"]["night_ret"] - 0.01) < 1e-9
    # 重複觸發不會重做
    market.task_preopen()
    assert market.load_day(today)["preopen"]["made_at"] == p["made_at"]


def test_intraday_and_close_tasks(mkt_env, monkeypatch):
    today = (pd.Timestamp(mkt_env["date"].iloc[-1]) + pd.offsets.BDay(1)).date()
    ymd = today.strftime("%Y%m%d")
    q = {"t00": {"name": "加權", "price": 10100.0, "prev": 10000.0, "open": 10050.0, "high": 10120.0,
                 "low": 10030.0, "date": ymd, "time": "09:40:00"}}
    monkeypatch.setattr(S, "fetch_mis", lambda codes: q)
    monkeypatch.setenv("BT_FAKE_NOW", today.isoformat() + "T09:40")
    market.task_intraday()
    i = market.load_day(today.isoformat())["intraday"]
    assert i["status"] == "ok" and i["close_lo"] <= i["close_mid"] <= i["close_hi"]
    # 收盤：全市場行情抓不到時只整理指數，並和預測對答案
    q["t00"] = dict(q["t00"], price=9950.0, time="13:30:00")
    monkeypatch.setattr(market.sources, "get_json", lambda *a, **k: {"stat": "no"})
    monkeypatch.setenv("BT_FAKE_NOW", today.isoformat() + "T13:45")
    market.task_close(wait_min=0)
    rec = market.load_day(today.isoformat())
    assert rec["close"]["status"] == "ok" and rec["intraday"]["actual_up"] is False
    assert rec["intraday"]["hit"] == (rec["intraday"]["pred_up"] is False)
    html = today_page.render()
    assert "今日盤勢" in html and "盤中收盤預測" in html and "NaN" not in html


def test_holiday_is_skipped(mkt_env, monkeypatch):
    monkeypatch.setattr(S, "fetch_mis", lambda codes: {"t00": {"date": "20000101", "price": 1, "prev": 1}})
    monkeypatch.setenv("BT_FAKE_NOW", "2026-12-25T09:40")
    market.task_intraday()
    assert market.load_day("2026-12-25")["intraday"]["status"] == "closed"
    monkeypatch.setenv("BT_FAKE_NOW", "2026-11-28T08:20")      # 週六
    market.task_preopen()
    assert "preopen" not in market.load_day("2026-11-28")


def test_rule_probability_is_historical_rate():
    """規則型方法不可回傳 0/1，要回傳歷史上同樣訊號時實際收漲的比例。"""
    ds = M.intraday_dataset(_fake_intraday(1500))
    x = {"r_at": -0.01, "gap": -0.005, "mom": -0.005, "rng": 0.01}
    pr = market._prob_for("sign_at", ds, M.INTRA_FEATS, x, M.INTRA_RULES)
    # 同方向（跌）且同幅度（0.01 → 大於 1% 這組）的歷史比例
    down_days = ds[(ds["r_at"] <= 0) & (ds["r_at"].abs() >= 0.01)]
    assert pr["pred_up"] is False and pr["prob_kind"] == "history"
    assert abs(pr["prob_up"] - down_days["y"].mean()) < 1e-12 and pr["prob_n"] == len(down_days)
    assert 0 < pr["prob_up"] < 0.5


def test_close_interval_scales_with_volatility():
    ds = M.intraday_dataset(_fake_intraday(1500))
    calm = M.close_forecast(ds, {"r_at": 0.0, "gap": 0.0, "rng": 0.002})
    wild = M.close_forecast(ds, {"r_at": 0.0, "gap": 0.0, "rng": 0.02})
    assert (wild["hi"] - wild["lo"]) > 5 * (calm["hi"] - calm["lo"])


def test_probability_uses_same_magnitude_bucket():
    """小幅度時要用同幅度的歷史比例，並回報這個幅度下的命中率。"""
    ds = M.intraday_dataset(_fake_intraday(1500))
    small = market._prob_for("sign_at", ds, M.INTRA_FEATS, {"r_at": -0.001, "gap": 0, "mom": 0, "rng": 0.005}, M.INTRA_RULES)
    big = market._prob_for("sign_at", ds, M.INTRA_FEATS, {"r_at": -0.02, "gap": 0, "mom": 0, "rng": 0.02}, M.INTRA_RULES)
    assert small["bucket"] == "小於 0.2%" and big["bucket"] == "大於 1%"
    hist = ds.dropna(subset=["y"])
    sel = (hist["r_at"] <= 0) & (hist["r_at"].abs() < 0.002)
    assert abs(small["prob_up"] - hist.loc[sel, "y"].mean()) < 1e-12 and small["prob_n"] == int(sel.sum())
    assert big["bucket_acc"] > small["bucket_acc"]          # 幅度大時命中率較高


def test_rerun_keeps_original_time():
    rec = {}
    market.put(rec, "close", {"status": "ok", "made_at": "13:47", "index": 1})
    market.put(rec, "close", {"status": "ok", "made_at": "22:25", "index": 2})
    assert rec["close"]["made_at"] == "13:47" and rec["close"]["rerun_at"] == "22:25" and rec["close"]["index"] == 2
    market.put(rec, "midday", {"status": "closed", "made_at": "12:40"})
    market.put(rec, "midday", {"status": "ok", "made_at": "12:50"})
    assert rec["midday"]["made_at"] == "12:50" and "rerun_at" not in rec["midday"]
