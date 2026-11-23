"""模擬測試：用合成市場跑完整流程（抓資料 → 回測 → 報告），並檢查前視偏差。

執行：python -m pytest -q tests
"""
from __future__ import annotations

import shutil
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import backtest, config, report, sources, store, update_data
from src.features import build_panel, features_at, decision_dates
from tests.fake_world import World


# ---------------------------------------------------------------- 解析器
def test_parse_mi_index_formats():
    w = World(end="2015-03-31", n=20)
    df = sources.parse_mi_index(w._mi_index(date(2015, 3, 2)))
    assert len(df) == 21 and df["close"].isna().sum() == 1
    assert df.loc[df.code == "0050", "close"].iloc[0] > 0
    # 舊格式 fields9 / data9
    js = w._mi_index(date(2015, 3, 2))
    t = js.pop("tables")[1]
    js["fields9"], js["data9"] = t["fields"], t["data"]
    assert len(sources.parse_mi_index(js)) == 21
    assert sources.parse_mi_index({"stat": "很抱歉，沒有符合條件的資料!"}) is None


def test_parse_t86_old_and_new_field_names():
    w = World(end="2018-01-31", n=10)
    old = sources.parse_t86(w._t86(date(2017, 12, 15)))
    new = sources.parse_t86(w._t86(date(2018, 1, 2)))
    for df in (old, new):
        assert list(df.columns) == ["code", "foreign_net", "trust_net"]
        assert (df["foreign_net"] == 1000).all()      # 不能誤抓到「外資自營商」欄位


def test_parse_twt49u_real_sample():
    # 2024-07 證交所實際回應的欄位與前幾列
    js = {"stat": "OK", "fields": ["資料日期", "股票代號", "股票名稱", "除權息前收盤價", "除權息參考價", "權值+息值", "權/息",
                                   "漲停價格", "跌停價格", "開盤競價基準", "減除股利參考價", "詳細資料",
                                   "最近一次申報資料 季別/日期", "最近一次申報每股 (單位)淨值", "最近一次申報每股 (單位)盈餘"],
          "data": [["113年07月01日", "1101", "台泥", "34.20", "33.20", "1.000000", "息", "36.50", "29.90", "33.20",
                    "33.20", "1101,20240701", "x", "30.86", "0.38"]]}
    df = sources.parse_twt49u(js)
    assert df.iloc[0].tolist() == ["2024-07-01", "1101", 34.2, 33.2]


def test_parse_mops_html_multiheader():
    w = World(end="2015-03-31", n=12)
    html = w.http_get("https://x/nas/t21/sii/t21sc03_104_2_0.html").content.decode("cp950")
    df = sources.parse_mops_revenue(html)
    assert len(df) == 12 and df["code"].str.fullmatch(r"\d{4}").all()
    per = pd.Period("2015-02", "M")
    c = df.iloc[0]["code"]
    assert abs(df.iloc[0]["rev"] - round(w.rev.at[per, c])) <= 1


def test_parse_openapi_revenue():
    rows = [{"出表日期": "1151117", "資料年月": "11510", "公司代號": "1101", "公司名稱": "台泥", "產業別": "水泥工業",
             "營業收入-當月營收": "100", "營業收入-上月營收": "90", "營業收入-去年當月營收": "80",
             "營業收入-上月比較增減(%)": "11", "營業收入-去年同月增減(%)": "25"}]
    ym, df = sources.parse_openapi_revenue(rows)
    assert ym == "2026-10" and df.iloc[0][["rev", "rev_prev", "rev_ly"]].tolist() == [100, 90, 80]


# ---------------------------------------------------------------- 完整流程
@pytest.fixture(scope="module")
def world_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("bt")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "DATA_DIR", root / "data")
    mp.setattr(config, "DOCS_DIR", root / "docs")
    mp.setattr(config, "REQUEST_SLEEP", 0)
    mp.setattr(config, "MIN_TRAIN_MONTHS", 18)
    mp.setattr(config, "MIN_VALUE_20", 1_000_000)
    w = World()
    mp.setattr(sources, "http_get", w.http_get)
    mp.setenv("BT_FAKE_NOW", "2019-01-02T18:00")
    yield root, w, mp
    mp.undo()


def test_incremental_fetch_resumes(world_env, monkeypatch):
    root, w, _ = world_env
    # 第一次只允許 300 次請求，模擬時間用完
    calls = {"n": 0}
    orig_ok = update_data.Budget.ok
    def limited(self):
        calls["n"] += 1
        return calls["n"] < 300 and orig_ok(self)
    monkeypatch.setattr(update_data.Budget, "ok", limited)
    s1 = update_data.run(999)
    monkeypatch.setattr(update_data.Budget, "ok", orig_ok)
    assert s1["days_done"] < s1["days_total"]
    # 最近 20 天優先抓
    assert store.has_daily("price", date(2018, 12, 31))
    s2 = update_data.run(999)
    assert s2["days_done"] >= s2["days_total"] - 1   # 今天（2019-01-02）在合成世界沒有資料
    assert s2["contiguous_until"] == "2019-01-01"   # 元旦是假日
    st = store.State.load()
    assert st.is_holiday(date(2016, 2, 8)) and not st.is_holiday(date(2016, 2, 15))
    # 第三次不應再發請求（除了當年度除權息）
    n0 = len(w.calls)
    update_data.run(999)
    assert len(w.calls) - n0 <= 2


def test_today_not_marked_holiday_before_close(world_env, monkeypatch):
    monkeypatch.setenv("BT_FAKE_NOW", "2019-01-03T10:00")   # 世界資料只到 2018-12-31
    update_data.run(5)
    assert not store.State.load().is_holiday(date(2019, 1, 3))
    monkeypatch.setenv("BT_FAKE_NOW", "2019-01-03T18:00")   # 收盤後仍無資料 → 今天先不判定假日
    update_data.run(5)
    assert not store.State.load().is_holiday(date(2019, 1, 3))


def test_dividend_adjustment(world_env):
    p = build_panel()
    exd = pd.Timestamp(w_exd := next(d for d in p.dates if d >= pd.Timestamp("2016-07-15")))
    c = p.close.columns[10]
    raw = p.close.at[exd, c] / p.close.shift(1).at[exd, c] - 1
    adj = p.ret.at[exd, c]
    assert adj > raw + 0.02      # 還原後不應出現 3% 的假跌幅


def test_backtest_finds_planted_signal(world_env):
    root, w, _ = world_env
    p = build_panel()
    bt = backtest.run_backtest(p)
    res = bt["results"]
    assert res["A"] and res["C"]
    sa, sc = backtest.summarize(res["A"]), backtest.summarize(res["C"])
    print("A", sa, "\nC", sc)
    assert sa["pick_hit"] > sa["baseline_hit"] + 0.05
    assert sc["pick_hit"] > sc["baseline_hit"] + 0.05
    # 每月 10 檔、日期順序正確
    for r in res["B"]:
        assert len(r["picks"]) == config.TOP_N
        assert r["decision"] < r["entry"] < r["exit"]
    # 下市股票被納入（無存活者偏差），下市後不會再被選到
    months_with_delisted = [r for r in res["A"] if any(pk["code"] in w.delisted for pk in r["picks"])]
    assert all(r["decision"] < pd.Timestamp(w.delist_date) for r in months_with_delisted)
    data = report.build(p, bt, {"days_done": 10, "days_total": 10, "contiguous_until": "2018-12-31", "log": ["ok"]})
    report.write(data)
    html = (config.DOCS_DIR / "index.html").read_text("utf-8")
    assert "策略比較" in html and "結論" in html and "本月清單" in html and "NaN" not in html


def test_no_lookahead(world_env, tmp_path):
    """把 2017-12 以後的股價、法人、營收全部打亂，之前已結束月份的結果必須完全一樣。"""
    p = build_panel()
    base = backtest.run_backtest(p)
    cut = pd.Timestamp("2017-12-01")
    backup = tmp_path / "data"
    shutil.copytree(config.DATA_DIR, backup)
    try:
        rng = np.random.default_rng(1)
        for kind in ("price", "t86"):
            for f in (config.DATA_DIR / kind).glob("*/*.csv.gz"):
                if pd.Timestamp(f.name[:8]) >= cut:
                    df = pd.read_csv(f, dtype={"code": str})
                    for col in ("close", "trust_net", "foreign_net"):
                        if col in df:
                            df[col] = df[col] * rng.uniform(0.5, 1.5, len(df))
                    df.to_csv(f, index=False, compression="gzip")
        for f in (config.DATA_DIR / "rev").glob("*.csv.gz"):
            if pd.Period(f.name[:7], "M") >= pd.Period("2017-11", "M"):
                df = pd.read_csv(f, dtype={"code": str})
                df["rev"] = df["rev"] * rng.uniform(0.2, 3, len(df))
                df.to_csv(f, index=False, compression="gzip")
        p2 = build_panel()
        alt = backtest.run_backtest(p2)
        for s in base["results"]:
            a = [r for r in base["results"][s] if r["exit"] < cut]
            b = [r for r in alt["results"][s] if r["exit"] < cut]
            assert len(a) == len(b) and len(a) > 0
            for x, y in zip(a, b):
                assert [k["code"] for k in x["picks"]] == [k["code"] for k in y["picks"]], (s, x["month"])
                assert np.allclose([k["ret"] for k in x["picks"]], [k["ret"] for k in y["picks"]])
    finally:
        shutil.rmtree(config.DATA_DIR)
        shutil.copytree(backup, config.DATA_DIR)


def test_pure_noise_market_shows_no_edge(tmp_path, monkeypatch):
    """沒有任何訊號的市場：三個策略的勝率都應該接近隨機基準，不能憑空做出 6 成。"""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(config, "REQUEST_SLEEP", 0)
    monkeypatch.setattr(config, "MIN_TRAIN_MONTHS", 18)
    monkeypatch.setattr(config, "MIN_VALUE_20", 1_000_000)
    w = World(signal=0.0, trust_effect=0.0, seed=11)
    monkeypatch.setattr(sources, "http_get", w.http_get)
    monkeypatch.setenv("BT_FAKE_NOW", "2019-01-02T18:00")
    update_data.run(999)
    res = backtest.run_backtest(build_panel())["results"]
    for s, rows in res.items():
        sm = backtest.summarize(rows)
        print(s, round(sm["pick_hit"], 3), round(sm["baseline_hit"], 3), round(sm["t_stat"], 2))
        assert abs(sm["pick_hit"] - sm["baseline_hit"]) < 0.08
        assert sm["pick_hit"] < config.TARGET_HIT
    # 沒有訊號的市場，穩健度檢查不應判定任何策略「穩健」
    from src import robustness
    p = build_panel()
    rb = robustness.run(p, backtest.run_backtest(p), with_liquidity=False)
    assert all(v != "robust" for v in rb["verdict"].values()), rb["verdict"]


def test_verdict_levels():
    from src.report import verdict, _picks_table
    def data(**kw):
        sm = {"months": 60, "pick_hit": 0.55, "baseline_hit": 0.46, "t_stat": 1.0, "cagr": 0.05, "bench_cagr": 0.08}
        sm.update(kw)
        return {"config": {"target": 0.6, "bench": "0050"}, "summary_common": {"A": sm, "B": {}, "C": {}}}
    assert verdict(data(pick_hit=0.62, t_stat=2.5, cagr=0.12))["level"] == "go"
    assert verdict(data(pick_hit=0.62, t_stat=1.5, cagr=0.12))["level"] == "no"      # 不顯著不能說達標
    assert verdict(data(pick_hit=0.62, t_stat=2.5, cagr=0.03))["level"] == "watch" or \
        verdict(data(pick_hit=0.62, t_stat=2.5, cagr=0.03))["level"] == "no"         # 年化輸 0050 不能給 go
    assert verdict(data(pick_hit=0.56, t_stat=2.2, cagr=0.10))["level"] == "watch"
    assert verdict(data(months=12))["level"] == "wait"
    # 顏色依正負（紅漲綠跌），勝負另外用文字
    html = _picks_table([{"code": "1101", "name": "x", "ret": -0.05}, {"code": "1102", "name": "y", "ret": 0.03}],
                        True, -0.10)
    assert "class='down'>-5.0%" in html and "class='up'>+3.0%" in html
    assert html.count("贏 ") == 2


def test_empty_finmind_file_keeps_revenue_numeric(world_env, tmp_path):
    """沒有營收的股票（例如存託憑證）FinMind 檔只有表頭，不可把營收欄位變成 object 而讓 LightGBM 當掉。"""
    backup = tmp_path / "data"
    shutil.copytree(config.DATA_DIR, backup)
    try:
        fm_dir = config.DATA_DIR / "rev_finmind"
        fm_dir.mkdir(parents=True, exist_ok=True)
        # 真實情況：有資料的檔和只有表頭的空檔混在一起合併
        off = store.load_revenue()
        code = off["code"].iloc[0]
        sample = off[off["code"] == code][["ym", "code", "rev"]].head(24)
        sample.to_csv(fm_dir / f"{code}.csv.gz", index=False, compression="gzip")
        pd.DataFrame(columns=["ym", "code", "rev"]).to_csv(fm_dir / "9103.csv.gz", index=False, compression="gzip")
        rv = store.load_revenue()
        assert rv["rev"].dtype.kind == "f" and rv["rev_ly"].dtype.kind == "f"
        bt = backtest.run_backtest(build_panel())
        assert len(bt["results"]["C"]) > 0
    finally:
        shutil.rmtree(config.DATA_DIR)
        shutil.copytree(backup, config.DATA_DIR)


def test_report_status_warns_when_no_progress(capsys, tmp_path, monkeypatch):
    from src import run_daily
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    run_daily.report_status({
        "days_done": 1979, "days_total": 3100, "contiguous_until": "2022-07-07",
        "new_price_days": 0, "new_t86_days": 0, "stopped_early": False,
        "warnings": ["月營收逐檔回補暫停：FinMind 本小時額度已用完，下次執行會接續（已完成 10/20 檔）"],
    })
    out = capsys.readouterr().out
    assert "::warning title=回補沒有進度::" in out
    assert "::warning title=抓資料::月營收逐檔回補暫停" in out
    text = summary.read_text("utf-8")
    assert "1979 / 3100" in text and "0 天" in text


def test_finmind_hint_messages():
    assert "額度" in update_data.finmind_hint(sources.FetchError("x: HTTP 402"))
    assert "token" in update_data.finmind_hint(sources.FetchError("x: HTTP 401"))


# ---------------------------------------------------------------- 穩健度、等權重基準、實盤追蹤、資料品質
def test_equal_weight_baseline(world_env):
    bt = backtest.run_backtest(build_panel())
    rows = bt["results"]["A"]
    assert rows and all("ew_ret" in r for r in rows)
    sm = backtest.summarize(rows)
    assert np.isfinite(sm["ew_cagr"]) and 0 <= sm["ew_month_win"] <= 1


def test_robustness_variants(world_env):
    from src import robustness
    p = build_panel()
    bt = backtest.run_backtest(p)
    rb = robustness.run(p, bt, with_liquidity=False)
    for s, rows in rb["variants"].items():
        names = [r["name"] for r in rows]
        assert names[:3] == ["成本 ×1", "成本 ×1.5", "成本 ×2"] and "選 5 檔" in names and "選 20 檔" in names
        cagr = [r["cagr"] for r in rows[:3]]
        assert cagr[0] > cagr[1] > cagr[2]          # 成本越高報酬越低
        assert rb["verdict"][s] in ("robust", "fragile", "none")
    # 合成市場裡 A 有植入訊號，成本 ×1 應贏隨機選股
    assert rb["variants"]["A"][0]["beats_random"]


def test_liquidity_variant_restores_config(world_env):
    from src import robustness
    p = build_panel()
    before = config.MIN_VALUE_20
    out = robustness.liquidity(p, mults=(2.0,))
    assert config.MIN_VALUE_20 == before
    assert out["A"] and out["A"][0]["name"] == "流動性門檻 ×2"


def test_live_log_freeze_is_permanent(world_env):
    from src import live_log
    p = build_panel()
    bt = backtest.run_backtest(p)
    last = bt["results"]["A"][-1]
    fake_live = {"month": last["month"], "decision": last["decision"], "entry": last["entry"],
                 "scores": {s: pd.Series({pk["code"]: pk["score"] for pk in bt["results"][s][-1]["picks"]})
                            for s in bt["results"]}}
    log = {}
    assert live_log.freeze(log, fake_live, today="2018-12-01")
    frozen = list(log[last["month"]]["picks"]["A"])
    # 之後分數改變，也不能改寫已凍結的名單
    changed = dict(fake_live, scores={s: -sc for s, sc in fake_live["scores"].items()})
    assert not live_log.freeze(log, changed)
    assert log[last["month"]]["picks"]["A"] == frozen
    ev = live_log.evaluate(log, p, bt["schedule"])
    assert ev[0]["status"] == "closed"
    # 凍結名單的報酬要和回測同月份一致（同樣的進出場日與成本）
    assert abs(ev[0]["strategies"]["A"]["port_ret"] - last["port_ret"]) < 1e-9


def test_quality_checks(world_env, tmp_path):
    from src import quality
    p = build_panel()
    q = quality.check(p, {"days_done": 90, "days_total": 100, "contiguous_until": "2018-06-01"},
                      today=date(2019, 3, 1))
    msgs = " ".join(x["msg"] for x in q)
    assert "尚未補完" in msgs and "沒有更新" in msgs
    assert not any(x["level"] == "error" for x in q)


def test_report_shows_new_sections(world_env):
    from src import live_log, quality, robustness
    p = build_panel()
    bt = backtest.run_backtest(p)
    data = report.build(p, bt, {"days_done": 10, "days_total": 10, "contiguous_until": "2018-12-31", "log": ["ok"]})
    data["robustness"] = report._clean(robustness.run(p, bt, with_liquidity=False))
    data["live_track"] = report._clean(live_log.evaluate({}, p, bt["schedule"]))
    data["quality"] = quality.check(p, {}, today=date(2019, 1, 2))
    html = report.render(data)
    assert "穩健度檢查" in html and "實盤追蹤" in html and "等權重年化" in html and "NaN" not in html


def test_strategy_a_excludes_extreme_yoy():
    f = pd.DataFrame({"rev_yoy": [0.5, 0.99, 1.0, 1.01, 4.0, np.nan]}, index=list("abcdef"))
    s = backtest.score_a(f)
    assert s.dropna().index.tolist() == ["a", "b", "c"]


def test_report_shows_recent_period(world_env):
    p = build_panel()
    bt = backtest.run_backtest(p)
    data = report.build(p, bt, {"days_done": 10, "days_total": 10, "contiguous_until": "2018-12-31", "log": []})
    assert all(len(backtest.common_period(bt["results"])[s]) >= data["summary_recent"][s]["months"] for s in "ABC")
    assert "近 4 年" in report.render(data)
