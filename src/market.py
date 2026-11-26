"""每日盤勢：開盤前預測（08:20）、盤中收盤預測（09:40）、收盤整理（13:45）、三大法人（17:20）。

python -m src.market preopen | intraday | close | flows | backfill | render [--force]

- 每個任務先檢查今天是否已完成，完成就略過（排程重複觸發也不會產生兩份）
- 非交易日自動略過
- 結果存在 data/market/days/YYYY-MM-DD.json，頁面輸出到 docs/today.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config, live_log, mkt_models as M, mkt_sources as S, rule_rebound as RR, sources, store
from .sources import FetchError

TPE = ZoneInfo("Asia/Taipei")


def now() -> datetime:
    fake = os.getenv("BT_FAKE_NOW")
    return datetime.fromisoformat(fake).replace(tzinfo=TPE) if fake else datetime.now(TPE)


def mdir():
    d = config.DATA_DIR / "market"
    (d / "days").mkdir(parents=True, exist_ok=True)
    return d


def _csv(name: str) -> pd.DataFrame:
    p = mdir() / name
    return pd.read_csv(p, dtype={"date": str}) if p.exists() else pd.DataFrame()


def _save_csv(df: pd.DataFrame, name: str, key: list[str]) -> None:
    df = df.drop_duplicates(key, keep="last").sort_values(key)
    df.to_csv(mdir() / name, index=False)


def day_path(d: str):
    return mdir() / "days" / f"{d}.json"


def load_day(d: str) -> dict:
    p = day_path(d)
    return json.loads(p.read_text("utf-8")) if p.exists() else {"date": d}


def save_day(rec: dict) -> None:
    day_path(rec["date"]).write_text(json.dumps(_clean(rec), ensure_ascii=False, indent=1), "utf-8")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def warn(msg: str) -> None:
    print(f"::warning title=盤勢::{msg}")


# ---------------------------------------------------------------- 歷史資料
def update_intraday(days: list[date], budget_min: float = 1e9) -> int:
    """回補加權指數每 5 秒資料（只存每天的關鍵點）。"""
    have = set(_csv("intraday.csv").get("date", pd.Series(dtype=str)))
    end = time.time() + budget_min * 60
    rows, n = [], 0
    for d in days:
        ds = d.isoformat()
        if ds in have or time.time() > end:
            continue
        try:
            df = S.fetch_5min_index(d)
        except FetchError as e:
            print("略過", ds, e)
            continue
        if df is None:
            continue
        rows.append({"date": ds, **S.summarize_5min(df)})
        n += 1
        if len(rows) >= 50:
            _append_intraday(rows)
            rows = []
    if rows:
        _append_intraday(rows)
    return n


def _append_intraday(rows):
    old = _csv("intraday.csv")
    _save_csv(pd.concat([old, pd.DataFrame(rows)], ignore_index=True), "intraday.csv", ["date"])


def update_taifex(start: date, end: date) -> None:
    """台指期：FinMind 為主（一年一次請求）；失敗或沒資料才逐月改抓期交所。都失敗只警告，不中斷。"""
    def save(new):
        if len(new):
            _save_csv(pd.concat([_csv("taifex.csv"), new], ignore_index=True), "taifex.csv", ["date", "session"])
    try:
        new = S.fetch_finmind_futures(start, end, os.getenv("FINMIND_TOKEN"))
        if len(new):
            save(new)
            print(f"台指期（FinMind）：{len(new)} 列")
            return
        print("FinMind 台指期沒有資料，改抓期交所")
    except FetchError as ex:
        print("FinMind 台指期失敗，改抓期交所：", ex)
    n = 0
    for s, e in S.month_ranges(start, end):
        try:
            new = S.fetch_taifex_month(s, e)
        except FetchError as ex:
            print("期交所失敗", s, ex)
            continue
        save(new)
        n += len(new)
    if n == 0:
        warn("台指期資料抓不到（FinMind 與期交所都沒有資料）")


def update_us(range_: str) -> None:
    us = S.fetch_us(range_)
    us.index.name = "date"
    old = _csv("us.csv")
    df = pd.concat([old, us.reset_index()], ignore_index=True) if len(old) else us.reset_index()
    _save_csv(df, "us.csv", ["date"])


def trading_days_from_store() -> list[date]:
    out = []
    for f in sorted(store._p("price").glob("*/*.csv.gz")):
        out.append(datetime.strptime(f.name[:8], "%Y%m%d").date())
    return out


def catch_up(today: date) -> None:
    """補最近幾天遺漏的每 5 秒指數（例如收盤任務時證交所還沒公布）。"""
    recent = [d for d in trading_days_from_store() if today - timedelta(days=14) <= d < today]
    update_intraday(recent)


def _load_all():
    intr = _csv("intraday.csv")
    tx = _csv("taifex.csv")
    us = _csv("us.csv")
    if len(us):
        us = us.set_index("date")
    return intr, tx, us


def _meta() -> dict:
    p = mdir() / "meta.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def _save_meta(m: dict) -> None:
    (mdir() / "meta.json").write_text(json.dumps(_clean(m), ensure_ascii=False, indent=1), "utf-8")


# ---------------------------------------------------------------- 模型（每天重算回測，約數秒）
def preopen_model():
    intr, tx, us = _load_all()
    if len(intr) < M.MIN_TRAIN or len(tx) == 0 or len(us) == 0:
        raise RuntimeError("開盤前模型的歷史資料不足，請先執行 backfill")
    lag = M.night_alignment(tx, intr)
    ds = M.preopen_dataset(intr, tx, us, lag)
    probs = M.walk_forward(ds, M.PRE_FEATS, M.PRE_RULES)
    stats = M.evaluate(ds, probs)
    return ds, lag, stats, M.choose(stats)


def intraday_model():
    intr, _, _ = _load_all()
    if len(intr) < M.MIN_TRAIN:
        raise RuntimeError("盤中模型的歷史資料不足，請先執行 backfill")
    ds = M.intraday_dataset(intr)
    probs = M.walk_forward(ds, M.INTRA_FEATS, M.INTRA_RULES)
    stats = M.evaluate(ds, probs)
    return ds, stats, M.choose(stats), M.close_forecast_backtest(ds)


def _prob_for(method: str, ds: pd.DataFrame, feats: list[str], x: dict, rules: dict) -> dict:
    """回傳 pred_up（預測方向）與 prob_up（收漲機率）。

    規則型方法本身只給漲或跌，機率改用「歷史上出現同樣訊號的日子，實際收漲的比例」，
    避免頁面顯示 0% / 100% 這種誤導的數字。"""
    if method == "logit":
        tr = ds.dropna(subset=feats + ["y"])
        m = M._logit().fit(tr[feats].values * 100, tr["y"].values)
        p = float(m.predict_proba(np.array([[x[f] for f in feats]]) * 100)[0, 1])
        return {"pred_up": p > 0.5, "prob_up": p, "prob_kind": "model", "prob_n": int(len(tr))}
    today_sig = float(rules[method](pd.DataFrame([x])).iloc[0])
    hist = ds.dropna(subset=["y"])
    same = hist[rules[method](hist) == today_sig]
    return {"pred_up": today_sig > 0.5, "prob_up": float(same["y"].mean()) if len(same) else None,
            "prob_kind": "history", "prob_n": int(len(same))}


# ---------------------------------------------------------------- 任務
def task_preopen(force=False) -> None:
    t = now()
    today = t.date()
    rec = load_day(today.isoformat())
    if rec.get("preopen") and not force:
        print("今天的開盤前預測已完成，略過")
        return
    if today.weekday() >= 5:
        print("週末不開盤")
        return
    update_taifex(today - timedelta(days=12), today)
    try:
        update_us("1mo")
    except FetchError as e:
        warn(f"美股資料抓不到：{e}")
    catch_up(today)
    try:
        ds, lag, stats, method = preopen_model()
    except RuntimeError as e:
        warn(str(e))
        rec["preopen"] = {"status": "no_model", "made_at": t.strftime("%H:%M"), "msg": str(e)}
        save_day(rec)
        return
    intr, tx, us = _load_all()
    # 今天的特徵
    days = sorted(intr["date"])
    prev = days[-1] if days and days[-1] < today.isoformat() else None
    day = tx[tx["session"] == "day"].set_index("date")["close"]
    night = tx[tx["session"] == "night"].set_index("date")["close"]
    key = today.isoformat() if lag == 0 else prev
    if prev is None or prev not in day.index or key not in night.index:
        warn(f"找不到 {today} 的期貨夜盤資料，今天可能休市或資料尚未公布")
        rec["preopen"] = {"status": "no_data", "made_at": t.strftime("%H:%M")}
        save_day(rec)
        return
    u = M.us_returns_for(us, [today.isoformat()])
    x = {"night_ret": night[key] / day[prev] - 1,
         **{c: float(u[c].iloc[0]) if c in u and len(u) else np.nan for c in ("spx", "sox", "tsm")}}
    if method == "logit" and not all(np.isfinite(x[f]) for f in M.PRE_FEATS):
        method = "night"
    pr = _prob_for(method, ds, M.PRE_FEATS, x, M.PRE_RULES)
    rec["preopen"] = {"status": "ok", "made_at": t.strftime("%H:%M"), "method": method,
                      "method_name": M.NAMES[method], **pr,
                      "inputs": x, "stats": stats, "night_alignment": lag}
    save_day(rec)
    print(json.dumps(_clean(rec["preopen"]), ensure_ascii=False))


def task_intraday(force=False) -> None:
    t = now()
    today = t.date()
    rec = load_day(today.isoformat())
    if rec.get("intraday") and not force:
        print("今天的盤中預測已完成，略過")
        return
    if today.weekday() >= 5:
        print("週末不開盤")
        return
    q = S.fetch_mis(["t00"]).get("t00")
    if not q or q["date"] != today.strftime("%Y%m%d"):
        print("今天沒有盤中資料，可能休市")
        rec["intraday"] = {"status": "closed", "made_at": t.strftime("%H:%M")}
        save_day(rec)
        return
    try:
        ds, stats, method, cf_bt = intraday_model()
    except RuntimeError as e:
        warn(str(e))
        rec["intraday"] = {"status": "no_model", "made_at": t.strftime("%H:%M"), "msg": str(e)}
        save_day(rec)
        return
    x = {"r_at": q["price"] / q["prev"] - 1, "gap": q["open"] / q["prev"] - 1,
         "mom": q["price"] / q["open"] - 1, "rng": (q["high"] - q["low"]) / q["prev"]}
    pr = _prob_for(method, ds, M.INTRA_FEATS, x, M.INTRA_RULES)
    cf = M.close_forecast(ds, x)
    rec["intraday"] = {"status": "ok", "made_at": t.strftime("%H:%M"), "quote_time": q["time"],
                       "index": q["price"], "prev": q["prev"], "open": q["open"],
                       "method": method, "method_name": M.NAMES[method], **pr,
                       "close_mid": q["prev"] * (1 + cf["ret"]), "close_lo": q["prev"] * (1 + cf["lo"]),
                       "close_hi": q["prev"] * (1 + cf["hi"]), "inputs": x, "stats": stats,
                       "close_backtest": cf_bt}
    save_day(rec)
    print(json.dumps(_clean(rec["intraday"]), ensure_ascii=False))


def _sector_table(js: dict) -> list[dict]:
    out = []
    for fields, data in sources._tables(js):
        if "指數" in fields and any("漲跌百分比" in f for f in fields):
            i_name = fields.index("指數")
            i_pct = next(i for i, f in enumerate(fields) if "漲跌百分比" in f)
            i_sign = next((i for i, f in enumerate(fields) if "漲跌(+/-)" in f), None)
            for r in data:
                n = str(r[i_name])
                if "類指數" not in n or "報酬" in n:
                    continue
                pct = sources.to_num(r[i_pct])
                if i_sign is not None and "-" in str(r[i_sign]) and pct > 0:
                    pct = -pct
                out.append({"name": n.replace("類指數", ""), "pct": pct / 100})
            if out:
                break
    return out


def _prev_close_map(today: date) -> pd.Series:
    files = [f for f in sorted(store._p("price").glob("*/*.csv.gz")) if f.name[:8] < today.strftime("%Y%m%d")]
    if not files:
        return pd.Series(dtype=float)
    df = pd.read_csv(files[-1], dtype={"code": str})
    return df.set_index("code")["close"]


def _holdings(today: date, close_map: dict[str, float]) -> list[dict]:
    log = live_log.load()
    if not log:
        return []
    month = sorted(log)[-1]
    rec = log[month]
    if not rec.get("entry"):
        return []
    ent = datetime.strptime(rec["entry"], "%Y-%m-%d").date()
    ef = store.daily_path("price", ent)
    entry_px = pd.read_csv(ef, dtype={"code": str}).set_index("code")["close"] if ef.exists() else pd.Series(dtype=float)
    prev = _prev_close_map(today)
    out = []
    for s, codes in rec["picks"].items():
        for c in codes:
            px = close_map.get(c)
            out.append({"strategy": s, "code": c, "price": px,
                        "day": (px / prev[c] - 1) if px and c in prev and prev[c] else None,
                        "since": (px / entry_px[c] - 1 - config.ROUND_TRIP_COST) if px and c in entry_px else None})
    return out


def task_close(force=False, wait_min: float = 25) -> None:
    t = now()
    today = t.date()
    rec = load_day(today.isoformat())
    if rec.get("close", {}).get("status") == "ok" and not force:
        print("今天的收盤整理已完成，略過")
        return
    if today.weekday() >= 5:
        print("週末不開盤")
        return
    q = S.fetch_mis(["t00"]).get("t00")
    if not q or q["date"] != today.strftime("%Y%m%d"):
        print("今天沒有交易，可能休市")
        rec["close"] = {"status": "closed"}
        save_day(rec)
        return
    # 全市場收盤行情：證交所約 13:45~14:30 公布，等到有資料為止
    js, px = None, None
    deadline = time.time() + wait_min * 60
    while time.time() < deadline:
        try:
            js = sources.get_json(sources.MI_INDEX_URL, {"date": today.strftime("%Y%m%d"),
                                                         "type": "ALLBUT0999", "response": "json"})
            px = sources.parse_mi_index(js)
        except FetchError as e:
            print("MI_INDEX", e)
        if px is not None and len(px):
            break
        time.sleep(90)
    c = {"status": "ok", "made_at": now().strftime("%H:%M"), "index": q["price"], "prev": q["prev"],
         "chg": q["price"] / q["prev"] - 1, "high": q["high"], "low": q["low"]}
    close_map = {}
    if px is not None and len(px):
        stocks = px[px["code"].str.fullmatch(r"[1-9]\d{3}")].copy()
        prev = _prev_close_map(today)
        stocks["chg"] = stocks["close"] / stocks["code"].map(prev) - 1
        st = stocks.dropna(subset=["chg"])
        c["breadth"] = {"up": int((st["chg"] > 0).sum()), "down": int((st["chg"] < 0).sum()),
                        "flat": int((st["chg"] == 0).sum()),
                        "limit_up": int((st["chg"] >= 0.095).sum()), "limit_down": int((st["chg"] <= -0.095).sum())}
        c["value"] = float(px["value"].sum())
        liquid = st[st["value"] >= 1e8]
        cols = ["code", "name", "close", "chg"]
        c["top_gainers"] = liquid.nlargest(5, "chg")[cols].to_dict("records")
        c["top_losers"] = liquid.nsmallest(5, "chg")[cols].to_dict("records")
        c["top_value"] = st.nlargest(5, "value")[cols + ["value"]].to_dict("records")
        c["sectors"] = sorted(_sector_table(js), key=lambda r: -r["pct"])
        close_map = dict(zip(px["code"], px["close"]))
    else:
        warn("證交所全市場收盤行情尚未公布，只整理大盤指數")
    c["holdings"] = _holdings(today, close_map)
    # 和今天的預測對答案
    up = q["price"] > q["prev"]
    for k in ("preopen", "intraday"):
        pr = rec.get(k) or {}
        if pr.get("status") == "ok":
            pr["actual_up"] = bool(up)
            pr["hit"] = bool(pr["pred_up"] == up)
            if k == "intraday":
                pr["close_actual"] = q["price"]
                pr["in_range"] = bool(pr["close_lo"] <= q["price"] <= pr["close_hi"])
    rec["close"] = c
    try:
        q2 = S.fetch_mis(RR.ASSETS)
        live = {"t00": q["price"], **{a: close_map.get(a) or (q2.get(a) or {}).get("price") for a in RR.ASSETS}}
        rec["rule"] = _rule_record(RR.replay(*rule_inputs(live)), now().strftime("%H:%M"), final=True)
    except Exception as e:  # noqa: BLE001  規則失敗不影響收盤整理
        warn(f"超跌反彈規則收盤確認失敗：{e}")
    save_day(rec)
    # 今天的每 5 秒指數（明天的模型會用到）；還沒公布就交給下一次 catch_up
    try:
        update_intraday([today])
    except FetchError as e:
        print("MI_5MINS_INDEX", e)
    print(json.dumps(_clean(c), ensure_ascii=False)[:2000])


def rule_inputs(live: dict | None = None):
    """加權指數收盤（每 5 秒指數的收盤）與規則標的收盤；live＝今天盤中暫定價（{'t00': 價, '0050': 價, ...}）。"""
    intr = _csv("intraday.csv")
    idx = pd.Series(intr["close"].values, index=pd.to_datetime(intr["date"])).sort_index()
    px = store.load_daily("price")
    px = px[px["code"].isin(RR.ASSETS)]
    prices = px.pivot_table(index="date", columns="code", values="close")
    prices.index = pd.to_datetime(prices.index)
    if live:
        today = pd.Timestamp(now().date())
        idx.loc[today] = live["t00"]
        prices = prices.reindex(prices.index.union([today]))
        for a in RR.ASSETS:
            if live.get(a) and np.isfinite(live[a]):
                prices.loc[today, a] = live[a]
        idx = idx.sort_index()
    prices = prices.reindex(columns=RR.ASSETS).reindex(idx.index).ffill()
    return idx, prices


def _rule_record(res: dict, made_at: str, final: bool) -> dict:
    closed = sorted((t for t in res["trades"] if "ret" in t), key=lambda t: (t["entry_date"], t["asset"]))
    return {"made_at": made_at, "final": final, **{k: res[k] for k in ("date", "rsi", "lower", "index", "cond", "new", "prev_cond")},
            "open": res["open"], "recent": closed[-12:],
            "stats": RR.stats(res["trades"], config.FEE_RATE * 2 + 0.001, config.ROUND_TRIP_COST)}


def task_midday(force=False) -> None:
    """12:40：用即時價格暫時判斷超跌反彈規則（買進訊號與出場條件）。"""
    t = now()
    today = t.date()
    rec = load_day(today.isoformat())
    if rec.get("midday") and not force:
        print("今天的 12:40 規則檢查已完成，略過")
        return
    if today.weekday() >= 5:
        print("週末不開盤")
        return
    q = S.fetch_mis(["t00"] + RR.ASSETS)
    if not q.get("t00") or q["t00"]["date"] != today.strftime("%Y%m%d"):
        print("今天沒有盤中資料，可能休市")
        rec["midday"] = {"status": "closed", "made_at": t.strftime("%H:%M")}
        save_day(rec)
        return
    live = {"t00": q["t00"]["price"], **{a: (q.get(a) or {}).get("price") for a in RR.ASSETS}}
    res = RR.replay(*rule_inputs(live))
    rec["midday"] = {"status": "ok", "quote_time": q["t00"]["time"],
                     **_rule_record(res, t.strftime("%H:%M"), final=False)}
    save_day(rec)
    print(json.dumps(_clean({k: rec["midday"][k] for k in ("rsi", "lower", "index", "cond", "new")}), ensure_ascii=False))
    if res["new"]:
        warn(f"超跌反彈規則：出現新的買進訊號（RSI {res['rsi']:.1f}，指數 {res['index']:.0f} ≤ 下軌 {res['lower']:.0f}）")


def task_flows(force=False) -> None:
    today = now().date()
    rec = load_day(today.isoformat())
    if rec.get("flows") and not force:
        print("今天的三大法人整理已完成，略過")
        return
    if not (rec.get("close") or {}).get("status") == "ok":
        print("今天沒有收盤整理（可能休市），略過")
        return
    f = S.fetch_bfi82u(today)
    if not f:
        warn("三大法人買賣金額尚未公布")
        return
    out = {"amount": f}
    t86p = store.daily_path("t86", today)
    pxp = store.daily_path("price", today)
    if t86p.exists() and pxp.exists():
        t86 = pd.read_csv(t86p, dtype={"code": str})
        px = pd.read_csv(pxp, dtype={"code": str}).set_index("code")
        t86 = t86[t86["code"].str.fullmatch(r"[1-9]\d{3}")].copy()
        t86["name"] = t86["code"].map(px["name"])
        for col, key in (("foreign_net", "foreign"), ("trust_net", "trust")):
            t86[key + "_amt"] = t86[col] * t86["code"].map(px["close"])
            out[key + "_buy"] = t86.nlargest(5, key + "_amt")[["code", "name", key + "_amt"]].to_dict("records")
            out[key + "_sell"] = t86.nsmallest(5, key + "_amt")[["code", "name", key + "_amt"]].to_dict("records")
    rec["flows"] = out
    save_day(rec)
    print(json.dumps(_clean(out), ensure_ascii=False)[:1500])


def task_probe() -> None:
    """每個資料來源打一次，印出狀態與回應開頭，用來確認網址與格式。"""
    import requests
    d = trading_days_from_store()[-1]
    checks = [("5 秒指數", lambda: S.fetch_5min_index(d)),
              ("MIS 即時", lambda: S.fetch_mis(["t00", "2330"])),
              ("法人金額", lambda: S.fetch_bfi82u(d)),
              ("期交所", lambda: S.fetch_taifex_month(d - timedelta(days=5), d)),
              ("FinMind 期貨", lambda: S.fetch_finmind_futures(d - timedelta(days=5), d, os.getenv("FINMIND_TOKEN"))),
              ("美股", lambda: S.fetch_us("5d"))]
    for name, fn in checks:
        try:
            r = fn()
            desc = f"{len(r)} 筆" if hasattr(r, "__len__") else str(r)
            print(f"✅ {name}：{desc}")
            print("   ", (r.head(3).to_string() if hasattr(r, "head") else str(r)[:300]).replace("\n", "\n    "))
        except Exception as e:  # noqa: BLE001
            print(f"❌ {name}：{type(e).__name__} {e}")
    # 原始回應開頭（找網址問題用）
    for u in S.FIVE_MIN_URLS:
        try:
            r = requests.get(u, params={"date": d.strftime("%Y%m%d"), "response": "json"},
                             headers=sources.HEADERS, timeout=30)
            print(f"   {u} → HTTP {r.status_code} {r.headers.get('content-type')} {r.text[:150]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"   {u} → {e}")
        time.sleep(2)


def task_backfill(minutes: float) -> None:
    t0 = time.time()
    today = now().date()
    try:
        update_us("15y")
    except FetchError as e:
        warn(f"美股資料抓不到：{e}")
    update_taifex(date(2017, 5, 1), today)
    days = [d for d in trading_days_from_store() if d >= date(2016, 1, 1) and d < today]
    left = minutes - (time.time() - t0) / 60
    n = update_intraday(days, budget_min=max(1, left - 5))
    intr = _csv("intraday.csv")
    print(f"每 5 秒指數：本次新增 {n} 天，共 {len(intr)} 天；期貨 {len(_csv('taifex.csv'))} 列；美股 {len(_csv('us.csv'))} 天")
    problems = [k for k, v in (("每 5 秒指數", len(intr)), ("期貨", len(_csv("taifex.csv"))), ("美股", len(_csv("us.csv")))) if v == 0]
    if problems:
        print(f"::error title=回補失敗::{'、'.join(problems)} 一筆都沒抓到")
        _save_meta(dict(_meta(), backfill_at=now().strftime("%Y-%m-%d %H:%M"), backfill_failed=problems))
        raise SystemExit(1)
    # 回補完成後跑一次回測，存下各方法命中率
    meta = _meta()
    try:
        ds, lag, st, m = preopen_model()
        meta["preopen"] = {"stats": st, "method": m, "night_alignment": lag, "n": int(len(ds))}
    except RuntimeError as e:
        print(e)
    try:
        ds, st, m, cf = intraday_model()
        meta["intraday"] = {"stats": st, "method": m, "close_backtest": cf, "n": int(len(ds))}
    except RuntimeError as e:
        print(e)
    meta["backfill_at"] = now().strftime("%Y-%m-%d %H:%M")
    _save_meta(meta)
    print(json.dumps(_clean(meta), ensure_ascii=False, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["preopen", "intraday", "midday", "close", "flows", "backfill", "render", "probe"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--minutes", type=float, default=300)
    a = ap.parse_args()
    from . import today_page
    try:
        if a.task == "backfill":
            task_backfill(a.minutes)
        elif a.task == "probe":
            task_probe()
        elif a.task != "render":
            {"preopen": task_preopen, "intraday": task_intraday, "midday": task_midday, "close": task_close,
             "flows": task_flows}[a.task](force=a.force)
    finally:
        if a.task not in ("backfill", "probe"):
            today_page.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
