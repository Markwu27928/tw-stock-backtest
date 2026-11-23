"""月頻 walk-forward 回測。

時序（每個月）：
  d     = 當月第一個 >= 11 日的交易日，收盤後計算特徵、排名
  entry = d 的下一個交易日收盤買進
  exit  = 下個月的 entry 收盤賣出（約 20 個交易日）
LightGBM 只用「出場日 <= d」的歷史月份訓練（上一個月的標籤在 d 時還沒實現，自動排除）。
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .features import FEATURES, Panel, bench_return, decision_dates, features_at, forward_returns

warnings.filterwarnings("ignore", category=UserWarning)

STRATEGIES = {
    "A": "月營收年增率單因子",
    "B": "多因子綜合分數",
    "C": "LightGBM 機器學習",
}


def _rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True).fillna(0.5)


def score_a(f: pd.DataFrame) -> pd.Series:
    return f["rev_yoy"]


def score_b(f: pd.DataFrame) -> pd.Series:
    parts = [_rank(f["rev_yoy"]), _rank(f["yoy_accel"]), _rank(f["mom_6_1"]),
             _rank(f["trust_20"]), _rank(-f["ret_21"]), _rank(f["high_252"])]
    return sum(parts) / len(parts)


def _lgbm():
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=2,
    )


def build_schedule(p: Panel) -> list[dict]:
    """含資料起點後所有月份（訓練需要），並標出是否在回測期間。"""
    dds = decision_dates(p)
    dates = p.dates
    sched = []
    for k, d in enumerate(dds):
        i = dates.get_loc(d)
        if i + 1 >= len(dates):
            entry = None
        else:
            entry = dates[i + 1]
        exit_ = None
        if k + 1 < len(dds):
            j = dates.get_loc(dds[k + 1])
            if j + 1 < len(dates):
                exit_ = dates[j + 1]
        sched.append({"month": str(d.to_period("M")), "decision": d, "entry": entry, "exit": exit_,
                      "in_test": d >= pd.Timestamp(config.BACKTEST_START + "-01")})
    return sched


def run_backtest(p: Panel) -> dict:
    sched = build_schedule(p)
    cost = config.ROUND_TRIP_COST
    hist_rows = []          # LightGBM 訓練資料：features + label + exit
    results = {s: [] for s in STRATEGIES}
    months = []             # 壓力測試用：每月毛報酬、基準、各策略分數
    live = None

    for m in sched:
        d = m["decision"]
        f = features_at(p, d)
        if len(f) < config.TOP_N * 3:
            continue
        done = m["entry"] is not None and m["exit"] is not None
        fwd = forward_returns(p, m["entry"], m["exit"], f.index) if done else None
        bench = bench_return(p, m["entry"], m["exit"]) if done else np.nan

        # 先用「過去」訓練、預測本月，再把本月加進歷史
        train = [r for r in hist_rows if r["exit"] <= d]
        scores = {"A": score_a(f), "B": score_b(f)}
        n_train_months = len({r["month"] for r in train})
        if n_train_months >= config.MIN_TRAIN_MONTHS:
            # 空月份的 DataFrame 欄位是 object，concat 後會污染整欄，LightGBM 會報 dtype 錯誤
            tr = pd.concat([r["df"] for r in train if len(r["df"])])
            model = _lgbm()
            model.fit(tr[FEATURES].astype(float), tr["label"].astype(float))
            scores["C"] = pd.Series(model.predict(f[FEATURES].astype(float)), index=f.index)

        if done:
            lab = f[FEATURES].copy()
            lab["label"] = fwd.rank(pct=True)
            hist_rows.append({"month": m["month"], "exit": m["exit"], "df": lab.dropna(subset=["label"])})

        if not m["in_test"]:
            continue

        if not done:   # 本月持股（還在持有中）
            live = {"month": m["month"], "decision": d, "entry": m["entry"], "scores": scores, "features": f}
            continue

        net = fwd - cost
        universe_hit = float((net > bench).mean())
        ew_ret = float(net.mean())      # 全部合格股票等權重持有（同樣扣成本）
        months.append({"month": m["month"], "exit": m["exit"], "fwd": fwd, "bench": bench, "scores": scores})
        for s, sc in scores.items():
            top = sc.dropna().sort_values(ascending=False).head(config.TOP_N)
            r = net.reindex(top.index)
            results[s].append({
                "month": m["month"], "decision": d, "entry": m["entry"], "exit": m["exit"],
                "picks": [{"code": c, "name": str(p.names.get(c, "")), "score": float(top[c]),
                           "ret": float(r[c])} for c in top.index],
                "port_ret": float(r.mean()), "bench_ret": bench,
                "hit_rate": float((r > bench).mean()), "universe_hit": universe_hit,
                "ew_ret": ew_ret, "n_universe": int(len(f)),
            })
    return {"results": results, "live": live, "schedule": sched, "months": months}


# ---------------------------------------------------------------- 績效指標
def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    n_picks = sum(len(r["picks"]) for r in rows)
    hits = sum(sum(pk["ret"] > r["bench_ret"] for pk in r["picks"]) for r in rows)
    nav = (1 + df["port_ret"]).cumprod()
    bnav = (1 + df["bench_ret"]).cumprod()
    years = len(df) / 12
    diff = df["hit_rate"] - df["universe_hit"]
    t = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if len(diff) > 2 and diff.std() > 0 else np.nan
    ew = df["ew_ret"] if "ew_ret" in df else pd.Series(np.nan, index=df.index)
    ex = df["port_ret"] - ew
    t_ew = float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if len(ex) > 2 and ex.std() > 0 else np.nan
    ewnav = (1 + ew).cumprod()
    last12 = rows[-12:]
    l12_picks = sum(len(r["picks"]) for r in last12)
    l12_hits = sum(sum(pk["ret"] > r["bench_ret"] for pk in r["picks"]) for r in last12)
    return {
        "months": len(df),
        "start": rows[0]["month"], "end": rows[-1]["month"],
        "pick_hit": hits / n_picks if n_picks else np.nan,
        "baseline_hit": float(df["universe_hit"].mean()),
        "month_win": float((df["port_ret"] > df["bench_ret"]).mean()),
        "abs_win": float((df["port_ret"] > 0).mean()),
        "cagr": float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
        "bench_cagr": float(bnav.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan,
        "mdd": float((nav / nav.cummax() - 1).min()),
        "t_stat": t,
        # 等權重基準：每月平均持有全部合格股票。策略若贏不了它，優勢可能只是「等權重、偏中小型股」
        "ew_cagr": float(ewnav.iloc[-1] ** (1 / years) - 1) if years > 0 and ew.notna().all() else np.nan,
        "ew_month_win": float((ex > 0).mean()) if ew.notna().all() else np.nan,
        "t_ew": t_ew,
        "last12_hit": l12_hits / l12_picks if l12_picks else np.nan,
    }


def by_year(rows: list[dict]) -> list[dict]:
    out = {}
    for r in rows:
        y = r["month"][:4]
        o = out.setdefault(y, {"year": y, "picks": 0, "hits": 0, "nav": 1.0, "bnav": 1.0})
        o["picks"] += len(r["picks"])
        o["hits"] += sum(pk["ret"] > r["bench_ret"] for pk in r["picks"])
        o["nav"] *= 1 + r["port_ret"]
        o["bnav"] *= 1 + r["bench_ret"]
    return [{"year": o["year"], "pick_hit": o["hits"] / o["picks"] if o["picks"] else np.nan,
             "ret": o["nav"] - 1, "bench": o["bnav"] - 1} for o in out.values()]


def common_period(results: dict) -> dict:
    """三個策略都有結果的月份，公平比較用。"""
    sets = [set(r["month"] for r in rows) for rows in results.values() if rows]
    if not sets:
        return {s: [] for s in results}
    common = set.intersection(*sets) if len(sets) == len(results) else set()
    return {s: [r for r in rows if r["month"] in common] for s, rows in results.items()}
