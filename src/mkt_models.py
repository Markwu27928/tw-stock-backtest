"""大盤漲跌預測：開盤前（08:20）與盤中（09:40）。

原則
- 每個預測都和最簡單的基準比較；模型沒有贏過基準，就直接用基準方法，並如實顯示命中率
- 回測一律逐月滾動：只用「當月以前」的資料訓練，預測當月（不會偷看未來）
- 選方法的依據是回測期間的命中率，選定後每天照用，頁面持續顯示實際命中率
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEST_START = "2019-01-01"
MIN_TRAIN = 250

PRE_FEATS = ["night_ret", "spx", "sox", "tsm"]
INTRA_FEATS = ["r_at", "gap", "mom", "rng"]


# ---------------------------------------------------------------- 資料集
def night_alignment(tx: pd.DataFrame, intr: pd.DataFrame) -> int:
    """期交所夜盤的「交易日期」標的是哪一天？
    0＝標在隔天（夜盤歸屬下一個交易日），1＝標在當天（開盤前一天）。
    用和隔天開盤跳空的相關性決定，兩種都算、取相關性高的。"""
    best, best_c = 0, -9.0
    for lag in (0, 1):
        f = _night_ret(tx, intr, lag)
        j = intr.set_index("date")
        gap = (j["open"] / j["prev"] - 1).reindex(f.index)
        c = pd.concat([f, gap], axis=1).dropna().corr().iloc[0, 1] if len(f) > 30 else -9
        if c > best_c:
            best, best_c = lag, c
    return best


def _night_ret(tx: pd.DataFrame, intr: pd.DataFrame, lag: int) -> pd.Series:
    days = sorted(intr["date"])
    prev_day = dict(zip(days[1:], days[:-1]))
    day = tx[tx["session"] == "day"].set_index("date")["close"]
    night = tx[tx["session"] == "night"].set_index("date")["close"]
    out = {}
    for d in days:
        p = prev_day.get(d)
        if p is None or p not in day.index:
            continue
        key = d if lag == 0 else p
        if key in night.index:
            out[d] = night[key] / day[p] - 1
    return pd.Series(out, dtype=float)


def us_returns_for(us: pd.DataFrame, tw_days: list[str]) -> pd.DataFrame:
    """台股交易日 D 對應「D 之前最近一個美股交易日」的報酬（台灣時間清晨收盤）。"""
    rets = us.sort_index().pct_change()
    idx = rets.index
    rows = {}
    for d in tw_days:
        k = idx.searchsorted(d) - 1          # 美國日期嚴格小於 D
        if k >= 1:
            rows[d] = rets.iloc[k]
    return pd.DataFrame(rows).T


def preopen_dataset(intr: pd.DataFrame, tx: pd.DataFrame, us: pd.DataFrame, lag: int) -> pd.DataFrame:
    df = intr.set_index("date").sort_index()
    out = pd.DataFrame(index=df.index)
    out["y"] = (df["close"] > df["prev"]).astype(int)
    out["ret"] = df["close"] / df["prev"] - 1
    out["night_ret"] = _night_ret(tx, intr, lag)
    u = us_returns_for(us, list(df.index))
    for c in ("spx", "sox", "tsm"):
        out[c] = u[c] if c in u else np.nan
    return out


def intraday_dataset(intr: pd.DataFrame) -> pd.DataFrame:
    df = intr.set_index("date").sort_index()
    out = pd.DataFrame(index=df.index)
    out["y"] = (df["close"] > df["prev"]).astype(int)
    out["ret"] = df["close"] / df["prev"] - 1
    out["r_at"] = df["at"] / df["prev"] - 1
    out["gap"] = df["open"] / df["prev"] - 1
    out["mom"] = df["at"] / df["open"] - 1
    out["rng"] = (df["hi_at"] - df["lo_at"]) / df["prev"]
    return out


# ---------------------------------------------------------------- 方法
def _logit():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=500)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]):
    tr = train.dropna(subset=feats + ["y"])
    if len(tr) < MIN_TRAIN or tr["y"].nunique() < 2:
        return None
    m = _logit().fit(tr[feats].values * 100, tr["y"].values)
    te = test[feats]
    p = pd.Series(np.nan, index=test.index)
    ok = te.notna().all(axis=1)
    if ok.any():
        p[ok] = m.predict_proba(te[ok].values * 100)[:, 1]
    return p


def walk_forward(ds: pd.DataFrame, feats: list[str], rules: dict) -> pd.DataFrame:
    """回傳每天各方法的「上漲機率」（規則型方法為 0/1）。"""
    res = pd.DataFrame(index=ds.index)
    for name, fn in rules.items():
        res[name] = fn(ds).astype(float)
    res["logit"] = np.nan
    months = sorted({d[:7] for d in ds.index if d >= TEST_START})
    for mth in months:
        tr = ds[ds.index < mth + "-01"]
        te = ds[(ds.index >= mth + "-01") & (ds.index < _next_month(mth) + "-01")]
        p = _fit_predict(tr, te, feats)
        if p is not None:
            res.loc[te.index, "logit"] = p
    return res


def _next_month(m: str) -> str:
    return str(pd.Period(m, "M") + 1)


PRE_RULES = {
    "up": lambda ds: pd.Series(1.0, index=ds.index),
    "night": lambda ds: (ds["night_ret"].fillna(0) > 0).astype(float),
}
INTRA_RULES = {
    "up": lambda ds: pd.Series(1.0, index=ds.index),
    "sign_at": lambda ds: (ds["r_at"] > 0).astype(float),
}
NAMES = {"up": "一律猜漲", "night": "照期貨夜盤方向", "sign_at": "照 09:40 當下漲跌",
         "logit": "邏輯迴歸模型"}


def evaluate(ds: pd.DataFrame, probs: pd.DataFrame) -> dict:
    """測試期間（三種方法都有預測的日子）各方法命中率。"""
    t = probs[probs.index >= TEST_START].dropna()
    y = ds["y"].reindex(t.index)
    out = {}
    for m in probs.columns:
        hit = ((t[m] > 0.5).astype(int) == y)
        out[m] = {"acc": float(hit.mean()) if len(hit) else np.nan, "n": int(len(hit)),
                  "acc_250": float(hit.iloc[-250:].mean()) if len(hit) else np.nan}
    return out


def choose(stats: dict) -> str:
    """命中率最高者；平手時取較簡單的方法（規則型優先於模型）。"""
    order = list(stats)                      # 規則在前、logit 在後
    def acc(m):
        a = stats[m]["acc"]
        return -1.0 if a is None or not np.isfinite(a) else round(a, 4)
    return max(order, key=lambda m: (acc(m), -order.index(m)))


# ---------------------------------------------------------------- 收盤點位預測（09:40）
def close_forecast(ds: pd.DataFrame, x_today: dict) -> dict:
    """OLS：收盤報酬 ~ 09:40 報酬 + 開盤跳空。
    區間依當天早盤波動（09:00~09:40 高低差）縮放：殘差除以早盤波動後取 10% / 90% 分位數，
    再乘回今天的早盤波動。波動大的日子區間自動放寬。
    （2019 年起逐月滾動回測，實際涵蓋率約 81%；固定寬度的區間只有 71%。）"""
    tr = ds.dropna(subset=["ret", "r_at", "gap", "rng"])
    X = np.column_stack([np.ones(len(tr)), tr["r_at"], tr["gap"]])
    beta, *_ = np.linalg.lstsq(X, tr["ret"].values, rcond=None)
    resid = tr["ret"].values - X @ beta
    z = resid / np.maximum(tr["rng"].values, RNG_FLOOR)
    k = max(float(x_today["rng"]), RNG_FLOOR)
    x = np.array([1.0, x_today["r_at"], x_today["gap"]])
    mid = float(x @ beta)
    return {"ret": mid, "lo": mid + float(np.quantile(z, 0.1)) * k, "hi": mid + float(np.quantile(z, 0.9)) * k,
            "n": int(len(tr))}


RNG_FLOOR = 1e-4


def close_forecast_backtest(ds: pd.DataFrame) -> dict:
    """逐月滾動檢驗收盤點位預測：實際收盤落在 80% 區間內的比例、平均誤差。"""
    inside, err = [], []
    for mth in sorted({d[:7] for d in ds.index if d >= TEST_START}):
        tr = ds[ds.index < mth + "-01"]
        te = ds[(ds.index >= mth + "-01") & (ds.index < _next_month(mth) + "-01")].dropna(subset=["ret", "r_at", "gap", "rng"])
        if len(tr.dropna(subset=["ret", "r_at", "gap"])) < MIN_TRAIN:
            continue
        for d, r in te.iterrows():
            f = close_forecast(tr, {"r_at": r["r_at"], "gap": r["gap"], "rng": r["rng"]})
            inside.append(f["lo"] <= r["ret"] <= f["hi"])
            err.append(abs(r["ret"] - f["ret"]))
    return {"coverage_80": float(np.mean(inside)) if inside else np.nan,
            "mae": float(np.mean(err)) if err else np.nan, "n": len(err)}
