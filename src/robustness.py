"""穩健度檢查：好的策略應該在一整片參數範圍內都站得住，而不是只在預設值才有效。

- 成本壓力：來回成本 ×1、×1.5、×2（我們以收盤價成交、未計滑價，所以要測悲觀情況）
- 選股檔數：5、10、20 檔
- 流動性門檻：20 日均成交金額 ×0.5、×1、×2（會改變母體，需要重跑整個回測）

每個變化都和兩個基準比較：
- 隨機選股基準（同月母體中跑贏 0050 的比例）
- 等權重基準（同月持有全部合格股票）
規則在看到結果前就固定，不依結果挑選參數。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .backtest import STRATEGIES, run_backtest

COST_MULTS = (1.0, 1.5, 2.0)
TOP_NS = (5, 10, 20)
LIQ_MULTS = (0.5, 2.0)          # ×1 就是主回測，不必重跑


def _common_months(bt: dict) -> list[dict]:
    """三個策略都有分數的月份（與主報告的共同期間一致）。"""
    return [m for m in bt["months"] if all(s in m["scores"] for s in STRATEGIES)]


def _variant(months: list[dict], strategy: str, top_n: int, cost: float) -> dict:
    hits = picks = 0
    port, ew, uni_hit, bench = [], [], [], []
    for m in months:
        net = m["fwd"] - cost
        b = m["bench"]
        top = m["scores"][strategy].dropna().sort_values(ascending=False).head(top_n)
        r = net.reindex(top.index)
        hits += int((r > b).sum())
        picks += len(r)
        port.append(float(r.mean()))
        ew.append(float(net.mean()))
        uni_hit.append(float((net > b).mean()))
        bench.append(b)
    if not port:
        return {}
    years = len(port) / 12
    port, ew = np.array(port), np.array(ew)
    cagr = float(np.prod(1 + port) ** (1 / years) - 1)
    ew_cagr = float(np.prod(1 + ew) ** (1 / years) - 1)
    ex = port - ew
    t_ew = float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))) if len(ex) > 2 and ex.std() > 0 else np.nan
    return {
        "months": len(port),
        "pick_hit": hits / picks if picks else np.nan,
        "baseline_hit": float(np.mean(uni_hit)),
        "cagr": cagr,
        "ew_cagr": ew_cagr,
        "bench_cagr": float(np.prod(1 + np.array(bench)) ** (1 / years) - 1),
        "t_ew": t_ew,
        "beats_random": hits / picks > float(np.mean(uni_hit)) if picks else False,
        "beats_ew": cagr > ew_cagr,
        # 0050 是買進持有、不扣成本，成本加倍時最容易輸給它
        "beats_bench": cagr > float(np.prod(1 + np.array(bench)) ** (1 / years) - 1),
    }


def stress(bt: dict) -> dict:
    """成本與檔數的變化，不需要重跑回測。"""
    months = _common_months(bt)
    out = {}
    for s in STRATEGIES:
        rows = []
        for k in COST_MULTS:
            rows.append({"name": f"成本 ×{k:g}", "group": "cost",
                         **_variant(months, s, config.TOP_N, config.ROUND_TRIP_COST * k)})
        for n in TOP_NS:
            if n == config.TOP_N:
                continue
            rows.append({"name": f"選 {n} 檔", "group": "top_n",
                         **_variant(months, s, n, config.ROUND_TRIP_COST)})
        out[s] = rows
    return out


def liquidity(panel, mults=LIQ_MULTS) -> dict:
    """流動性門檻變化：改變母體，需要重跑整個回測（含 LightGBM）。"""
    out = {s: [] for s in STRATEGIES}
    base = config.MIN_VALUE_20
    try:
        for k in mults:
            config.MIN_VALUE_20 = base * k
            bt = run_backtest(panel)
            months = _common_months(bt)
            for s in STRATEGIES:
                out[s].append({"name": f"流動性門檻 ×{k:g}", "group": "liquidity",
                               **_variant(months, s, config.TOP_N, config.ROUND_TRIP_COST)})
    finally:
        config.MIN_VALUE_20 = base
    return out


def passes(r: dict) -> bool:
    """同時贏隨機選股、等權重基準與 0050。"""
    return bool(r.get("beats_random") and r.get("beats_ew") and r.get("beats_bench"))


def verdict(rows: list[dict]) -> str:
    """robust：所有變化都通過；fragile：過半通過；none：多數不通過。"""
    rows = [r for r in rows if r.get("months")]
    if not rows:
        return "none"
    ok = sum(1 for r in rows if passes(r))
    if ok == len(rows):
        return "robust"
    return "fragile" if ok >= len(rows) / 2 else "none"


def failed(rows: list[dict]) -> list[str]:
    return [r["name"] for r in rows if r.get("months") and not passes(r)]


def run(panel, bt: dict, with_liquidity: bool = True) -> dict:
    res = stress(bt)
    if with_liquidity:
        liq = liquidity(panel)
        for s in res:
            res[s] += liq[s]
    return {"variants": res, "verdict": {s: verdict(rows) for s, rows in res.items()},
            "failed": {s: failed(rows) for s, rows in res.items()}}
