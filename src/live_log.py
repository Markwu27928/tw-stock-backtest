"""實盤追蹤：每月決策後把三個策略的名單凍結存檔，之後只計算報酬、不再更動名單。

回測的策略規則是看過歷史資料才設計的，屬於「樣本內」；
從開始記錄起的真實月份，才是真正的樣本外驗證。
名單一旦寫入就不會被之後的程式或資料修正改寫。
"""
from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from . import config

TPE = ZoneInfo("Asia/Taipei")


def _path():
    return config.DATA_DIR / "live_log.json"


def load() -> dict:
    p = _path()
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def save(log: dict) -> None:
    _path().write_text(json.dumps(log, ensure_ascii=False, indent=1), "utf-8")


def freeze(log: dict, live: dict | None, today: str | None = None) -> bool:
    """本月名單第一次出現時寫入；已存在就不動。回傳是否新增。"""
    if not live or live["month"] in log:
        return False
    today = today or datetime.now(TPE).strftime("%Y-%m-%d")
    log[live["month"]] = {
        "decision": pd.Timestamp(live["decision"]).strftime("%Y-%m-%d"),
        "entry": pd.Timestamp(live["entry"]).strftime("%Y-%m-%d") if live.get("entry") is not None else None,
        "recorded": today,
        # 決策日之後才記錄的月份要標出來，避免誤以為是事前預測
        "late": today > pd.Timestamp(live["decision"]).strftime("%Y-%m-%d"),
        "picks": {s: [str(c) for c in sc.dropna().sort_values(ascending=False).head(config.TOP_N).index]
                  for s, sc in live["scores"].items()},
    }
    return True


def evaluate(log: dict, panel, schedule: list[dict]) -> list[dict]:
    """用最新行情計算凍結名單的報酬（已結束月份用出場日，持有中用最新收盤）。"""
    exits = {m["month"]: m["exit"] for m in schedule}
    entries = {m["month"]: m["entry"] for m in schedule}
    last = panel.dates[-1]
    out = []
    for month in sorted(log):
        rec = log[month]
        entry = entries.get(month)
        if entry is None and rec.get("entry"):
            entry = pd.Timestamp(rec["entry"])
        if entry is None or entry not in panel.tri.index:
            out.append({"month": month, **rec, "status": "waiting", "strategies": {}})
            continue
        exit_ = exits.get(month)
        done = exit_ is not None and exit_ in panel.tri.index
        end = exit_ if done else last
        b = config.BENCH
        bench = float(panel.tri.at[end, b] / panel.tri.at[entry, b] - 1) if b in panel.tri.columns else None
        strategies = {}
        for s, codes in rec["picks"].items():
            rets = []
            for c in codes:
                if c in panel.tri.columns and pd.notna(panel.tri.at[entry, c]) and pd.notna(panel.tri.at[end, c]):
                    rets.append(float(panel.tri.at[end, c] / panel.tri.at[entry, c] - 1 - config.ROUND_TRIP_COST))
            strategies[s] = {
                "port_ret": sum(rets) / len(rets) if rets else None,
                "hit": sum(r > bench for r in rets) / len(rets) if rets and bench is not None else None,
                "n": len(rets),
            }
        out.append({"month": month, **rec, "status": "closed" if done else "holding",
                    "asof": end.strftime("%Y-%m-%d"), "bench_ret": bench, "strategies": strategies})
    return out
