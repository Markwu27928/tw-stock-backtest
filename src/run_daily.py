"""每日流程：抓資料 → 回測 → 產生報告頁。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

from . import backtest, config, report, update_data
from .features import build_panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=45, help="抓資料時間上限（分鐘）")
    ap.add_argument("--skip-fetch", action="store_true")
    a = ap.parse_args()

    if a.skip_fetch:
        p = config.DATA_DIR / "update_log.json"
        info = json.loads(p.read_text("utf-8")) if p.exists() else {}
    else:
        info = update_data.run(a.max_minutes, os.getenv("FINMIND_TOKEN"))
        print(json.dumps(info, ensure_ascii=False, indent=1))

    try:
        until = pd.Timestamp(info["contiguous_until"]) if info.get("contiguous_until") else None
        # 回測只用連續完整的區間；本月持股另外用最新資料
        t0 = time.time()
        panel = build_panel()
        bt = backtest.run_backtest(panel)
        if until is not None and until < panel.dates[-1]:
            # 歷史還沒補完：只保留連續完整區間內已結束的月份
            for s in bt["results"]:
                bt["results"][s] = [r for r in bt["results"][s] if r["exit"] <= until]
            bt["live"] = None   # 中間有缺口時，動能等特徵不可靠，先不顯示本月持股
        data = report.build(panel, bt, info)
        report.write(data)
        print(f"回測完成，用時 {time.time() - t0:.0f} 秒")
        for s, sm in data["summary_common"].items():
            if sm:
                print(s, f"選股勝率 {sm['pick_hit']:.3f} 基準 {sm['baseline_hit']:.3f} 月勝率 {sm['month_win']:.3f}")
    except RuntimeError as e:
        print("略過回測：", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
