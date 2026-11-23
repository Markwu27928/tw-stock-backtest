"""每日流程：抓資料 → 回測 → 產生報告頁。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

from . import backtest, config, live_log, quality, report, robustness, update_data
from .features import build_panel


def report_status(info: dict) -> None:
    """在 GitHub Actions 頁面顯示抓資料結果：有問題時出現黃色警告，不必讀 log。"""
    done, total = info.get("days_done", 0), info.get("days_total", 0)
    new = info.get("new_price_days", 0)
    lines = [
        "## 抓資料結果",
        f"- 本次新增行情：**{new} 天**、法人：{info.get('new_t86_days', 0)} 天",
        f"- 回補進度：**{done} / {total}** 天，連續完整到 {info.get('contiguous_until')}",
        f"- 提早停止：{'是' if info.get('stopped_early') else '否'}",
    ]
    for w in info.get("warnings", []):
        lines.append(f"- ⚠️ {w}")
        print(f"::warning title=抓資料::{w}")
    if total and done < total and new == 0:
        msg = f"歷史資料尚未補完（{done}/{total}），但本次沒有新增任何一天"
        lines.append(f"- ⚠️ {msg}")
        print(f"::warning title=回補沒有進度::{msg}")
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=45, help="抓資料時間上限（分鐘）")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--no-liquidity", action="store_true", help="略過流動性門檻的穩健度檢查（要重跑兩次回測）")
    a = ap.parse_args()

    if a.skip_fetch:
        p = config.DATA_DIR / "update_log.json"
        info = json.loads(p.read_text("utf-8")) if p.exists() else {}
    else:
        info = update_data.run(a.max_minutes, os.getenv("FINMIND_TOKEN"))
        print(json.dumps(info, ensure_ascii=False, indent=1))
        report_status(info)

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
            bt["months"] = [m for m in bt["months"] if m["exit"] <= until]
            bt["live"] = None   # 中間有缺口時，動能等特徵不可靠，先不顯示本月持股
        data = report.build(panel, bt, info)

        # 穩健度檢查（成本、檔數、流動性門檻）
        rb = robustness.run(panel, bt, with_liquidity=not a.no_liquidity)
        data["robustness"] = report._clean(rb)

        # 實盤追蹤：凍結本月名單，計算歷次凍結名單的報酬
        log = live_log.load()
        if live_log.freeze(log, bt["live"]):
            live_log.save(log)
        data["live_track"] = report._clean(live_log.evaluate(log, panel, bt["schedule"]))

        # 資料品質
        findings = quality.check(panel, info)
        data["quality"] = findings
        for q in findings:
            if q["level"] in ("error", "warning"):
                print(f"::warning title=資料品質::{q['msg']}")

        report.write(data)
        print(f"回測完成，用時 {time.time() - t0:.0f} 秒")
        for s, sm in data["summary_common"].items():
            if sm:
                print(s, f"選股勝率 {sm['pick_hit']:.3f} 基準 {sm['baseline_hit']:.3f} 月勝率 {sm['month_win']:.3f}",
                      f"年化 {sm['cagr']:.3f} 等權重 {sm['ew_cagr']:.3f} 0050 {sm['bench_cagr']:.3f} 穩健度 {rb['verdict'][s]}")
    except RuntimeError as e:
        print("略過回測：", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
