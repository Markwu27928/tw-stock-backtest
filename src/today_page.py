"""產生 docs/today.html：今日盤勢與預測紀錄。"""
from __future__ import annotations

import html
import json
from datetime import date

from . import config
from .mkt_models import NAMES
from .report import CSS, _pct, _sign_cls

WD = "一二三四五六日"


def _days() -> list[dict]:
    d = config.DATA_DIR / "market" / "days"
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(f.read_text("utf-8")))
        except ValueError:
            continue
    return out


def _meta() -> dict:
    p = config.DATA_DIR / "market" / "meta.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def _n(x, d=2):
    return "—" if x is None else f"{x:,.{d}f}"


def _arrow(up):
    if up is None:
        return "—"
    return "<span class='up'>▲ 漲</span>" if up else "<span class='down'>▼ 跌</span>"


def live_record(days: list[dict], key: str) -> dict:
    hits = [d[key]["hit"] for d in days if (d.get(key) or {}).get("hit") is not None]
    last20 = hits[-20:]
    return {"n": len(hits), "acc": sum(hits) / len(hits) if hits else None,
            "n20": len(last20), "acc20": sum(last20) / len(last20) if last20 else None}


def _stats_table(stats: dict, chosen: str | None) -> str:
    if not stats:
        return "<p class='note'>回測資料尚未建立。</p>"
    rows = "".join(
        f"<tr><td>{NAMES.get(m, m)}{' ✓' if m == chosen else ''}</td><td>{_pct(v.get('acc'))}</td>"
        f"<td>{_pct(v.get('acc_250'))}</td><td>{v.get('n', 0)}</td></tr>" for m, v in stats.items())
    return ("<div class='scroll'><table><tr><th>方法</th><th>回測命中率</th><th>近 250 天</th><th>天數</th></tr>"
            f"{rows}</table></div><p class='note'>✓＝目前採用（回測命中率最高；平手時用較簡單的方法）。"
            "回測從 2019 年起逐月滾動，只用過去資料訓練。</p>")


def _preopen_html(p: dict, meta: dict) -> str:
    if not p:
        return "<p class='note'>今天的開盤前預測尚未產生（每個交易日約 08:20）。</p>"
    if p.get("status") == "no_model":
        return "<p class='note'>預測模型的歷史資料還在建立中，今天沒有開盤前預測。</p>"
    if p.get("status") != "ok":
        return "<p class='note'>今天沒有開盤前預測（可能休市，或期貨夜盤資料尚未公布）。</p>"
    x = p["inputs"]
    res = ""
    if p.get("hit") is not None:
        res = f"<p class='sub'>收盤結果：{_arrow(p['actual_up'])}，預測{'<b class=ok>命中</b>' if p['hit'] else '未命中'}</p>"
    return (f"<p class='big'>{_arrow(p['pred_up'])}　上漲機率 {_pct(p['prob_up'], 0)}</p>"
            f"<p class='sub'>{p['made_at']} 產生・方法：{html.escape(p['method_name'])}</p>{res}"
            "<div class='scroll'><table><tr><th>依據</th><th>漲跌</th></tr>"
            f"<tr><td>台指期夜盤（對前一日日盤收盤）</td><td class='{_sign_cls(x.get('night_ret'))}'>{_pct(x.get('night_ret'), 2, True)}</td></tr>"
            f"<tr><td>美股 S&amp;P 500</td><td class='{_sign_cls(x.get('spx'))}'>{_pct(x.get('spx'), 2, True)}</td></tr>"
            f"<tr><td>費城半導體</td><td class='{_sign_cls(x.get('sox'))}'>{_pct(x.get('sox'), 2, True)}</td></tr>"
            f"<tr><td>台積電 ADR</td><td class='{_sign_cls(x.get('tsm'))}'>{_pct(x.get('tsm'), 2, True)}</td></tr>"
            "</table></div>"
            "<details><summary>各方法回測命中率</summary>"
            f"{_stats_table(p.get('stats') or (meta.get('preopen') or {}).get('stats'), p.get('method'))}</details>")


def _intraday_html(p: dict) -> str:
    if not p:
        return "<p class='note'>今天的盤中收盤預測尚未產生（每個交易日約 09:40）。</p>"
    if p.get("status") == "no_model":
        return "<p class='note'>預測模型的歷史資料還在建立中，今天沒有盤中預測。</p>"
    if p.get("status") != "ok":
        return "<p class='note'>今天沒有盤中預測（可能休市）。</p>"
    res = ""
    if p.get("hit") is not None:
        res = (f"<p class='sub'>實際收盤 {_n(p['close_actual'])}，方向{'<b class=ok>命中</b>' if p['hit'] else '未命中'}，"
               f"{'在' if p['in_range'] else '不在'}預測區間內</p>")
    bt = p.get("close_backtest") or {}
    return (f"<p class='big'>{_arrow(p['pred_up'])}　收盤上漲機率 {_pct(p['prob_up'], 0)}</p>"
            f"<p class='sub'>{p['made_at']} 產生（報價時間 {html.escape(str(p.get('quote_time', '')))}），"
            f"當時加權指數 {_n(p['index'])}（{_pct(p['index'] / p['prev'] - 1, 2, True)}）・方法：{html.escape(p['method_name'])}</p>"
            f"<p class='sub'>收盤預估 <b>{_n(p['close_mid'], 0)}</b>，80% 區間 {_n(p['close_lo'], 0)} ～ {_n(p['close_hi'], 0)}</p>{res}"
            "<details><summary>各方法回測命中率</summary>"
            f"{_stats_table(p.get('stats'), p.get('method'))}"
            f"<p class='note'>收盤點位區間回測：實際收盤落在 80% 區間內的比例 {_pct(bt.get('coverage_80'), 0)}，"
            f"平均誤差 {_pct(bt.get('mae'), 2)}。</p></details>")


def _rows(lst, val="chg", amt=False):
    out = []
    for r in lst or []:
        v = r.get(val)
        cell = f"{v / 1e8:+,.1f} 億" if amt else _pct(v, 1, True)
        out.append(f"<tr><td>{html.escape(str(r['code']))} {html.escape(str(r.get('name', '')))}</td>"
                   f"<td class='{_sign_cls(v)}'>{cell}</td></tr>")
    return "".join(out)


def _close_html(c: dict) -> str:
    if not c:
        return "<p class='note'>今天的收盤整理尚未產生（每個交易日約 13:45）。</p>"
    if c.get("status") != "ok":
        return "<p class='note'>今天休市。</p>"
    b = c.get("breadth") or {}
    out = (f"<p class='big'><span class='{_sign_cls(c['chg'])}'>{_n(c['index'])}　{_pct(c['chg'], 2, True)}</span></p>"
           f"<p class='sub'>最高 {_n(c.get('high'))}・最低 {_n(c.get('low'))}"
           + (f"・成交金額 {c['value'] / 1e8:,.0f} 億" if c.get("value") else "") + "</p>")
    if b:
        out += (f"<p class='sub'>上漲 <span class='up'>{b['up']}</span>（漲停 {b['limit_up']}）・"
                f"下跌 <span class='down'>{b['down']}</span>（跌停 {b['limit_down']}）・持平 {b['flat']}</p>")
    sec = c.get("sectors") or []
    if sec:
        out += ("<h3>類股</h3><div class='scroll'><table><tr><th>最強</th><th></th></tr>"
                + "".join(f"<tr><td>{html.escape(s['name'])}</td><td class='{_sign_cls(s['pct'])}'>{_pct(s['pct'], 2, True)}</td></tr>" for s in sec[:5])
                + "<tr><th>最弱</th><th></th></tr>"
                + "".join(f"<tr><td>{html.escape(s['name'])}</td><td class='{_sign_cls(s['pct'])}'>{_pct(s['pct'], 2, True)}</td></tr>" for s in sec[-5:][::-1])
                + "</table></div>")
    if c.get("top_gainers"):
        out += ("<h3>漲幅前五（成交金額 1 億以上）</h3><div class='scroll'><table>" + _rows(c["top_gainers"]) + "</table></div>"
                "<h3>跌幅前五</h3><div class='scroll'><table>" + _rows(c["top_losers"]) + "</table></div>"
                "<h3>成交金額前五</h3><div class='scroll'><table>" + _rows(c["top_value"]) + "</table></div>")
    h = c.get("holdings") or []
    if h:
        trs = "".join(f"<tr><td>{x['strategy']}</td><td>{html.escape(x['code'])}</td>"
                      f"<td class='{_sign_cls(x.get('day'))}'>{_pct(x.get('day'), 1, True)}</td>"
                      f"<td class='{_sign_cls(x.get('since'))}'>{_pct(x.get('since'), 1, True)}</td></tr>" for x in h)
        out += ("<h3>本月持股（實盤追蹤名單）</h3><div class='scroll'><table><tr><th>策略</th><th>股票</th><th>今日</th>"
                f"<th>進場以來</th></tr>{trs}</table></div>"
                "<p class='note'>月策略的持股依規則抱到換股日，不因每日漲跌調整。</p>")
    return out


def _flows_html(f: dict) -> str:
    if not f:
        return "<p class='note'>三大法人資料約 15:00 後公布，每個交易日約 17:20 整理。</p>"
    a = f["amount"]
    out = ("<div class='scroll'><table><tr><th>法人</th><th>買賣超</th></tr>"
           + "".join(f"<tr><td>{n}</td><td class='{_sign_cls(a.get(k))}'>{(a.get(k) or 0) / 1e8:+,.1f} 億</td></tr>"
                     for k, n in (("foreign", "外資"), ("trust", "投信"), ("dealer", "自營商"), ("total", "合計"))
                     if a.get(k) is not None)
           + "</table></div>")
    for key, n in (("foreign", "外資"), ("trust", "投信")):
        if f.get(key + "_buy"):
            out += (f"<h3>{n}買超前五</h3><div class='scroll'><table>{_rows(f[key + '_buy'], key + '_amt', True)}</table></div>"
                    f"<h3>{n}賣超前五</h3><div class='scroll'><table>{_rows(f[key + '_sell'], key + '_amt', True)}</table></div>")
    return out + "<p class='note'>個股金額＝買賣超股數 × 收盤價，為估計值。</p>"


def _history_html(days: list[dict]) -> str:
    rows = []
    for d in reversed(days[-30:]):
        p, i, c = d.get("preopen") or {}, d.get("intraday") or {}, d.get("close") or {}
        if c.get("status") == "closed" or (not p and not i and not c):
            continue
        hit = lambda x: "" if x.get("hit") is None else ("✓" if x["hit"] else "✗")
        rows.append(f"<tr><td>{d['date'][5:]}</td>"
                    f"<td>{_arrow(p.get('pred_up')) if p.get('status') == 'ok' else '—'} {hit(p)}</td>"
                    f"<td>{_arrow(i.get('pred_up')) if i.get('status') == 'ok' else '—'} {hit(i)}</td>"
                    f"<td class='{_sign_cls(c.get('chg'))}'>{_pct(c.get('chg'), 2, True) if c.get('status') == 'ok' else '—'}</td></tr>")
    return ("<div class='scroll'><table><tr><th>日期</th><th>開盤前</th><th>09:40</th><th>實際</th></tr>"
            + "".join(rows) + "</table></div>")


def render() -> str:
    days = _days()
    meta = _meta()
    today = days[-1] if days else {}
    lp, li = live_record(days, "preopen"), live_record(days, "intraday")
    rec_html = (f"<p class='sub'>開盤前預測：上線後 {lp['n']} 天命中 {_pct(lp['acc'], 0)}，近 20 天 {_pct(lp['acc20'], 0)}<br>"
                f"09:40 預測：上線後 {li['n']} 天命中 {_pct(li['acc'], 0)}，近 20 天 {_pct(li['acc20'], 0)}</p>")
    d = today.get("date", "")
    wd = f"（{WD[date.fromisoformat(d).weekday()]}）" if d else ""
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>今日盤勢</title><style>{CSS}.big{{font-size:22px;font-weight:700;margin:4px 0}}h3{{font-size:14px;margin:12px 0 4px;color:var(--text-secondary)}}</style></head>
<body><div class="viz-root"><main>
<h1>今日盤勢 {d}{wd}</h1>
<div class="sub"><a href="index.html">← 回測報告</a>・<span class="up">紅＝漲</span> <span class="down">綠＝跌</span></div>
<section class="card"><h2>開盤前預測（08:20）</h2>{_preopen_html(today.get('preopen') or {}, meta)}</section>
<section class="card"><h2>盤中收盤預測（09:40）</h2>{_intraday_html(today.get('intraday') or {})}</section>
<section class="card"><h2>收盤整理（13:45）</h2>{_close_html(today.get('close') or {})}</section>
<section class="card"><h2>三大法人（17:20）</h2>{_flows_html(today.get('flows') or {})}</section>
<section class="card"><h2>預測紀錄</h2>{rec_html}{_history_html(days)}</section>
<p class="note">預測只看大盤（加權指數）收盤相對前一日漲或跌，僅供研究，不構成投資建議。
每天漲跌很難預測，請以頁面上的實際命中率判斷參考價值。</p>
</main></div></body></html>"""


def write() -> None:
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DOCS_DIR / "today.html").write_text(render(), "utf-8")
