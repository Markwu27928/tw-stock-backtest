"""產生手機版報告頁 docs/index.html 與 docs/results.json。"""
from __future__ import annotations

import html
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config
from .backtest import STRATEGIES, by_year, common_period, summarize

COLORS = {"A": "var(--series-1)", "B": "var(--series-2)", "C": "var(--series-3)"}


def _pct(x, digits=1, sign=False):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"


def _num(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.strftime("%Y-%m-%d")
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _rolling_hit(rows, window=12):
    out = []
    for i in range(len(rows)):
        w = rows[max(0, i - window + 1):i + 1]
        n = sum(len(r["picks"]) for r in w)
        h = sum(sum(pk["ret"] > r["bench_ret"] for pk in r["picks"]) for r in w)
        out.append(h / n if n and len(w) == window else None)
    return out


def live_table(panel, live) -> dict:
    """本月持股：進場後到最新收盤的報酬。"""
    if not live:
        return {}
    last = panel.dates[-1]
    out = {"month": live["month"], "decision": live["decision"], "entry": live["entry"], "asof": last, "strategies": {}}
    b = config.BENCH
    bench_now = None
    if live["entry"] is not None and b in panel.tri.columns:
        bench_now = float(panel.tri.at[last, b] / panel.tri.at[live["entry"], b] - 1)
    out["bench_ret"] = bench_now
    f = live["features"]
    for s, sc in live["scores"].items():
        top = sc.dropna().sort_values(ascending=False).head(config.TOP_N)
        picks = []
        for c in top.index:
            r = None
            if live["entry"] is not None:
                r = float(panel.tri.at[last, c] / panel.tri.at[live["entry"], c] - 1 - config.ROUND_TRIP_COST)
            picks.append({"code": c, "name": str(panel.names.get(c, "")), "score": float(top[c]),
                          "rev_yoy": float(f.at[c, "rev_yoy"]), "close": float(f.at[c, "close"]), "ret": r})
        out["strategies"][s] = picks
    return out


def build(panel, bt: dict, update_info: dict) -> dict:
    res = bt["results"]
    full = {s: summarize(rows) for s, rows in res.items()}
    com_rows = common_period(res)
    com = {s: summarize(rows) for s, rows in com_rows.items()}
    years = {s: by_year(rows) for s, rows in com_rows.items()}
    # 圖表資料（共同期間）
    ref = next((rows for rows in com_rows.values() if rows), [])
    chart = {
        "x": [r["month"] for r in ref],
        "hit": {s: _rolling_hit(rows) for s, rows in com_rows.items()},
        "baseline": _rolling_baseline(ref),
        "nav": {s: list(np.cumprod([1 + r["port_ret"] for r in rows])) for s, rows in com_rows.items()},
        "bench_nav": list(np.cumprod([1 + r["bench_ret"] for r in ref])),
    }
    last_month = {s: rows[-1] for s, rows in res.items() if rows}
    return _clean({
        "generated": datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M"),
        "data_until": panel.dates[-1],
        "update": update_info,
        "config": {"top_n": config.TOP_N, "cost": config.ROUND_TRIP_COST, "bench": config.BENCH,
                   "target": config.TARGET_HIT, "min_value": config.MIN_VALUE_20, "min_price": config.MIN_PRICE},
        "summary_full": full, "summary_common": com, "years": years, "chart": chart,
        "live": live_table(panel, bt["live"]), "last_month": last_month,
    })


def _rolling_baseline(rows, window=12):
    out = []
    for i in range(len(rows)):
        w = rows[max(0, i - window + 1):i + 1]
        out.append(float(np.mean([r["universe_hit"] for r in w])) if len(w) == window else None)
    return out


# ---------------------------------------------------------------- HTML
CSS = """
.viz-root{color-scheme:light;--surface-0:#f4f4f2;--surface-1:#fcfcfb;--border:#e2e1dc;--text-primary:#0b0b0b;
--text-secondary:#52514e;--text-muted:#7a7974;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;
--good:#0a7d32;--bad:#c42b2b;--grid:#ecebe7}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-0:#111110;
--surface-1:#1a1a19;--border:#33322f;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--good:#3fbf6a;--bad:#ef6b6b;--grid:#2a2a28}}
*{box-sizing:border-box}body{margin:0}
.viz-root{background:var(--surface-0);color:var(--text-primary);font:15px/1.55 -apple-system,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif;min-height:100vh}
main{max-width:760px;margin:0 auto;padding:16px 14px 48px}
h1{font-size:20px;margin:4px 0 2px}h2{font-size:17px;margin:0 0 10px}
.sub{color:var(--text-secondary);font-size:13px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:14px;margin:14px 0}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.kpi{border:1px solid var(--border);border-radius:10px;padding:10px}
.kpi .v{font-size:22px;font-weight:650;font-variant-numeric:tabular-nums}
.kpi .l{font-size:12px;color:var(--text-secondary)}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:baseline}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}
th,td{padding:7px 6px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--text-secondary);font-weight:500;font-size:12px}
.scroll{overflow-x:auto}
.ok{color:var(--good);font-weight:600}.no{color:var(--text-secondary)}
.pos{color:var(--good)}.neg{color:var(--bad)}
.tabs{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
.tabs button{border:1px solid var(--border);background:transparent;color:var(--text-primary);border-radius:999px;padding:5px 12px;font:inherit;font-size:13px}
.tabs button[aria-pressed=true]{background:var(--text-primary);color:var(--surface-1)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--text-secondary);margin:4px 0 6px}
.chart{position:relative}.chart svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:absolute;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;
padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;white-space:nowrap;z-index:2}
.note{font-size:12px;color:var(--text-muted)}
.bar{height:8px;background:var(--grid);border-radius:4px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--series-1)}
details summary{cursor:pointer;color:var(--text-secondary);font-size:13px}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
function css(v){return getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim()}
function lineChart(el, o){
  const W=700,H=380,L=56,R=10,T=14,B=30, n=o.x.length; if(!n){el.innerHTML='<p class="note">資料不足</p>';return;}
  const all=o.series.flatMap(s=>s.values).filter(v=>v!=null).concat(o.refs?o.refs.map(r=>r.y):[]);
  let lo=Math.min(...all),hi=Math.max(...all); if(o.yMin!=null)lo=Math.min(lo,o.yMin); const pad=(hi-lo)*0.08||0.05; lo-=pad;hi+=pad;
  const X=i=>L+(W-L-R)*(n==1?0.5:i/(n-1)), Y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${o.label}">`;
  for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4;s+=`<line x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--grid)"/>`+
    `<text x="${L-6}" y="${Y(v)+4}" text-anchor="end" font-size="16" fill="var(--text-muted)">${o.fmt(v)}</text>`;}
  const step=Math.max(1,Math.ceil(n/6));for(let i=0;i<n;i+=step)s+=`<text x="${X(i)}" y="${H-6}" text-anchor="middle" font-size="16" fill="var(--text-muted)">${o.x[i]}</text>`;
  (o.refs||[]).forEach(r=>{s+=`<line x1="${L}" x2="${W-R}" y1="${Y(r.y)}" y2="${Y(r.y)}" stroke="var(--text-secondary)" stroke-dasharray="5 4"/>`+
    `<text x="${W-R}" y="${Y(r.y)-5}" text-anchor="end" font-size="16" fill="var(--text-secondary)">${r.label}</text>`});
  o.series.forEach(se=>{let d='',pen=false;se.values.forEach((v,i)=>{if(v==null){pen=false;return}d+=(pen?'L':'M')+X(i).toFixed(1)+' '+Y(v).toFixed(1);pen=true});
    s+=`<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2" ${se.dash?'stroke-dasharray="3 3"':''} stroke-linejoin="round"/>`;});
  s+=`<line class="xh" y1="${T}" y2="${H-B}" stroke="var(--text-muted)" visibility="hidden"/>`;
  o.series.forEach((se,k)=>{s+=`<circle class="dot${k}" r="4" fill="${se.color}" stroke="var(--surface-1)" stroke-width="2" visibility="hidden"/>`});
  s+=`<rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="transparent"/></svg><div class="tip"></div>`;
  el.innerHTML=s; const svg=el.querySelector('svg'),tip=el.querySelector('.tip'),xh=svg.querySelector('.xh');
  function move(ev){const p=ev.touches?ev.touches[0]:ev,b=svg.getBoundingClientRect(),sx=(p.clientX-b.left)*W/b.width;
    let i=Math.round((sx-L)/(W-L-R)*(n-1));i=Math.max(0,Math.min(n-1,i));
    xh.setAttribute('x1',X(i));xh.setAttribute('x2',X(i));xh.setAttribute('visibility','visible');
    let h=`<b>${o.x[i]}</b>`;o.series.forEach((se,k)=>{const c=svg.querySelector('.dot'+k),v=se.values[i];
      if(v==null){c.setAttribute('visibility','hidden');return}c.setAttribute('cx',X(i));c.setAttribute('cy',Y(v));c.setAttribute('visibility','visible');
      h+=`<br><span class="sw" style="background:${se.color}"></span>${se.name}：${o.fmt(v)}`});
    tip.innerHTML=h;tip.style.display='block';const left=X(i)/W*b.width;tip.style.left=Math.min(Math.max(0,left-60),b.width-150)+'px';tip.style.top='0px';}
  function out(){tip.style.display='none';xh.setAttribute('visibility','hidden');o.series.forEach((_,k)=>svg.querySelector('.dot'+k).setAttribute('visibility','hidden'))}
  svg.addEventListener('mousemove',move);svg.addEventListener('touchmove',move,{passive:true});svg.addEventListener('mouseleave',out);svg.addEventListener('touchend',out);
}
const NAMES=D.names, COL=D.colors;
const pctf=v=>(v*100).toFixed(0)+'%';
const hitSeries=Object.keys(D.chart.hit).filter(s=>D.chart.hit[s].length).map(s=>({name:s+' '+NAMES[s],color:COL[s],values:D.chart.hit[s]}));
hitSeries.push({name:'全市場平均',color:'var(--text-muted)',values:D.chart.baseline,dash:true});
lineChart(document.getElementById('c-hit'),{x:D.chart.x,series:hitSeries,fmt:pctf,refs:[{y:D.config.target,label:'目標 60%'}],label:'近12個月選股勝率'});
const navSeries=Object.keys(D.chart.nav).filter(s=>D.chart.nav[s].length).map(s=>({name:s+' '+NAMES[s],color:COL[s],values:D.chart.nav[s]}));
navSeries.push({name:D.config.bench,color:'var(--text-muted)',values:D.chart.bench_nav,dash:true});
lineChart(document.getElementById('c-nav'),{x:D.chart.x,series:navSeries,fmt:v=>v.toFixed(2),label:'累積淨值'});
document.querySelectorAll('.tabs').forEach(t=>t.addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;
  t.querySelectorAll('button').forEach(x=>x.setAttribute('aria-pressed',x===b));
  const grp=t.dataset.group;document.querySelectorAll('[data-pane="'+grp+'"]').forEach(p=>p.hidden=p.dataset.key!==b.dataset.key)}));
"""


def _picks_table(picks, show_ret=True, bench=None):
    rows = []
    for pk in picks:
        r = pk.get("ret")
        cls = "" if r is None else ("pos" if bench is not None and r > bench else "neg" if bench is not None else "")
        rows.append(
            f"<tr><td>{html.escape(pk['code'])} {html.escape(pk['name'])}</td>"
            + (f"<td>{_pct(pk.get('rev_yoy'), 0, True)}</td>" if "rev_yoy" in pk else "")
            + (f"<td class='{cls}'>{_pct(r, 1, True)}</td>" if show_ret else "") + "</tr>")
    head = "<tr><th>股票</th>" + ("<th>營收YoY</th>" if picks and "rev_yoy" in picks[0] else "") + \
           ("<th>報酬(扣成本)</th>" if show_ret else "") + "</tr>"
    return f"<div class='scroll'><table>{head}{''.join(rows)}</table></div>"


def render(data: dict) -> str:
    names = STRATEGIES
    com, full = data["summary_common"], data["summary_full"]
    upd = data.get("update") or {}
    best = max((s for s in com if com[s]), key=lambda s: com[s]["pick_hit"], default=None)

    # 頂部 KPI
    if best:
        b = com[best]
        reach = b["pick_hit"] is not None and b["pick_hit"] >= data["config"]["target"]
        kpi = f"""<div class="kpis">
<div class="kpi"><div class="v {'ok' if reach else ''}">{_pct(b['pick_hit'])}</div><div class="l">最佳選股勝率（{best}）</div></div>
<div class="kpi"><div class="v">{_pct(b['baseline_hit'])}</div><div class="l">隨機選股基準</div></div>
<div class="kpi"><div class="v">{_pct(b['month_win'])}</div><div class="l">月勝率（組合贏 {data['config']['bench']}）</div></div></div>
<p class="sub" style="margin:10px 0 0">目標：選中個股跑贏 {data['config']['bench']} 的比例 ≥ 60%。
目前{'<span class="ok">已達標</span>' if reach else '尚未達標'}。共同比較期間 {b['start']}～{b['end']}（{b['months']} 個月）。</p>"""
    else:
        kpi = "<p>歷史資料回補中，回測結果尚未產生。</p>"

    # 策略比較表
    def row(s, sm):
        if not sm:
            return f"<tr><td><span class='sw' style='background:{COLORS[s]}'></span>{s} {names[s]}</td><td colspan=7 class='no'>資料不足</td></tr>"
        ok = sm["pick_hit"] >= data["config"]["target"]
        return (f"<tr><td><span class='sw' style='background:{COLORS[s]}'></span>{s} {names[s]}</td>"
                f"<td class='{'ok' if ok else ''}'>{_pct(sm['pick_hit'])}{' ✓' if ok else ''}</td>"
                f"<td>{_pct(sm['last12_hit'])}</td><td>{_pct(sm['month_win'])}</td>"
                f"<td>{_pct(sm['cagr'], 1, True)}</td><td>{_pct(sm['bench_cagr'], 1, True)}</td>"
                f"<td>{_pct(sm['mdd'])}</td><td>{_num(sm['t_stat'])}</td></tr>")
    head = ("<tr><th>策略</th><th>選股勝率</th><th>近12月</th><th>月勝率</th><th>年化</th>"
            f"<th>{data['config']['bench']}年化</th><th>最大回撤</th><th>t值</th></tr>")
    tbl_com = "".join(row(s, com.get(s)) for s in names)
    tbl_full = "".join(row(s, full.get(s)) for s in names)

    # 逐年
    yrs = sorted({y["year"] for ys in data["years"].values() for y in ys})
    ymap = {s: {y["year"]: y for y in ys} for s, ys in data["years"].items()}
    yrows = []
    for y in yrs:
        cells = "".join(f"<td>{_pct(ymap[s].get(y, {}).get('pick_hit'), 0)}</td>" for s in names)
        anyrow = next((ymap[s][y] for s in names if y in ymap[s]), {})
        yrows.append(f"<tr><td>{y}</td>{cells}<td>{_pct(anyrow.get('bench'), 1, True)}</td></tr>")
    ytbl = (f"<tr><th>年度</th>{''.join(f'<th>{s} 勝率</th>' for s in names)}<th>{data['config']['bench']}</th></tr>"
            + "".join(yrows))

    # 本月持股
    live = data.get("live") or {}
    if live.get("strategies"):
        tabs = "".join(f"<button data-key='{s}' aria-pressed='{str(i == 0).lower()}'>{s} {names[s]}</button>"
                       for i, s in enumerate(live["strategies"]))
        panes = "".join(f"<div data-pane='live' data-key='{s}' {'hidden' if i else ''}>"
                        f"{_picks_table(pk, live.get('entry') is not None, live.get('bench_ret'))}</div>"
                        for i, (s, pk) in enumerate(live["strategies"].items()))
        entry_txt = f"{live['entry']} 收盤進場" if live.get("entry") else "下一個交易日收盤進場"
        live_html = (f"<p class='sub'>{live['month']} 月決策日 {live['decision']}，{entry_txt}，持有到下月 11 日後換股。"
                     f"資料至 {live['asof']}，同期 {data['config']['bench']} {_pct(live.get('bench_ret'), 1, True)}。</p>"
                     f"<div class='tabs' data-group='live'>{tabs}</div>{panes}")
    else:
        live_html = "<p class='note'>尚無本月持股（資料回補中或今天還沒到決策日）。</p>"

    # 上個月結果
    lm = data.get("last_month") or {}
    if lm:
        tabs = "".join(f"<button data-key='{s}' aria-pressed='{str(i == 0).lower()}'>{s}</button>" for i, s in enumerate(lm))
        panes = "".join(
            f"<div data-pane='last' data-key='{s}' {'hidden' if i else ''}><p class='sub'>{r['month']}：組合 {_pct(r['port_ret'], 1, True)}，"
            f"{data['config']['bench']} {_pct(r['bench_ret'], 1, True)}，勝率 {_pct(r['hit_rate'], 0)}</p>"
            f"{_picks_table(r['picks'], True, r['bench_ret'])}</div>" for i, (s, r) in enumerate(lm.items()))
        last_html = f"<div class='tabs' data-group='last'>{tabs}</div>{panes}"
    else:
        last_html = "<p class='note'>尚無資料。</p>"

    total = upd.get("days_total") or 0
    done = upd.get("days_done") or 0
    prog = done / total if total else 0
    upd_log = "".join(f"<li>{html.escape(x)}</li>" for x in (upd.get("log") or [])[-10:])

    payload = dict(data, names=names, colors=COLORS)
    legend = "".join(f"<span><span class='sw' style='background:{COLORS[s]}'></span>{s} {names[s]}</span>" for s in names)
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow">
<title>台股 20 日預測回測</title><style>{CSS}</style></head>
<body><div class="viz-root"><main>
<h1>台股 20 日選股回測</h1>
<div class="sub">資料至 {data['data_until']}・更新於 {data['generated']}・每月選 {data['config']['top_n']} 檔・來回成本 {_pct(data['config']['cost'], 3)}</div>

<section class="card"><h2>是否達到 6 成</h2>{kpi}</section>

<section class="card"><h2>策略比較（共同期間）</h2><div class="scroll"><table>{head}{tbl_com}</table></div>
<p class="note">選股勝率＝選中個股（扣成本）跑贏 {data['config']['bench']} 的比例。t 值 &gt; 2 代表勝率明顯高於隨機選股，不太可能是運氣。</p>
<details><summary>各策略完整期間</summary><div class="scroll"><table>{head}{tbl_full}</table></div></details></section>

<section class="card"><h2>近 12 個月選股勝率</h2><div class="legend">{legend}<span>虛線＝全市場平均</span></div>
<div class="chart" id="c-hit"></div></section>

<section class="card"><h2>累積淨值</h2><div class="legend">{legend}<span>虛線＝{data['config']['bench']}</span></div>
<div class="chart" id="c-nav"></div></section>

<section class="card"><h2>本月持股</h2>{live_html}</section>
<section class="card"><h2>上個完整月份結果</h2>{last_html}</section>

<section class="card"><h2>逐年選股勝率</h2><div class="scroll"><table>{ytbl}</table></div></section>

<section class="card"><h2>資料狀態</h2>
<div class="bar"><i style="width:{prog * 100:.1f}%"></i></div>
<p class="sub">歷史交易日 {done}/{total}（{_pct(prog, 0)}），連續完整至 {upd.get('contiguous_until') or '—'}。最近一次抓取 {upd.get('run_at', '—')}。</p>
<details><summary>抓取紀錄</summary><ul class="note">{upd_log}</ul></details></section>

<p class="note">僅供研究，不構成投資建議。回測未計入流動性衝擊與減資等特殊事件；過去績效不代表未來。</p>
</main></div>
<script type="application/json" id="data">{json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')}</script>
<script>{JS}</script></body></html>"""


def write(data: dict) -> None:
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (config.DOCS_DIR / "results.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), "utf-8")
    (config.DOCS_DIR / "index.html").write_text(render(data), "utf-8")
    (config.DOCS_DIR / ".nojekyll").write_text("", "utf-8")
