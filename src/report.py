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
RECENT_MONTHS = 48


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
    # 近 4 年：共同期間最後 48 個月。全期間平均可能被早期表現撐起來，要分開看
    recent = {s: summarize(rows[-RECENT_MONTHS:]) for s, rows in com_rows.items()}
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
        "summary_full": full, "summary_common": com, "summary_recent": recent, "years": years, "chart": chart,
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
--text-secondary:#52514e;--text-muted:#7a7974;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#8a5cf5;
--good:#0a7d32;--bad:#c42b2b;--up:#d12f2f;--down:#13854a;--grid:#ecebe7}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface-0:#111110;
--surface-1:#1a1a19;--border:#33322f;--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#8f8e86;
--series-1:#3987e5;--series-2:#d95926;--series-3:#8a5cf5;--good:#3fbf6a;--bad:#ef6b6b;--up:#f26d6d;--down:#3fbf6a;--grid:#2a2a28}}
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
th,td{padding:7px 4px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--text-secondary);font-weight:500;font-size:12px}
.scroll{overflow-x:auto}
.ok{color:var(--good);font-weight:600}.no{color:var(--text-secondary)}
.up{color:var(--up)}.down{color:var(--down)}
.verdict{border-radius:10px;padding:12px;margin-bottom:12px;border:2px solid var(--border)}
.verdict .t{font-size:19px;font-weight:700;margin-bottom:4px}
.verdict.go{border-color:var(--up)}.verdict.watch{border-color:var(--series-2)}.verdict.no{border-color:var(--text-muted)}
.verdict ul{margin:6px 0 0;padding-left:20px;font-size:14px}
.win{font-weight:600}.lose{color:var(--text-secondary)}
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


def _sign_cls(x):
    """台股習慣：紅漲綠跌。"""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return "up" if x > 0 else "down" if x < 0 else ""


def _vs_bench(r, bench):
    if r is None or bench is None:
        return "<td>—</td>"
    d = r - bench
    return (f"<td class='win'>贏 {abs(d) * 100:.1f}</td>" if d > 0
            else f"<td class='lose'>輸 {abs(d) * 100:.1f}</td>")


def _picks_table(picks, show_ret=True, bench=None, bench_name="0050"):
    rows = []
    has_yoy = bool(picks) and "rev_yoy" in picks[0]
    for pk in picks:
        r = pk.get("ret")
        rows.append(
            f"<tr><td>{html.escape(pk['code'])} {html.escape(pk['name'])}</td>"
            + (f"<td>{_pct(pk.get('rev_yoy'), 0, True)}</td>" if has_yoy else "")
            + (f"<td class='{_sign_cls(r)}'>{_pct(r, 1, True)}</td>{_vs_bench(r, bench)}" if show_ret else "")
            + "</tr>")
    head = "<tr><th>股票</th>" + ("<th>營收年增</th>" if has_yoy else "") + \
           (f"<th>報酬</th><th>vs {bench_name}</th>" if show_ret else "") + "</tr>"
    note = (f"<p class='note'>報酬已扣手續費與證交稅；vs {bench_name} 為與 {bench_name} 相差的百分點。</p>"
            if show_ret else "")
    return f"<div class='scroll'><table>{head}{''.join(rows)}</table></div>{note}"


MIN_MONTHS_FOR_VERDICT = 24
ROBUST_TEXT = {
    "robust": "成本加倍、改變檔數與流動性門檻後，仍同時贏隨機選股、等權重與 0050",
    "fragile": "部分變化下優勢消失，結果偏脆弱",
    "none": "多數變化下都贏不了隨機選股、等權重或 0050",
}


def verdict(data: dict) -> dict:
    """依回測結果給出結論。

    go    ：選股勝率 ≥ 60%、t 值 ≥ 2、年化報酬贏 0050 → 可參考本月清單
    watch ：勝率明顯高於隨機（t ≥ 2）且年化贏 0050，但未達 6 成 → 只觀察
    no    ：沒有策略穩定勝過隨機選股 → 不建議依清單買進
    """
    target = data["config"]["target"]
    com = {s: sm for s, sm in data["summary_common"].items()
           if sm and (sm.get("months") or 0) >= MIN_MONTHS_FOR_VERDICT}
    if not com:
        return {"level": "wait", "strategy": None}

    def beats(sm):
        return (sm.get("cagr") or -9) > (sm.get("bench_cagr") or 9)

    def sig(sm):
        return (sm.get("t_stat") or 0) >= 2

    go = [s for s, sm in com.items() if sm["pick_hit"] >= target and sig(sm) and beats(sm)]
    if go:
        s = max(go, key=lambda k: com[k]["pick_hit"])
        return {"level": "go", "strategy": s, "sm": com[s]}
    watch = [s for s, sm in com.items() if sig(sm) and beats(sm) and sm["pick_hit"] > sm["baseline_hit"]]
    if watch:
        s = max(watch, key=lambda k: com[k]["pick_hit"])
        return {"level": "watch", "strategy": s, "sm": com[s]}
    s = max(com, key=lambda k: com[k]["pick_hit"])
    return {"level": "no", "strategy": s, "sm": com[s]}


def _next_exit_hint(decision: str) -> str:
    """下次換股日的大約日期（下月 11 日起第一個平日的下一個平日；未計國定假日）。"""
    d = pd.Timestamp(decision) + pd.offsets.MonthBegin(1) + pd.Timedelta(days=config.REBALANCE_DAY - 1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    d += pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _verdict_html(data: dict) -> str:
    v = verdict(data)
    bench = data["config"]["bench"]
    names = STRATEGIES
    if v["level"] == "wait":
        return ("<div class='verdict no'><div class='t'>結論：資料還不夠，暫時沒有結論</div>"
                f"<p class='sub'>回測需要至少 {MIN_MONTHS_FOR_VERDICT} 個月的結果才會判斷。</p></div>")
    s, sm = v["strategy"], v["sm"]
    stats = (f"<ul><li>選股勝率 {_pct(sm['pick_hit'])}（目標 60%，隨機選股 {_pct(sm['baseline_hit'])}）</li>"
             f"<li>年化報酬 {_pct(sm['cagr'], 1, True)}，同期 {bench} {_pct(sm['bench_cagr'], 1, True)}，"
             f"全部合格股票等權重 {_pct(sm.get('ew_cagr'), 1, True)}</li>"
             f"<li>t 值 {_num(sm['t_stat'])}（≥ 2 才算不是運氣）・回測 {sm['months']} 個月</li></ul>")
    title = {
        "go": f"結論：可參考策略 {s}（{names[s]}）的本月清單",
        "watch": f"結論：策略 {s} 有優勢但未達 6 成，建議只觀察",
        "no": "結論：目前沒有策略穩定勝過 0050，不建議依清單買進",
    }[v["level"]]
    out = f"<div class='verdict {v['level']}'><div class='t'>{title}</div>{stats}"
    ew = sm.get("ew_cagr")
    if ew is not None and sm.get("cagr") is not None and sm["cagr"] <= ew:
        out += (f"<p class='sub'>⚠️ 年化報酬沒有贏過「全部合格股票等權重」，"
                f"贏 {bench} 的部分可能只是等權重、偏中小型股的效果，不是選股能力。</p>")
    rc = (data.get("summary_recent") or {}).get(s) or {}
    if rc.get("cagr") is not None and rc.get("bench_cagr") is not None:
        lost = rc["cagr"] <= rc["bench_cagr"]
        out += (f"<p class='sub'>{'⚠️ ' if lost else ''}近 4 年（{rc['start']}～{rc['end']}）年化 {_pct(rc['cagr'], 1, True)}，"
                f"同期 {bench} {_pct(rc['bench_cagr'], 1, True)}"
                f"{'，沒有贏過 ' + bench + '，全期間的優勢主要來自較早的年份' if lost else ''}。</p>")
    rb = (data.get("robustness") or {}).get("verdict", {}).get(s)
    if rb:
        out += f"<p class='sub'>穩健度：{ROBUST_TEXT[rb]}（見下方「穩健度檢查」）</p>"
    out += "</div>"

    live = data.get("live") or {}
    picks = (live.get("strategies") or {}).get(s)
    if not picks:
        return out + "<p class='note'>本月清單尚未產生（資料回補中或還沒到決策日）。</p>"
    label = {"go": "本月清單", "watch": "本月清單（僅供觀察）", "no": f"表現最好的策略 {s} 本月清單（僅供觀察，不建議買進）"}[v["level"]]
    entry = f"{live['entry']} 收盤買進" if live.get("entry") else "下一個交易日收盤買進"
    out += (f"<h2 style='margin-top:4px'>{label}</h2>"
            f"<p class='sub'>每檔資金相同，{entry}，約 {_next_exit_hint(str(live['decision']))} 收盤賣出並換成下個月清單。"
            f"資料至 {live['asof']}，同期 {bench} {_pct(live.get('bench_ret'), 1, True)}。</p>"
            + _picks_table(picks, live.get("entry") is not None, live.get("bench_ret"), bench))
    return out


def _robust_html(data: dict) -> str:
    rb = data.get("robustness") or {}
    var = rb.get("variants") or {}
    if not var:
        return ""
    names = STRATEGIES
    tabs = "".join(f"<button data-key='{s}' aria-pressed='{str(i == 0).lower()}'>{s}</button>" for i, s in enumerate(var))
    panes = []
    for i, (s, rows) in enumerate(var.items()):
        trs = []
        for r in rows:
            if not r.get("months"):
                continue
            ok = r.get("beats_random") and r.get("beats_ew") and r.get("beats_bench")
            trs.append(f"<tr><td>{html.escape(r['name'])}</td><td>{_pct(r['pick_hit'])}</td><td>{_pct(r['baseline_hit'])}</td>"
                       f"<td>{_pct(r['cagr'], 1, True)}</td><td>{_pct(r['ew_cagr'], 1, True)}</td>"
                       f"<td>{_pct(r.get('bench_cagr'), 1, True)}</td>"
                       f"<td class='{'win' if ok else 'lose'}'>{'✓' if ok else '✗'}</td></tr>")
        v = (rb.get("verdict") or {}).get(s, "none")
        fl = (rb.get("failed") or {}).get(s) or []
        fl_txt = f"未通過：{'、'.join(html.escape(x) for x in fl)}" if fl else "全部通過"
        panes.append(f"<div data-pane='robust' data-key='{s}' {'hidden' if i else ''}>"
                     f"<p class='sub'><b>{s} {names[s]}</b>：{ROBUST_TEXT[v]}（{fl_txt}）</p>"
                     "<div class='scroll'><table><tr><th>變化</th><th>選股勝率</th><th>隨機基準</th><th>年化</th>"
                     f"<th>等權重</th><th>{data['config']['bench']}</th><th>三者都贏</th></tr>{''.join(trs)}</table></div></div>")
    return (f"<section class='card'><h2>穩健度檢查（共同期間）</h2>"
            "<p class='sub'>好的策略在不同成本、檔數、流動性門檻下都應該站得住，而不是只在預設值才有效。"
            "這些變化在看結果前就固定，不依結果挑參數。</p>"
            f"<div class='tabs' data-group='robust'>{tabs}</div>{''.join(panes)}"
            "<p class='note'>成本 ×1.5、×2 用來模擬滑價與較差的成交價（回測以收盤價成交）。</p></section>")


def _live_track_html(data: dict) -> str:
    track = data.get("live_track")
    if track is None:
        return ""
    if not track:
        body = "<p class='note'>還沒有凍結的名單。下個決策日起每月自動記錄。</p>"
    else:
        rows = []
        for t in reversed(track):
            st = {"closed": "已結束", "holding": "持有中", "waiting": "等待進場"}[t["status"]]
            cells = "".join(
                f"<td class='{_sign_cls((t['strategies'].get(s) or {}).get('port_ret'))}'>"
                f"{_pct((t['strategies'].get(s) or {}).get('port_ret'), 1, True)}</td>" for s in STRATEGIES)
            late = " *" if t.get("late") else ""
            rows.append(f"<tr><td>{t['month']}{late}</td><td>{st}</td>{cells}"
                        f"<td class='{_sign_cls(t.get('bench_ret'))}'>{_pct(t.get('bench_ret'), 1, True)}</td></tr>")
        head = ("<tr><th>月份</th><th>狀態</th>" + "".join(f"<th>{s}</th>" for s in STRATEGIES)
                + f"<th>{data['config']['bench']}</th></tr>")
        body = (f"<div class='scroll'><table>{head}{''.join(rows)}</table></div>"
                "<p class='note'>* 該月是在決策日之後才開始記錄，不算事前預測。報酬已扣成本。</p>")
    return ("<section class='card'><h2>實盤追蹤（樣本外）</h2>"
            "<p class='sub'>每月決策後把三個策略的名單凍結存檔，之後只算報酬、不再更動。"
            "回測規則是看過歷史才設計的，這裡的真實月份才是真正的驗證。</p>"
            f"{body}</section>")


def _quality_html(data: dict) -> str:
    q = data.get("quality")
    if q is None:
        return ""
    if not q:
        return "<p class='sub'>✓ 資料品質檢查沒有發現問題。</p>"
    icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
    items = "".join(f"<li>{icon.get(x['level'], '')} {html.escape(x['msg'])}</li>" for x in q)
    return f"<ul class='sub' style='padding-left:18px'>{items}</ul>"


def render(data: dict) -> str:
    names = STRATEGIES
    com, full = data["summary_common"], data["summary_full"]
    upd = data.get("update") or {}
    best = max((s for s in com if com[s]), key=lambda s: com[s]["pick_hit"], default=None)

    kpi = _verdict_html(data)

    # 策略比較表
    def row(s, sm):
        if not sm:
            return f"<tr><td><span class='sw' style='background:{COLORS[s]}'></span>{s} {names[s]}</td><td colspan=8 class='no'>資料不足</td></tr>"
        ok = sm["pick_hit"] >= data["config"]["target"]
        return (f"<tr><td><span class='sw' style='background:{COLORS[s]}'></span>{s} {names[s]}</td>"
                f"<td class='{'ok' if ok else ''}'>{_pct(sm['pick_hit'])}{' ✓' if ok else ''}</td>"
                f"<td>{_pct(sm['last12_hit'])}</td><td>{_pct(sm['month_win'])}</td>"
                f"<td>{_pct(sm['cagr'], 1, True)}</td><td>{_pct(sm.get('ew_cagr'), 1, True)}</td><td>{_pct(sm['bench_cagr'], 1, True)}</td>"
                f"<td>{_pct(sm['mdd'])}</td><td>{_num(sm['t_stat'])}</td></tr>")
    head = ("<tr><th>策略</th><th>選股勝率</th><th>近12月</th><th>月勝率</th><th>年化</th><th>等權重年化</th>"
            f"<th>{data['config']['bench']}年化</th><th>最大回撤</th><th>t值</th></tr>")
    tbl_com = "".join(row(s, com.get(s)) for s in names)
    rec = data.get("summary_recent") or {}
    tbl_rec = "".join(row(s, rec.get(s)) for s in names)
    rec_any = next((v for v in rec.values() if v), {})
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
                        f"{_picks_table(pk, live.get('entry') is not None, live.get('bench_ret'), data['config']['bench'])}</div>"
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
            f"<div data-pane='last' data-key='{s}' {'hidden' if i else ''}><p class='sub'>{r['month']}：10 檔平均 "
            f"<span class='{_sign_cls(r['port_ret'])}'>{_pct(r['port_ret'], 1, True)}</span>，"
            f"{data['config']['bench']} <span class='{_sign_cls(r['bench_ret'])}'>{_pct(r['bench_ret'], 1, True)}</span>，"
            f"{'贏' if r['port_ret'] > r['bench_ret'] else '輸'} {data['config']['bench']}；"
            f"{round(r['hit_rate'] * len(r['picks']))}/{len(r['picks'])} 檔贏 {data['config']['bench']}</p>"
            f"{_picks_table(r['picks'], True, r['bench_ret'], data['config']['bench'])}</div>" for i, (s, r) in enumerate(lm.items()))
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
<div class="sub"><a href="today.html">今日盤勢與預測 →</a></div>
<div class="sub">資料至 {data['data_until']}・更新於 {data['generated']}・每月選 {data['config']['top_n']} 檔・來回成本 {_pct(data['config']['cost'], 3)}・<span class="up">紅＝漲</span> <span class="down">綠＝跌</span></div>

<section class="card">{kpi}</section>

<section class="card"><h2>策略比較（共同期間）</h2><div class="scroll"><table>{head}{tbl_com}</table></div>
<p class="note">選股勝率＝選中個股（扣成本）跑贏 {data['config']['bench']} 的比例。t 值 &gt; 2 代表勝率明顯高於隨機選股，不太可能是運氣。
等權重年化＝每月平均持有全部合格股票（同樣扣成本），用來檢查策略是不是只靠等權重、偏中小型股贏 {data['config']['bench']}。</p>
<h2 style="margin-top:14px">近 4 年（{rec_any.get('start', '—')}～{rec_any.get('end', '—')}）</h2>
<div class="scroll"><table>{head}{tbl_rec}</table></div>
<p class="note">全期間的平均可能被早期表現撐起來；這張表只看最後 {RECENT_MONTHS} 個月，確認優勢是否還在。</p>
<details><summary>各策略完整期間</summary><div class="scroll"><table>{head}{tbl_full}</table></div></details></section>

<section class="card"><h2>近 12 個月選股勝率</h2><div class="legend">{legend}<span>虛線＝全市場平均</span></div>
<div class="chart" id="c-hit"></div></section>

<section class="card"><h2>累積淨值</h2><div class="legend">{legend}<span>虛線＝{data['config']['bench']}</span></div>
<div class="chart" id="c-nav"></div></section>

<section class="card"><h2>三個策略的本月清單</h2>{live_html}</section>
<section class="card"><h2>上個完整月份結果</h2>{last_html}</section>

<section class="card"><h2>逐年選股勝率</h2><div class="scroll"><table>{ytbl}</table></div></section>

{_robust_html(data)}
{_live_track_html(data)}
<section class="card"><h2>資料狀態</h2>{_quality_html(data)}
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
