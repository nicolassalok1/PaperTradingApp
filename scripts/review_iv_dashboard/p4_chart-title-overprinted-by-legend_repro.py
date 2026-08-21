"""
p4_chart-title-overprinted-by-legend_repro.py — skeptic repro (offline, deterministic).

Builds the 3 figures through the view's own functions, applies the Streamlit 1.51
"streamlit" plotly template as the frontend does (TF(): theme written into
layout.template.layout, title bolded; title font 16px x=0 xanchor=left, body/legend
font 12px, bg #0E1117, axes automargin) and renders each figure at realistic
container widths in headless Chrome/Edge from a local file:// URL (plotly.js embedded,
no network). A post_script measures the SVG bounding boxes of the title (.gtitle)
and legend (g.legend) and the number of legend rows; the DOM is dumped and parsed.

Widths (wide mode, sidebar collapsed, 5rem padding each side, 17px scrollbar):
  viewport 1366 -> content 1189, half column 586
  viewport 1536 -> content 1359, half column 671
  viewport 1920 -> content 1743, half column 863
plus the finder's 900 / 560 for comparison.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "p4_chart_overlap_out"
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

captured: dict = {}
st.plotly_chart = lambda fig, *a, **k: captured.__setitem__(k.get("key", "?"), fig)  # type: ignore[assignment]

from app.model.iv_dashboard import analytics as ivx  # noqa: E402
from app.vue.tabs import tab_iv_dashboard as tab  # noqa: E402

# --- synthetic result: 2 years ('2 ans' default), IV available, 5 iv_history points
rng = np.random.default_rng(7)
n = 504 + 40
closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, n))),
                   index=pd.bdate_range(end="2026-08-21", periods=n))
rv = ivx.compute_realized_vol(closes, 20)
pct = ivx.compute_percentile_series(rv, 252)
series = pd.DataFrame({"close": closes, "vol": rv, "vol_percentile": pct}).dropna(subset=["vol"])
analysis = ivx.analyze_forward_vol(series["vol"], forward_window=30, percentile=series["vol_percentile"])
cur = float(series["vol"].iloc[-1])
result = {
    "symbol": "SPY", "rv_window": 20, "series": series, "analysis": analysis,
    "current_iv": {"iv": cur + 0.04, "dte": 30, "expiry": dt.date(2026, 9, 18), "n_contracts": 6,
                   "method": "greeks Alpaca", "feed": "indicative"},
    "iv_history": pd.DataFrame({"date": pd.bdate_range(end="2026-08-21", periods=5),
                                "iv": np.linspace(cur, cur + 0.04, 5)}),
}
tab._render_series_chart(result)
tab._render_forward_chart(result)
tab._render_diff_chart(result)

# --- Streamlit 1.51 plotly template (dark custom theme of this repo), as in TF()/yF()
BG, TXT, HEAD, GRID = "#0E1117", "#fafafa", "#ffffff", "rgba(250,250,250,0.2)"
FONT = "'Source Sans', 'Source Sans Pro', sans-serif"
ST_TEMPLATE_LAYOUT = dict(
    font=dict(color=TXT, family=FONT, size=12),
    title=dict(font=dict(family=FONT, size=16, color=HEAD), pad=dict(l=4), xanchor="left", x=0),
    legend=dict(title=dict(font=dict(size=12, color=TXT), side="top"), valign="top",
                bordercolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=12, color=TXT)),
    paper_bgcolor=BG, plot_bgcolor=BG,
    yaxis=dict(ticklabelposition="outside", zerolinecolor=GRID, title=dict(font=dict(color=TXT, size=14), standoff=24),
               tickcolor=GRID, tickfont=dict(color=TXT, size=12), gridcolor=GRID, automargin=True),
    xaxis=dict(zerolinecolor=GRID, gridcolor=GRID, showgrid=False, tickfont=dict(color=TXT, size=12),
               tickcolor=GRID, title=dict(font=dict(color=TXT, size=14), standoff=16), zeroline=False, automargin=True),
    margin=dict(pad=8, r=0, l=0),
)

POST = """
var gd = document.getElementById('{plot_id}');
function bb(el){ if(!el) return null; var r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; }
var title = gd.querySelector('.gtitle');
var legend = gd.querySelector('g.legend');
var rows = 0;
if (legend) {
  var ys = {};
  legend.querySelectorAll('g.traces').forEach(function(t){ var r=t.getBoundingClientRect(); ys[Math.round(r.top)] = 1; });
  rows = Object.keys(ys).length;
}
var info = {title: bb(title), legend: bb(legend), legend_bg: bb(legend && legend.querySelector('rect.bg')),
            rows: rows, plot: bb(gd.querySelector('.plot')), margin_t: gd._fullLayout.margin.t,
            legend_h: gd._fullLayout.legend._height, title_text: title ? title.textContent : null,
            title_after_legend: (title && legend) ? !!(legend.compareDocumentPosition(title) & Node.DOCUMENT_POSITION_FOLLOWING) : null};
var pre = document.createElement('pre'); pre.id = 'measure'; pre.textContent = JSON.stringify(info);
document.body.appendChild(pre);
"""

BROWSERS = [
    Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path(r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
]
browser = next((b for b in BROWSERS if b.exists()), None)
assert browser, "no chromium browser found"

WIDTHS = {
    "iv_dash_series_chart": [900, 1189, 1359, 1743],
    "iv_dash_forward_chart": [560, 586, 671, 863],
    "iv_dash_diff_chart": [560, 586, 671, 863],
}


def render(fig, width: int, tag: str, with_template: bool) -> dict:
    f = fig.full_figure_for_development(warn=False) if False else fig  # keep the view's figure untouched semantics
    f = f.__class__(f)  # deep copy
    if with_template:
        f.update_layout(template=dict(layout=ST_TEMPLATE_LAYOUT))
        if f.layout.title.text:
            f.update_layout(title_text=f"<b>{f.layout.title.text}</b>")
    f.update_layout(width=width)
    p = OUT / f"{tag}.html"
    p.write_text(f.to_html(include_plotlyjs=True, full_html=True, post_script=POST,
                           config={"responsive": False, "displayModeBar": False}), encoding="utf-8")
    h = int(f.layout.height or 450)
    cmd = [str(browser), "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
           "--disable-extensions", "--no-first-run", f"--window-size={width + 40},{h + 80}",
           "--virtual-time-budget=4000", "--dump-dom", p.resolve().as_uri()]
    dom = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120).stdout
    m = re.search(r'<pre id="measure">(.*?)</pre>', dom, flags=re.S)
    assert m, f"no measurement for {tag}"
    return json.loads(html.unescape(m.group(1)))


def overlap(a, b):
    if not a or not b:
        return 0.0, 0.0
    ox = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    oy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ox, oy


print(f"browser: {browser.name}")
summary = []
for key, fig in captured.items():
    n_legend = sum(1 for t in fig.data if t.showlegend is not False)
    print(f"\n== {key}: {n_legend} legend entries, margin.t={fig.layout.margin.t}, legend y={fig.layout.legend.y} "
          f"yanchor={fig.layout.legend.yanchor}, title.y={fig.layout.title.y!r}")
    for with_template in (True, False):
        for w in WIDTHS[key]:
            tag = f"{key}_{w}_{'st' if with_template else 'plain'}"
            info = render(fig, w, tag, with_template)
            ox, oy = overlap(info["title"], info["legend"])
            tb, lb = info["title"], info["legend"]
            th = (tb[3] - tb[1]) if tb else 0
            line = (f"  {'streamlit-template' if with_template else 'plain-template   '} w={w:4d}: rows={info['rows']} "
                    f"margin.t={info['margin_t']:.0f} legend_h={info['legend_h']:.0f} "
                    f"title y[{tb[1]:.0f},{tb[3]:.0f}] x[{tb[0]:.0f},{tb[2]:.0f}] | "
                    f"legend y[{lb[1]:.0f},{lb[3]:.0f}] x[{lb[0]:.0f},{lb[2]:.0f}] | "
                    f"overlap x={ox:.0f}px y={oy:.0f}px ({oy / th:.0%} of title height) "
                    f"title_drawn_after_legend={info['title_after_legend']}")
            print(line)
            summary.append((key, with_template, w, info["rows"], round(oy / th, 2) if th else None, round(ox)))

print("\nRESULT (streamlit template): rows / fraction of title height covered by legend bbox / horizontal overlap px")
for key, tmpl, w, rows, frac, ox in summary:
    if tmpl:
        print(f"  {key:24s} w={w:4d} rows={rows} covered={frac} x_overlap={ox}")
print(f"\nhtml files in {OUT}")
