"""
p1_view_render_charts.py — capture the 3 Plotly figures exactly as the view builds
them (st.plotly_chart monkeypatched), dump layout facts and write standalone HTML
files (plotly.js embedded, no CDN) into the scratchpad for a local screenshot.

Usage: python p1_view_render_charts.py <out_dir>
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
out_dir.mkdir(parents=True, exist_ok=True)

captured = {}


def _capture(fig, *a, **kw):
    captured[kw.get("key", f"fig{len(captured)}")] = fig


st.plotly_chart = _capture  # type: ignore[assignment]
st.markdown = lambda *a, **k: None  # type: ignore[assignment]
st.caption = lambda *a, **k: None  # type: ignore[assignment]
st.metric = lambda *a, **k: None  # type: ignore[assignment]
st.warning = lambda *a, **k: None  # type: ignore[assignment]

from app.model.iv_dashboard import analytics as ivx  # noqa: E402
from app.vue.tabs import tab_iv_dashboard as tab  # noqa: E402

# 5-year series (~1260 pts) with an old IV-history observation
rng = np.random.default_rng(7)
n = 1260 + 40
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
    "iv_history": pd.DataFrame({"date": pd.bdate_range(end="2026-08-21", periods=5), "iv": np.linspace(cur, cur + 0.04, 5)}),
}

import time  # noqa: E402

t0 = time.perf_counter()
tab._render_series_chart(result)
t1 = time.perf_counter()
tab._render_forward_chart(result)
tab._render_diff_chart(result)
t2 = time.perf_counter()

facts = {"build_ms_series": round((t1 - t0) * 1000, 1), "build_ms_fwd_diff": round((t2 - t1) * 1000, 1)}
for key, fig in captured.items():
    lay = fig.layout
    facts[key] = {
        "n_traces": len(fig.data),
        "legend_entries": [t.name for t in fig.data if t.showlegend is not False],
        "single_point_traces": [t.name for t in fig.data if t.x is not None and len(t.x) == 1],
        "title": lay.title.text,
        "margin_t": lay.margin.t,
        "legend": dict(orientation=lay.legend.orientation, y=lay.legend.y, yanchor=lay.legend.yanchor),
        "title_y": lay.title.y,
        "n_shapes": len(lay.shapes or ()),
        "height": lay.height,
    }
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117",
                      width=900 if key == "iv_dash_series_chart" else 560)
    (out_dir / f"{key}.html").write_text(fig.to_html(include_plotlyjs=True, full_html=True), encoding="utf-8")

print(json.dumps(facts, indent=1, ensure_ascii=False, default=str))
print("written:", [str(p) for p in out_dir.glob("*.html")])
