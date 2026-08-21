"""
p4_split-annotation-units-mismatch_repro.py — skeptic repro (offline, deterministic).

Build the diff / forward figures through the view's own functions (st.plotly_chart
monkeypatched) on a synthetic series and read back, from the figure objects:
  - x-axis tickformat
  - the vline annotation text
  - the forward-chart title
and the journal lines emitted by _render_log (st.code monkeypatched).
Also checks what the legacy Tkinter script did (axes + label both in decimal).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

captured: dict = {}
code_blocks: list = []
st.plotly_chart = lambda fig, *a, **k: captured.__setitem__(k.get("key", "?"), fig)  # type: ignore[assignment]
st.code = lambda body, *a, **k: code_blocks.append(body)  # type: ignore[assignment]


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


st.expander = lambda *a, **k: _Ctx()  # type: ignore[assignment]

from app.model.iv_dashboard import analytics as ivx  # noqa: E402
from app.vue.tabs import tab_iv_dashboard as tab  # noqa: E402

rng = np.random.default_rng(7)
n = 760
closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, n))),
                   index=pd.bdate_range(end="2026-08-21", periods=n))
rv = ivx.compute_realized_vol(closes, 20)
pct = ivx.compute_percentile_series(rv, 252)
series = pd.DataFrame({"close": closes, "vol": rv, "vol_percentile": pct}).dropna(subset=["vol"])
analysis = ivx.analyze_forward_vol(series["vol"], forward_window=30, percentile=series["vol_percentile"])
result = {"symbol": "SPY", "rv_window": 20, "series": series, "analysis": analysis, "log": []}

tab._render_forward_chart(result)
tab._render_diff_chart(result)
tab._render_log(result)

diff = captured["iv_dash_diff_chart"]
fwd = captured["iv_dash_forward_chart"]

ann = [a.text for a in (diff.layout.annotations or ())]
xticks = [t.text for t in ()]  # placeholder (ticks are computed client-side)
print(f"intersection (model)            = {analysis['intersection']:.6f}")
print(f"diff chart xaxis.tickformat      = {diff.layout.xaxis.tickformat!r}")
print(f"diff chart yaxis.tickformat      = {diff.layout.yaxis.tickformat!r}")
print(f"diff chart annotations           = {ann}")
print(f"diff chart hovertemplate (x)     = {diff.data[0].hovertemplate!r}")
print(f"forward chart title              = {fwd.layout.title.text!r}")
print(f"forward chart xaxis.tickformat   = {fwd.layout.xaxis.tickformat!r}")
print("journal lines mentioning the intersection:")
for line in code_blocks[0].splitlines():
    if "intersection" in line or "VOL HAUTE" in line or "VOL BASSE" in line:
        print("   ", line)

# What the tick labels around the vline will read (simulated d3 '.0%' on a 5% grid)
x = float(analysis["intersection"])
lo, hi = np.floor(x * 20) / 20, np.ceil(x * 20) / 20
print(f"vline sits between tick labels {lo:.0%} and {hi:.0%}, annotated '{ann[0]}'")

# Mixed-unit check, mechanical
mixed = (diff.layout.xaxis.tickformat or "").endswith("%") and not any("%" in a for a in ann)
print(f"\nRESULT: x-axis in percent while annotation is decimal -> {mixed}")

legacy = Path(r"C:/Users/Nathalie Asus/Downloads/option_trading_dashboard.py")
if legacy.exists():
    src = legacy.read_text(encoding="utf-8", errors="replace")
    pct_fmt = bool(re.search(r"PercentFormatter|FuncFormatter", src))
    lab = re.search(r"label=f'Regime Split \(Vol=\{intersection_x:\.3f\}\)'", src)
    print(f"legacy: axes use a percent formatter = {pct_fmt}; split label decimal .3f = {bool(lab)}"
          " -> legacy was internally consistent (both decimal); the port changed only the axes.")
