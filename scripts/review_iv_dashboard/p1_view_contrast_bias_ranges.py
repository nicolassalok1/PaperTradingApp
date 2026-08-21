"""
p1_view_contrast_bias_ranges.py — §4.4 View probes (pure python, no Streamlit run).

A. WCAG contrast of the hard-coded palette on the forced dark bg (#0E1117) and on a
   light bg (#FFFFFF) in case the viewer switches theme from the settings menu.
B. Structural bias of the "Signal (IV)" chip: percentile of IV within the RV
   distribution for typical variance-risk-premium spreads on a constant-vol GBM.
C. Series chart x-range pollution: iv_history older than the requested window.
D. Diff chart: intersection far outside the data range when slope ~ 1
   (smooth RV: rv_window=120, forward_window=5) -> vline/annotation position.
E. Dummy single-point legend traces: do they change the autoscale y-range?
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ---------------------------------------------------------------- A. contrast
def _lum(hexcol: str) -> float:
    h = hexcol.lstrip("#")
    rgb = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg: str, bg: str) -> float:
    l1, l2 = _lum(fg), _lum(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


palette = {
    "_COL_RED": "#f87171", "_COL_ORANGE": "#fb923c", "_COL_GREEN": "#34d399",
    "_COL_BLUE": "#60a5fa", "_COL_GOLD": "#fbbf24", "_COL_NEUTRAL": "#e5e7eb",
    "_COL_MUTED": "#9ca3af",
}
print("A. WCAG contrast (AA normal text >= 4.5, large/bold >= 3.0)")
for name, col in palette.items():
    print(f"  {name:13s} {col}  on #0E1117: {contrast(col, '#0E1117'):5.2f}   on #FFFFFF: {contrast(col, '#FFFFFF'):5.2f}")
print(f"  chip caption 0.78rem (~12.5px) #9ca3af on #0E1117: {contrast('#9ca3af', '#0E1117'):.2f} (needs 4.5)")

# ---------------------------------------------------------------- B. IV signal bias
from app.model.iv_dashboard import analytics as ivx  # noqa: E402

print("\nB. 'Signal (IV)' percentile_within(trailing RV, IV) on a constant-vol GBM (sigma=18%)")
rng = np.random.default_rng(11)
n = 520
closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.18 / np.sqrt(252), n))),
                   index=pd.bdate_range("2024-01-01", periods=n))
rv = ivx.compute_realized_vol(closes, 20).dropna()
trailing = rv.tail(252)
print(f"  RV 20d trailing-252 distribution: p10={trailing.quantile(.1):.3f} median={trailing.median():.3f} p90={trailing.quantile(.9):.3f}")
for spread_pts in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0):
    iv = float(trailing.median()) + spread_pts / 100.0
    p = ivx.percentile_within(trailing, iv)
    reg = ivx.classify_regime(p)
    print(f"  IV = median RV + {spread_pts:>3.0f} pts -> pct {p:5.3f} -> chip '{reg['label']}' / '{reg['signal_label']}'")

# ---------------------------------------------------------------- C. iv_history x-range
print("\nC. Series chart x-range with an old iv_history observation")
series_idx = pd.bdate_range(end=pd.Timestamp("2026-08-21"), periods=252)  # '1 an'
iv_hist = pd.DataFrame({"date": [pd.Timestamp("2023-03-01"), pd.Timestamp("2026-08-21")], "iv": [0.2, 0.19]})
x_min = min(series_idx[0], iv_hist["date"].min())
print(f"  series spans {series_idx[0].date()} -> {series_idx[-1].date()} ({(series_idx[-1]-series_idx[0]).days} days)")
print(f"  with overlay autoscale x starts {x_min.date()} -> span {(series_idx[-1]-x_min).days} days "
      f"(RV series occupies {(series_idx[-1]-series_idx[0]).days/(series_idx[-1]-x_min).days:.0%} of the axis)")
print("  view L314-326 plots iv_history unfiltered; service.load_iv_history has no date cutoff")

# ---------------------------------------------------------------- D. intersection scale
print("\nD. Diff chart vline position when forward ~ current (rv_window=120, forward_window=5)")
for seed in range(5):
    rng = np.random.default_rng(100 + seed)
    closes = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.18 / np.sqrt(252), 900))),
                       index=pd.bdate_range("2023-01-01", periods=900))
    rv = ivx.compute_realized_vol(closes, 120).dropna()
    a = ivx.analyze_forward_vol(rv, forward_window=5)
    lo, hi = float(a["df"]["current_vol"].min()), float(a["df"]["current_vol"].max())
    inter = a["intersection"]
    inside = lo <= inter <= hi
    print(f"  seed {seed}: slope={a['reg_forward']['slope']:.4f} data x in [{lo:.3f},{hi:.3f}] "
          f"intersection={inter:.3f} inside={inside} n_high={a['n_high']} n_low={a['n_low']}")

# ---------------------------------------------------------------- E. dummy traces
print("\nE. Dummy legend points vs autoscale")
vol = rv
q25, q75, mean = vol.quantile(.25), vol.quantile(.75), vol.mean()
print(f"  vol range [{vol.min():.4f},{vol.max():.4f}]; dummy y values q25={q25:.4f} q75={q75:.4f} mean={mean:.4f}"
      f" all inside range: {all(vol.min() <= v <= vol.max() for v in (q25, q75, mean))}")
import plotly.graph_objects as go  # noqa: E402
fig = go.Figure()
fig.add_trace(go.Scatter(x=[vol.index[0]], y=[float(q75)], mode="lines", hoverinfo="skip", name="75e"))
tr = fig.data[0]
print(f"  single-point mode='lines' trace: n points={len(tr.x)} hoverinfo={tr.hoverinfo} -> draws no segment, no hover")
