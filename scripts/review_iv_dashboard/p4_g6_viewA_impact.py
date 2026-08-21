"""
p4_g6_viewA_impact.py — skeptic probe (impact & severity) for G6_viewA:
  - iv-disabled-says-alpaca-inaccessible
  - stale-result-under-error-and-params-drift
  - iv-history-overlay-unbounded-x-range

No network (sockets blocked). Service is driven with patched fetchers.
Run: .venv/Scripts/python.exe scripts/review_iv_dashboard/p4_g6_viewA_impact.py
"""
from __future__ import annotations

import json
import re
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _blocked(*a, **k):  # noqa: ANN002, ANN003
    raise RuntimeError("network forbidden in probe")


socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.create_connection = _blocked  # type: ignore[assignment]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.model.iv_dashboard import service as svc  # noqa: E402

out: dict = {}

# ---------------------------------------------------------------- A. iv_error reachability
rng = np.random.default_rng(1)
n = 560
closes = pd.DataFrame({
    "Date": pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n),
    "Close": 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))),
})
svc.fetch_daily_closes = lambda sym, years, extra_days: (closes.copy(), "alpaca", ["stub bars"])  # type: ignore
svc.record_iv_observation = lambda *a, **k: None  # type: ignore
svc.load_iv_history = lambda sym: pd.DataFrame(columns=["date", "iv"])  # type: ignore

cases = {}
for label, ret, inc in (
    ("include_iv=False", (None, ["should not be called"]), False),
    ("include_iv=True, no log", (None, []), True),
    ("include_iv=True, log", (None, ["Alpaca 401 - cles invalides"]), True),
):
    svc.fetch_current_atm_iv = lambda sym, _r=ret: _r  # type: ignore
    res = svc.get_iv_dashboard_data("SPY", years=2.0, include_current_iv=inc)
    cases[label] = {"current_iv": res["current_iv"], "iv_error": res["iv_error"]}
out["A_iv_error_by_case"] = cases
# The view's fallback string is used only when iv_error is falsy -> only the disabled case.

# ---------------------------------------------------------------- B. which result params are displayed by the view
view_src = (REPO / "app/vue/tabs/tab_iv_dashboard.py").read_text(encoding="utf-8")
out["B_params_surfaced_in_view"] = {
    "rv_window": "result.get('rv_window')" in view_src,
    "percentile_window": "result.get('percentile_window')" in view_src,
    "forward_window": 'analysis["forward_window"]' in view_src,
    "years": "result.get('years')" in view_src or 'result["years"]' in view_src,
    "symbol_in_caption": "**{result.get('symbol')}**" in view_src,
    "series_x_axis_is_dates": "x=series.index" in view_src,
    "generated_at_fmt": re.search(r'strftime\("([^"]+)"\)', view_src).group(1),
    "state_pop_on_error": "session_state.pop" in view_src,
}

# ---------------------------------------------------------------- C. IV overlay x-range: when does it matter?
today = pd.Timestamp.today().normalize()
for years in (1.0, 2.0):
    start = today - pd.Timedelta(days=int(years * 365.25))
    span = (today - start).days
    for used_days in (90, 365, 730, 1095):
        obs = pd.date_range(end=today, periods=max(2, used_days // 30), freq="30D")
        x_min = min(start, obs.min())
        share = span / (today - x_min).days
        out[f"C_years={years:g}_used={used_days}d"] = {
            "oldest_obs_days_ago": int((today - obs.min()).days),
            "rv_share_of_x_axis": round(share, 3),
            "x_axis_extended": bool(obs.min() < start),
        }

out["C_sparse_segments_note"] = (
    "mode='lines+markers' joins consecutive analysis days; with a 45-day gap one straight gold "
    "segment spans 45 d. Caption L257-260 already states 'une par jour d'analyse'."
)

# ---------------------------------------------------------------- D. real cache state
cache_dir = svc.CACHE_IV_HISTORY_DIR
files = sorted(p.name for p in cache_dir.glob("*.csv")) if cache_dir.exists() else []
out["D_real_cache_files"] = files

print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
