"""p4 repro — are the parameter bounds duplicated between view and controller, and
does the controller clamp silently (no log / no signal back to the caller)?

Offline: the model service function is monkeypatched to capture the kwargs the
controller forwards; nothing hits the network.
"""
from __future__ import annotations

import ast
import io
import logging
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.controller import iv_dashboard_controller as ctrl  # noqa: E402

# ---- 1) dynamic: capture what the controller forwards to the model ----------
captured = {}


def _fake_model(sym, **kw):
    captured.clear()
    captured.update(kw)
    return {"symbol": sym, **kw}


ctrl._svc.get_iv_dashboard_data = _fake_model  # type: ignore[attr-defined]

log_buf = io.StringIO()
handler = logging.StreamHandler(log_buf)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.DEBUG)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out = ctrl.get_iv_analysis(
        "spy", years=0.1, rv_window=2, forward_window=500, percentile_window=30
    )
    out2 = ctrl.get_iv_analysis(
        "spy", years=99, rv_window=10_000, forward_window=0, percentile_window=-5
    )
    out3 = ctrl.get_iv_analysis("spy", years="abc", rv_window=None, forward_window="x")

print("[clamp] low inputs  (0.1, 2, 500, 30)  ->", {k: captured_v for k, captured_v in out.items() if k != 'symbol'})
print("[clamp] high inputs (99, 1e4, 0, -5)   ->", {k: v for k, v in out2.items() if k != 'symbol'})
print("[clamp] junk inputs ('abc', None, 'x') ->", {k: v for k, v in out3.items() if k != 'symbol'})
print("[signal] warnings emitted:", len(caught), "| log lines:", repr(log_buf.getvalue()))
print("[signal] payload carries any 'clamped'/'warnings' key:",
      any(k in out for k in ("clamped", "warnings", "adjusted", "notes")))
print("[signal] controller imports logging/warnings:",
      bool(re.search(r"^\s*import (logging|warnings)", (ROOT / 'app/controller/iv_dashboard_controller.py').read_text(encoding='utf8'), re.M)))

# ---- 2) static: extract bounds from both files ------------------------------
view_src = (ROOT / "app/vue/tabs/tab_iv_dashboard.py").read_text(encoding="utf8")
ctrl_src = (ROOT / "app/controller/iv_dashboard_controller.py").read_text(encoding="utf8")

view_bounds = {}
for m in re.finditer(
    r"(\w+)\s*=\s*st\.number_input\(\s*\"([^\"]+)\",\s*min_value=(\d+),\s*max_value=(\d+)",
    view_src, re.S,
):
    view_bounds[m.group(1)] = (int(m.group(3)), int(m.group(4)))
dur = dict(re.findall(r"\"(\d) ans?\":\s*([\d.]+)", view_src))
view_bounds["years"] = (min(map(float, dur.values())), max(map(float, dur.values())))

ctrl_bounds = {}
for m in re.finditer(r"(\w+)=_clamp_(?:int|float)\(\w+,\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)", ctrl_src):
    ctrl_bounds[m.group(1)] = (float(m.group(3)), float(m.group(4)))

print("[static] view bounds      :", view_bounds)
print("[static] controller bounds:", ctrl_bounds)

# Is there a shared constant? (any name appearing in both files other than function names)
shared = re.findall(r"ctrl\.([A-Z_]+)", view_src)
print("[static] view reads an UPPER_CASE constant from ctrl:", shared or "NONE")

# ---- 3) can the clamp ever fire through the real UI? -----------------------
fires = {}
for k, (lo, hi) in ctrl_bounds.items():
    vlo, vhi = view_bounds.get(k, (None, None))
    fires[k] = not (vlo is not None and lo <= vlo and vhi <= hi)
print("[ui] clamp reachable from the view's own widget ranges:", fires)

# ---- 4) does the view display the clamped (payload) values or the widget's? -
labels = re.findall(r"result\.get\('(\w+_window)'\)", view_src)
print("[ui] labels read window sizes from payload keys:", sorted(set(labels)))
