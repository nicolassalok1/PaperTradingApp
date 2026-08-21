"""Probe: controller clamp drift vs view bounds, cache path derivation, write location.

Offline: the service orchestrator is stubbed so nothing touches the network.
"""
import os
import re
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
    os.environ.pop(k, None)


def _blocked(*a, **k):
    raise RuntimeError("network blocked by probe")


socket.socket.connect = _blocked  # type: ignore[assignment]
socket.create_connection = _blocked  # type: ignore[assignment]

from app.controller import iv_dashboard_controller as ctrl  # noqa: E402
from app.model.iv_dashboard import service as svc  # noqa: E402

# ---- 1. controller clamp: what does the model actually receive? ----------------
captured = {}


def _fake(sym, **kw):
    captured.update(kw)
    return {"symbol": sym, **kw}


svc.get_iv_dashboard_data = _fake  # type: ignore[assignment]
ctrl.get_iv_analysis("spy", years=0.1, rv_window=2, forward_window=500, percentile_window=30)
print("controller forwarded (silently clamped):", captured)

# ---- 2. view widget bounds vs controller clamp bounds (static) ----------------
view_src = (ROOT / "app/vue/tabs/tab_iv_dashboard.py").read_text(encoding="utf-8")
ctrl_src = (ROOT / "app/controller/iv_dashboard_controller.py").read_text(encoding="utf-8")
view_bounds = re.findall(r'"([^"]+)",\s*min_value=(\d+),\s*max_value=(\d+)', view_src, flags=re.S)
view_bounds += re.findall(r'"([^"]+)",\s*\n\s*min_value=(\d+),\s*\n\s*max_value=(\d+)', view_src)
ctrl_bounds = re.findall(r"(\w+)=_clamp_(?:int|float)\(\w+, [\d.]+, ([\d.]+), ([\d.]+)\)", ctrl_src)
print("view widget bounds :", view_bounds)
print("controller bounds  :", ctrl_bounds)
print("view duration choices:", re.findall(r'_DURATION_CHOICES = (\{.*?\})', view_src))

# ---- 3. cache path derivation from the raw symbol ------------------------------
for sym in ("spy", "BRK.B", "A/B", "../evil", "SPY "):
    p = svc._iv_history_path(sym)
    try:
        rel = p.resolve().relative_to(svc.CACHE_IV_HISTORY_DIR.resolve())
        inside = True
    except ValueError:
        rel, inside = p.resolve(), False
    print(f"symbol {sym!r:12} -> {p.name!r:28} parent={p.parent.name!r:14} inside IVHistory={inside} rel={rel}")

# ---- 4. write location: redirect CACHE_IV_HISTORY_DIR to a temp dir -------------
tmp = Path(tempfile.mkdtemp(prefix="ivh_"))
svc.CACHE_IV_HISTORY_DIR = tmp
info = {"iv": 0.2, "dte": 30, "n_contracts": 4, "method": "x", "spot": 100.0}
svc.record_iv_observation("A/B", info)
svc.record_iv_observation("SPY", info)
svc.record_iv_observation("SPY", {**info, "iv": 0.25})  # same-day upsert
written = sorted(str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file())
print("files written under temp IVHistory:", written)
print("SPY history rows after 2 same-day upserts:", len(svc.load_iv_history("SPY")), "iv=", svc.load_iv_history("SPY")["iv"].tolist())
