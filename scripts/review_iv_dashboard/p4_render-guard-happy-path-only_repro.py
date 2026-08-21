"""p4 skeptic repro: which view branches do the two driver runs exercise, and do the
untested degraded / submit paths actually render (or crash)?

Standalone pristine interpreter (same reason as the driver). Sockets blocked.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import sys

ROOT = r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca"
sys.path.insert(0, ROOT)
for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "OPENAI_API_KEY"):
    os.environ.pop(k, None)


def _blocked(*a, **k):  # noqa: ANN002, ANN003
    raise RuntimeError("network forbidden")


socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.create_connection = _blocked  # type: ignore[assignment]

import coverage  # noqa: E402

VIEW = os.path.join(ROOT, "app", "vue", "tabs", "tab_iv_dashboard.py")
cov = coverage.Coverage(branch=True, include=[VIEW], concurrency=["thread"])
cov.start()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

# reuse the real driver's payload builder and script so the measurement is faithful
sys.path.insert(0, os.path.join(ROOT, "tests", "integration"))
sys.argv = [sys.argv[0], ROOT]
import _iv_dashboard_render_driver as drv  # noqa: E402


def _summ(at):
    return {
        "exceptions": [str(e.value) for e in at.exception],
        "n_charts": len(at.get("plotly_chart")),
        "n_metrics": len(at.metric),
        "warnings": [w.value for w in at.warning],
        "infos": len(at.info),
        "errors": [e.value for e in at.error],
    }


# --- the two existing runs (exactly as the driver does)
at = AppTest.from_function(drv._tab_script, default_timeout=120)
p = drv._build_payload()
at.session_state["iv_dashboard_result"] = p
at.run()
seeded = _summ(at)
seeded["reg_high_present"] = p["analysis"]["reg_high"] is not None
seeded["reg_low_present"] = p["analysis"]["reg_low"] is not None
seeded["len_iv_history"] = len(p["iv_history"])
at2 = AppTest.from_function(drv._tab_script, default_timeout=120)
at2.run()
empty = _summ(at2)

cov.stop()
cov.save()
an = cov._analyze(VIEW)
missing_lines = sorted(an.missing)
missing_arcs = sorted(an.missing_branch_arcs().items())
total = len(an.statements)
print("SEEDED:", json.dumps(seeded))
print("EMPTY :", json.dumps(empty))
print(f"view coverage (2 driver runs): {total - len(missing_lines)}/{total} stmts = "
      f"{100*(total-len(missing_lines))/total:.1f}% ; missing lines={missing_lines}")
print("missing arcs:", [(a, b) for a, b in missing_arcs])

# --- degraded payload: what the service emits without keys + no analysis + no history
cov2 = coverage.Coverage(branch=True, include=[VIEW], concurrency=["thread"], data_file=None)
cov2.start()
pd_ = drv._build_payload()
pd_.update(
    current_iv=None,
    iv_error="chaîne d'options Alpaca inaccessible (test)",
    iv_regime=None,
    iv_minus_rv=None,
    iv_vs_series_percentile=float("nan"),
    iv_history=pd.DataFrame(columns=["date", "iv"]),
    analysis=None,
    analysis_error="Série insuffisante (test)",
)
at3 = AppTest.from_function(drv._tab_script, default_timeout=120)
at3.session_state["iv_dashboard_result"] = pd_
at3.run()
degraded = _summ(at3)
print("DEGRADED:", json.dumps(degraded, ensure_ascii=False))

# --- degraded variant closer to real service: iv_history None, current_iv present but reg_high None
pd2 = drv._build_payload()
pd2["iv_history"] = None
pd2["analysis"]["reg_high"] = None
pd2["analysis"]["reg_low"] = None
at3b = AppTest.from_function(drv._tab_script, default_timeout=120)
at3b.session_state["iv_dashboard_result"] = pd2
at3b.run()
degraded_b = _summ(at3b)
print("DEGRADED_B (reg None, iv_history None):", json.dumps(degraded_b, ensure_ascii=False))

# --- form submit with the controller failing
from app.controller import iv_dashboard_controller as ctrl  # noqa: E402

orig = ctrl.get_iv_analysis


def _boom(symbol, **kw):
    raise RuntimeError("Alpaca HS (test)")


ctrl.get_iv_analysis = _boom
at4 = AppTest.from_function(drv._tab_script, default_timeout=120)
at4.run()
print("buttons:", [b.label for b in at4.button])
at4.button[0].click().run()
submit_err = _summ(at4)
print("SUBMIT_ERROR:", json.dumps(submit_err, ensure_ascii=False))

# --- form submit with the controller returning a payload
calls = []


def _ok(symbol, **kw):
    calls.append((symbol, kw))
    return drv._build_payload() | {"symbol": symbol}


ctrl.get_iv_analysis = _ok
at5 = AppTest.from_function(drv._tab_script, default_timeout=120)
at5.run()
at5.text_input[0].set_value("qqq")
at5.button[0].click().run()
submit_ok = _summ(at5)
submit_ok["controller_calls"] = [(s, sorted(k)) for s, k in calls]
submit_ok["state_symbol"] = at5.session_state["iv_dashboard_result"]["symbol"]
print("SUBMIT_OK:", json.dumps(submit_ok, ensure_ascii=False))

# --- blank symbol
at6 = AppTest.from_function(drv._tab_script, default_timeout=120)
at6.run()
at6.text_input[0].set_value("   ")
at6.button[0].click().run()
print("SUBMIT_BLANK:", json.dumps(_summ(at6), ensure_ascii=False))
ctrl.get_iv_analysis = orig

cov2.stop()
an2 = cov2._analyze(VIEW)
print(f"view coverage (extra runs only): missing lines={sorted(an2.missing)}")
union_missing = sorted(set(missing_lines) & set(an2.missing))
print(f"still missing after all runs: {union_missing}")
