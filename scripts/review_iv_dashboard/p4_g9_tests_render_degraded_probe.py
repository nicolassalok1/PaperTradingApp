"""G9 skeptic probe: does the tab actually survive the degraded states the render
guard never exercises? (If it crashes today, the finding is under-rated; if it
renders cleanly, the finding is a pure test-coverage gap.)

Run in a pristine interpreter (same protocol as the repo driver):
    python p4_g9_tests_render_degraded_probe.py <repo_root>
"""

from __future__ import annotations

import datetime as dt
import json
import socket
import sys
import time

repo_root = sys.argv[1]
sys.path.insert(0, repo_root)


def _blocked(*a, **k):  # noqa: ANN002, ANN003
    raise RuntimeError("network forbidden")


socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.create_connection = _blocked  # type: ignore[assignment]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

sys.path.insert(0, f"{repo_root}/tests/integration")
from _iv_dashboard_render_driver import _build_payload, _tab_script  # noqa: E402


def _summary(at) -> dict:
    return {
        "exc": [str(e.value) for e in at.exception],
        "charts": len(at.get("plotly_chart")),
        "metrics": len(at.metric),
        "warnings": [w.value for w in at.warning],
        "infos": len(at.info),
        "errors": [e.value for e in at.error],
    }


def run(payload, label):
    t0 = time.perf_counter()
    at = AppTest.from_function(_tab_script, default_timeout=120)
    if payload is not None:
        at.session_state["iv_dashboard_result"] = payload
    at.run()
    s = _summary(at)
    s["sec"] = round(time.perf_counter() - t0, 2)
    print(label, json.dumps(s, ensure_ascii=False))
    return at


# A. no-keys degraded payload as the service really produces it
p = _build_payload()
p.update(
    current_iv=None,
    iv_error="Snapshots filtrés indisponibles (Clés Alpaca absentes) ; fallback chaîne complète.",
    iv_regime=None,
    iv_minus_rv=None,
    iv_vs_series_percentile=float("nan"),
    iv_history=pd.DataFrame(columns=["date", "iv"]),
    analysis=None,
    analysis_error="Série insuffisante (test)",
)
run(p, "A.degraded_no_iv_no_analysis_empty_hist")

# B. IV checkbox unticked (service: current_iv None, iv_error None)
p = _build_payload()
p.update(current_iv=None, iv_error=None, iv_regime=None, iv_minus_rv=None,
         iv_vs_series_percentile=float("nan"))
run(p, "B.iv_disabled")

# C. analysis present but one regime regression missing (reg_high None, n_high=0)
p = _build_payload()
p["analysis"] = dict(p["analysis"])
p["analysis"]["reg_high"] = None
run(p, "C.reg_high_none")

# D. iv_history None (defensive branch) + current_iv with missing optional keys
p = _build_payload()
p["iv_history"] = None
p["current_iv"] = {"iv": 0.22}
run(p, "D.hist_none_sparse_iv")

# E. form submit, controller raising (module object shared with the AppTest script)
from app.controller import iv_dashboard_controller as ctrl  # noqa: E402

orig = ctrl.get_iv_analysis
ctrl.get_iv_analysis = lambda symbol, **kw: (_ for _ in ()).throw(RuntimeError("Alpaca HS (test)"))
at = AppTest.from_function(_tab_script, default_timeout=120)
at.run()
at.button[0].click().run()
print("E.submit_controller_raises", json.dumps(_summary(at), ensure_ascii=False))

# F. form submit, controller returning a payload -> state populated and charts drawn
ctrl.get_iv_analysis = lambda symbol, **kw: _build_payload() | {"symbol": symbol}
at = AppTest.from_function(_tab_script, default_timeout=120)
at.run()
at.text_input[0].set_value("qqq")
at.button[0].click().run()
s = _summary(at)
s["state_symbol"] = at.session_state["iv_dashboard_result"]["symbol"]
print("F.submit_ok", json.dumps(s, ensure_ascii=False))

# G. blank symbol
at = AppTest.from_function(_tab_script, default_timeout=120)
at.run()
at.text_input[0].set_value("   ")
at.button[0].click().run()
print("G.submit_blank", json.dumps(_summary(at), ensure_ascii=False))
ctrl.get_iv_analysis = orig
