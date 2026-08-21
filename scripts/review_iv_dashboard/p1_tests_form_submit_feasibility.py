"""
Probe: can the untested form-submit branch of tab_iv_dashboard (lines 150-168) and the
degraded payload branches (current_iv None / analysis None / empty iv_history) be
rendered through AppTest in the same subprocess pattern as the driver?  Controller is
monkeypatched in-process (module object is shared with the AppTest script), sockets
blocked. Prints what each run shows.
"""
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests" / "integration"))


def _blocked(*a, **k):
    raise RuntimeError("no network")


socket.socket.connect = _blocked
socket.create_connection = _blocked

import pandas as pd  # noqa: E402

sys.argv = [sys.argv[0], str(REPO)]
import _iv_dashboard_render_driver as drv  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402
from app.controller import iv_dashboard_controller as ctrl  # noqa: E402


def summary(at):
    return dict(
        exc=[str(e.value) for e in at.exception],
        charts=len(at.get("plotly_chart")), metrics=len(at.metric),
        warnings=[w.value[:60] for w in at.warning], infos=len(at.info),
        errors=[e.value[:60] for e in at.error], buttons=len(at.button),
    )


# ---- A. degraded payload: no IV, no analysis, empty history, reg_high None ----------
payload = drv._build_payload()
payload.update(
    current_iv=None, iv_error="chaîne d'options Alpaca inaccessible (test)",
    iv_regime=None, iv_minus_rv=None, iv_vs_series_percentile=float("nan"),
    iv_history=pd.DataFrame(columns=["date", "iv"]),
    analysis=None, analysis_error="Série insuffisante (test)",
)
at = AppTest.from_function(drv._tab_script, default_timeout=120)
at.session_state["iv_dashboard_result"] = payload
at.run()
print("[A degraded payload]", summary(at))

# ---- B. form submit -> controller raises -> st.error ----------------------------------
def _boom(symbol, **kw):
    raise RuntimeError("Alpaca HS (test)")

ctrl.get_iv_analysis = _boom
at = AppTest.from_function(drv._tab_script, default_timeout=120)
at.run()
print("[B before click] buttons:", len(at.button), "labels:", [b.label for b in at.button])
if at.button:
    at.button[0].click().run()
    print("[B after click, controller raises]", summary(at))

# ---- C. form submit -> controller returns payload -> full dashboard -------------------
good = drv._build_payload()
ctrl.get_iv_analysis = lambda symbol, **kw: dict(good, symbol=symbol, log=[f"kw={sorted(kw)}"])
at = AppTest.from_function(drv._tab_script, default_timeout=120)
at.run()
if at.button:
    at.text_input[0].set_value(" qqq ")
    at.button[0].click().run()
    s = summary(at)
    print("[C after click, controller ok]", s, "| symbol in state:", at.session_state["iv_dashboard_result"]["symbol"])

# ---- D. form submit with empty symbol -> warning ---------------------------------------
at = AppTest.from_function(drv._tab_script, default_timeout=120)
at.run()
if at.button:
    at.text_input[0].set_value("   ")
    at.button[0].click().run()
    print("[D empty symbol]", summary(at))
