"""
Probe: which lines/branches of app/vue/tabs/tab_iv_dashboard.py are executed by the
two payloads of tests/integration/_iv_dashboard_render_driver.py (seeded + empty)?
Re-uses the driver's own _build_payload() under coverage.py, in a pristine process
with sockets blocked. Prints the missing lines of the view.
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

import coverage  # noqa: E402

VIEW = REPO / "app" / "vue" / "tabs" / "tab_iv_dashboard.py"
cov = coverage.Coverage(include=[str(VIEW)], branch=True)
cov.start()

# the driver parses sys.argv[1] at import time
sys.argv = [sys.argv[0], str(REPO)]
import _iv_dashboard_render_driver as drv  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_function(drv._tab_script, default_timeout=120)
payload = drv._build_payload()
at.session_state["iv_dashboard_result"] = payload
at.run()
print("seeded: exceptions=", [str(e.value) for e in at.exception],
      "charts=", len(at.get("plotly_chart")), "metrics=", len(at.metric),
      "warnings=", len(at.warning), "infos=", len(at.info), "errors=", len(at.error))
print("payload: reg_high is None?", payload["analysis"]["reg_high"] is None,
      "| reg_low is None?", payload["analysis"]["reg_low"] is None,
      "| n_high/n_low=", payload["analysis"]["n_high"], payload["analysis"]["n_low"],
      "| len(iv_history)=", len(payload["iv_history"]))

at2 = AppTest.from_function(drv._tab_script, default_timeout=120)
at2.run()
print("empty: exceptions=", [str(e.value) for e in at2.exception], "infos=", len(at2.info))

cov.stop()
import io  # noqa: E402

buf = io.StringIO()
cov.report(file=buf, show_missing=True)
print(buf.getvalue())
data = cov.get_data()
analysis = cov._analyze(str(VIEW))
print("missing lines:", sorted(analysis.missing))
print("missing branch arcs:", sorted(analysis.missing_branch_arcs().items()))
