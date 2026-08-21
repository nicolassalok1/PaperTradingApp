"""Probe: import ordering / cycles / Streamlit leakage for the iv_dashboard change set.

Runs each import order in a fresh subprocess (offline, no network), then checks
whether importing the model/controller pulls streamlit / app.vue into sys.modules.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable

CASES = {
    "service_then_logic": "import app.model.iv_dashboard.service; import app.model.options.logic",
    "logic_then_service": "import app.model.options.logic; import app.model.iv_dashboard.service",
    "analytics_alone": "import app.model.iv_dashboard.analytics",
    "controller_alone": "import app.controller.iv_dashboard_controller",
    "market_data_then_service": "import app.model.market_data.market_data; import app.model.iv_dashboard.service",
    "view_alone": "import app.vue.tabs.tab_iv_dashboard",
}

SNIPPET = (
    "import sys; {code}; "
    "print('streamlit' in sys.modules, 'app.vue' in sys.modules, "
    "any(m.startswith('app.controller') for m in sys.modules))"
)

env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT)
env.pop("APCA_API_KEY_ID", None)
env.pop("APCA_API_SECRET_KEY", None)

for name, code in CASES.items():
    r = subprocess.run(
        [PY, "-c", SNIPPET.format(code=code)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    status = "OK" if r.returncode == 0 else "FAIL"
    tail = r.stdout.strip() or r.stderr.strip().splitlines()[-1:]
    print(f"{name:28s} {status} -> (streamlit, app.vue, app.controller*) loaded = {tail}")
