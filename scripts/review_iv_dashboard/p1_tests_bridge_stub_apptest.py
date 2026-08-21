"""
Probe: does importing controller_bridge in-process really break a later AppTest?
Verifies the rationale in tests/integration/test_iv_dashboard_render.py docstring.
Run with the venv interpreter; no network.
"""
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _blocked(*a, **k):
    raise RuntimeError("no network")


socket.socket.connect = _blocked
socket.create_connection = _blocked

import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402


def _script():
    import streamlit as st
    c1, c2 = st.columns(2)
    with c1:
        st.metric("a", "1")
    with c2:
        st.info("hello")
    st.session_state["k"] = 1
    st.markdown("x")


def run(label):
    at = AppTest.from_function(_script, default_timeout=60)
    at.run()
    print(f"[{label}] exceptions={[str(e.value) for e in at.exception]} "
          f"n_metric={len(at.metric)} n_info={len(at.info)} "
          f"n_markdown={len(at.markdown)} stubbed={getattr(st, '_codex_fake_streamlit', False)}")


run("before bridge import")
import app.vue.components.options.controller_bridge  # noqa: E402,F401
print("columns is lambda:", type(st.columns).__name__, "| session_state type:", type(st.session_state).__name__)
try:
    run("after bridge import")
except Exception as exc:  # noqa: BLE001
    print("[after bridge import] RAISED:", type(exc).__name__, exc)
