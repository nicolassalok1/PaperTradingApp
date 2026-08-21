"""
p4 skeptic repro — finding `stale-result-under-error-and-params-drift`.

Own harness (real service payload via patched offline fetchers, real view):
 A. first analysis SPY / "2 ans" succeeds -> stored in session_state.
 B. second submit with symbol XYZ / "5 ans" where the controller raises:
    does the SPY dashboard stay rendered under the red error? What does the
    caption say? Does the word "ans"/years appear anywhere in the rendered text?
 C. one more plain rerun after the failed submit (e.g. any widget interaction):
    does the error survive, or is the stale SPY dashboard shown with no hint?
 D. successful resubmit with another symbol -> state replaced (control).

Run: .venv/Scripts/python.exe scripts/review_iv_dashboard/p4_stale-result-under-error-and-params-drift_repro.py
"""
from __future__ import annotations

import json
import socket
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _blocked(*a, **k):  # noqa: ANN002, ANN003
    raise RuntimeError("network forbidden in probe")


socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.create_connection = _blocked  # type: ignore[assignment]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="p4_stale_"))


def _synthetic_closes(symbol, years=2.0, extra_days=0):  # noqa: ANN001
    if symbol == "XYZ":
        return None, "none", [f"Barres alpaca indisponibles : symbole {symbol} inconnu"]
    rng = np.random.default_rng(11)
    n = int(years * 252) + extra_days
    rets = rng.normal(0.0003, 0.011, n)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame({"Date": idx, "Close": 100.0 * np.exp(np.cumsum(rets))}), "synthetic", []


def _script():
    # Patch the model layer only; controller + view are the real ones.
    # (AppTest re-executes this function's source in a fresh module: keep it self-contained.)
    import os
    import tempfile

    import numpy as np
    import pandas as pd

    from app.model.iv_dashboard import service as svc

    def _synthetic_closes(symbol, years=2.0, extra_days=0):
        if symbol == "XYZ":
            return None, "none", [f"Barres alpaca indisponibles : symbole {symbol} inconnu"]
        rng = np.random.default_rng(11)
        n = int(years * 252) + extra_days
        rets = rng.normal(0.0003, 0.011, n)
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
        return pd.DataFrame({"Date": idx, "Close": 100.0 * np.exp(np.cumsum(rets))}), "synthetic", []

    svc.fetch_daily_closes = _synthetic_closes
    svc.fetch_current_atm_iv = lambda sym: (None, ["IV disabled in probe."])
    from pathlib import Path as _P

    svc.CACHE_IV_HISTORY_DIR = _P(os.environ.get("P4_TMP", tempfile.gettempdir()))
    from app.vue.tabs import tab_iv_dashboard as tab

    tab.render_tab()


def _all_text(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "warning", "error", "info", "code", "metric"):
        for el in getattr(at, kind):
            v = getattr(el, "value", None)
            if v is None:
                v = f"{getattr(el, 'label', '')} {getattr(el, 'value', '')}"
            parts.append(str(v))
    return "\n".join(parts)


def snapshot(at: AppTest, tag: str):
    st = at.session_state["iv_dashboard_result"] if "iv_dashboard_result" in at.session_state else None
    return {
        "exceptions": [str(e.value) for e in at.exception],
        "errors": [e.value[:140] for e in at.error],
        "n_plotly_charts": len(at.get("plotly_chart")),
        "header_caption": next((c.value for c in at.caption if "source série" in c.value), None),
        "form_symbol_now": at.text_input[0].value,
        "form_duration_now": at.selectbox[0].value,
        "state_symbol": None if st is None else st.get("symbol"),
        "state_years": None if st is None else st.get("years"),
        "rendered_text_mentions_years": any(
            tok in _all_text(at) for tok in (" an ", " ans", "1 an", "2 ans", "3 ans", "5 ans", "Durée")
        ),
    }


def main():
    out = {}
    at = AppTest.from_function(_script, default_timeout=120)
    at.run()
    out["0_empty_state"] = {"n_info": len(at.info), "n_charts": len(at.get("plotly_chart"))}

    # A. first successful analysis SPY / 2 ans (default selectbox index=1)
    at.text_input[0].set_value("SPY")
    at.button[0].click()
    at.run()
    out["A_after_SPY_ok"] = snapshot(at, "A")

    # B. failed resubmit XYZ / 5 ans
    at.text_input[0].set_value("XYZ")
    at.selectbox[0].set_value("5 ans")
    at.button[0].click()
    at.run()
    out["B_after_XYZ_failed"] = snapshot(at, "B")

    # C. one more rerun without submitting (any widget interaction elsewhere)
    at.run()
    out["C_plain_rerun_after_failure"] = snapshot(at, "C")

    # D. control: successful resubmit with another symbol replaces the state
    at.text_input[0].set_value("QQQ")
    at.button[0].click()
    at.run()
    out["D_after_QQQ_ok"] = snapshot(at, "D")

    # generated_at format
    out["generated_at_format"] = at.session_state["iv_dashboard_result"]["generated_at"]
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
