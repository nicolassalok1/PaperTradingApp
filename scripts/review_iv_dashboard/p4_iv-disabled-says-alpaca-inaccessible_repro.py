"""
p4 skeptic repro — finding `iv-disabled-says-alpaca-inaccessible`.

Independent of the finder's probe: goes through the REAL service
(`get_iv_dashboard_data`) with `include_current_iv=False`, offline (sockets
blocked, `fetch_daily_closes` patched with synthetic closes, IV-history cache
redirected to a temp dir), asserts that `fetch_current_atm_iv` is never called,
then renders the real payload through the real view with AppTest and dumps the
warning text the user sees.

Run: .venv/Scripts/python.exe scripts/review_iv_dashboard/p4_iv-disabled-says-alpaca-inaccessible_repro.py
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

TMP = Path(tempfile.mkdtemp(prefix="p4_ivdis_"))


def _synthetic_closes(symbol, years=2.0, extra_days=0):  # noqa: ANN001
    rng = np.random.default_rng(11)
    n = int(years * 252) + extra_days
    rets = rng.normal(0.0003, 0.011, n)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    df = pd.DataFrame({"Date": idx, "Close": 100.0 * np.exp(np.cumsum(rets))})
    return df, "synthetic-offline", ["[probe] closes synthetiques"]


def build_real_payload(include_iv: bool):
    from app.model.iv_dashboard import service as svc

    calls = {"atm_iv": 0}

    def _spy_atm(sym):  # noqa: ANN001
        calls["atm_iv"] += 1
        raise AssertionError("fetch_current_atm_iv must not be called when include_current_iv=False")

    svc.fetch_daily_closes = _synthetic_closes
    svc.fetch_current_atm_iv = _spy_atm
    svc.CACHE_IV_HISTORY_DIR = TMP
    payload = svc.get_iv_dashboard_data(
        "SPY", years=2.0, rv_window=20, forward_window=30, percentile_window=252,
        include_current_iv=include_iv,
    )
    return payload, calls


def _script():
    from app.vue.tabs import tab_iv_dashboard as tab

    tab.render_tab()


def main():
    out = {}
    payload, calls = build_real_payload(include_iv=False)
    payload["generated_at"] = "12:00:00"
    out["service"] = {
        "fetch_current_atm_iv_calls": calls["atm_iv"],
        "current_iv": payload["current_iv"],
        "iv_error": payload["iv_error"],
        "log_mentions_iv": [m for m in payload["log"] if "IV" in m],
    }

    at = AppTest.from_function(_script, default_timeout=120)
    at.session_state["iv_dashboard_result"] = payload
    at.run()
    out["view"] = {
        "exceptions": [str(e.value) for e in at.exception],
        "warnings": [w.value for w in at.warning],
        "infos": [i.value[:80] for i in at.info],
        "captions": [c.value[:100] for c in at.caption],
    }
    # contrast: same path with include_iv=True but the chain call failing (real error string)
    from app.model.iv_dashboard import service as svc

    def _failing_atm(sym):  # noqa: ANN001
        return None, ["Chaîne d'options Alpaca indisponible : 401 Unauthorized."]

    svc.fetch_current_atm_iv = _failing_atm
    p2 = svc.get_iv_dashboard_data("SPY", include_current_iv=True)
    p2["generated_at"] = "12:00:00"
    at2 = AppTest.from_function(_script, default_timeout=120)
    at2.session_state["iv_dashboard_result"] = p2
    at2.run()
    out["view_when_chain_really_fails"] = {
        "iv_error": p2["iv_error"],
        "warnings": [w.value for w in at2.warning],
    }
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
