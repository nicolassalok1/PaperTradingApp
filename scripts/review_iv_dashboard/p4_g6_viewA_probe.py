"""p4 / G6_viewA — code-reading probe for three view findings.

A) iv-disabled-says-alpaca-inaccessible : service branch when include_current_iv=False
   + exact warning string produced by _render_metrics.
B) stale-result-under-error-and-params-drift : AppTest — seed a result, make the
   controller raise, re-submit, count charts / caption / state.
C) iv-history-overlay-unbounded-x-range : capture the series figure, compare the
   x-extent of the IV overlay with the RV series, check no xaxis.range is set.
No network: fetch_daily_closes / fetch_current_atm_iv are patched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

out: dict = {}


def _synthetic_result(years: float = 1.0, iv_dates=("2023-03-01", "2026-08-20")) -> dict:
    end = pd.Timestamp("2026-08-21")
    idx = pd.bdate_range(end=end, periods=int(252 * years))
    rng = np.random.default_rng(0)
    vol = pd.Series(0.15 + 0.05 * rng.standard_normal(len(idx)).cumsum() / 10, index=idx).abs() + 0.05
    series = pd.DataFrame({"close": 100.0, "vol": vol, "vol_percentile": 0.5}, index=idx)
    iv_hist = pd.DataFrame({"date": pd.to_datetime(list(iv_dates)), "iv": [0.2, 0.22]})
    return {
        "symbol": "SPY", "source": "alpaca", "years": years, "rv_window": 20,
        "forward_window": 30, "percentile_window": 252, "series": series,
        "current_vol": float(vol.iloc[-1]), "current_percentile": 0.5,
        "regime": {"key": "normal", "label": "Normal", "signal_key": "neutral", "signal_label": "Neutre"},
        "vol_stats": {"min": 0.1, "mean": 0.2, "max": 0.3},
        "current_iv": None, "iv_error": None, "iv_vs_series_percentile": float("nan"),
        "iv_regime": None, "iv_minus_rv": None, "iv_history": iv_hist,
        "analysis": None, "analysis_error": "skip", "log": [], "generated_at": "12:00:00",
    }


# ----------------------------------------------------------------------------- A
from app.model.iv_dashboard import service as svc  # noqa: E402

closes = pd.DataFrame({
    "Date": pd.bdate_range(end="2026-08-21", periods=600),
    "Close": 100 + np.cumsum(np.random.default_rng(1).standard_normal(600)),
})
with mock.patch.object(svc, "fetch_daily_closes", return_value=(closes, "alpaca", ["stub"])), \
     mock.patch.object(svc, "fetch_current_atm_iv", side_effect=AssertionError("must not be called")), \
     mock.patch.object(svc, "load_iv_history", return_value=pd.DataFrame(columns=["date", "iv"])):
    payload = svc.get_iv_dashboard_data("SPY", years=1.0, include_current_iv=False)
out["A_service_disabled"] = {
    "current_iv": payload["current_iv"], "iv_error": payload["iv_error"],
    "payload_has_include_flag": "include_current_iv" in payload,
    "series_index_tz": str(payload["series"].index.tz),
}

from app.vue.tabs import tab_iv_dashboard as tab  # noqa: E402

captured_warn: list[str] = []
with mock.patch.object(tab.st, "warning", side_effect=lambda m, *a, **k: captured_warn.append(str(m))):
    tab._render_metrics(_synthetic_result())
out["A_view_warning_when_disabled"] = captured_warn

# ----------------------------------------------------------------------------- C
captured_fig: list = []
with mock.patch.object(tab.st, "plotly_chart", side_effect=lambda fig, *a, **k: captured_fig.append(fig)):
    tab._render_series_chart(_synthetic_result(years=1.0))
fig = captured_fig[0]
rv_trace = next(t for t in fig.data if t.name.startswith("Vol réalisée"))
iv_trace = next(t for t in fig.data if t.name == "IV ATM (historique local)")
rv_x = pd.to_datetime(np.asarray(rv_trace.x))
iv_x = pd.to_datetime(np.asarray(iv_trace.x))
all_x = rv_x.append(iv_x)
span_all = (all_x.max() - all_x.min()).days
span_rv = (rv_x.max() - rv_x.min()).days
out["C_overlay"] = {
    "iv_mode": iv_trace.mode,
    "xaxis_range_set": fig.layout.xaxis.range is not None,
    "xaxis_autorange": fig.layout.xaxis.autorange,
    "rv_start": str(rv_x.min().date()), "iv_min": str(iv_x.min().date()),
    "span_rv_days": span_rv, "span_all_days": span_all,
    "rv_share_of_axis_pct": round(100 * span_rv / span_all, 1),
    "fix_filter_comparison_ok": bool((_synthetic_result()["iv_history"]["date"] >= rv_x[0]).any()),
}

# ----------------------------------------------------------------------------- B
from streamlit.testing.v1 import AppTest  # noqa: E402


def _app():
    import streamlit as st
    from app.vue.tabs import tab_iv_dashboard as t
    if "seed" not in st.session_state:
        from scripts.review_iv_dashboard.p4_g6_viewA_probe import _synthetic_result  # noqa
    t.render_tab()


def _script():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import streamlit as st
    from unittest import mock
    from app.vue.tabs import tab_iv_dashboard as t
    from app.controller import iv_dashboard_controller as c
    import numpy as np, pandas as pd
    if "iv_dashboard_result" not in st.session_state:
        idx = pd.bdate_range(end="2026-08-21", periods=252)
        vol = pd.Series(0.2, index=idx)
        st.session_state["iv_dashboard_result"] = {
            "symbol": "SPY", "source": "alpaca", "years": 2.0, "rv_window": 20,
            "forward_window": 30, "percentile_window": 252,
            "series": pd.DataFrame({"close": 100.0, "vol": vol, "vol_percentile": 0.5}, index=idx),
            "current_vol": 0.2, "current_percentile": 0.5,
            "regime": {"key": "normal", "label": "Normal", "signal_key": "neutral", "signal_label": "Neutre"},
            "vol_stats": {"min": 0.1, "mean": 0.2, "max": 0.3},
            "current_iv": None, "iv_error": "x", "iv_vs_series_percentile": float("nan"),
            "iv_regime": None, "iv_minus_rv": None, "iv_history": None,
            "analysis": None, "analysis_error": "skip", "log": [], "generated_at": "12:00:00",
        }
    with mock.patch.object(c, "get_iv_analysis", side_effect=RuntimeError("boom")):
        t.render_tab()


at = AppTest.from_function(_script, default_timeout=60)
at.run()
n_charts_before = len([e for e in at.main if getattr(e, "type", "") == "plotly_chart"])
at.text_input[0].set_value("XYZ")
at.selectbox[0].set_value("5 ans")
at.button[0].click()
at.run()
out["B_stale_after_error"] = {
    "exceptions": [str(e) for e in at.exception],
    "errors": [e.value for e in at.error],
    "captions": [c.value for c in at.caption],
    "selectbox_now": at.selectbox[0].value,
    "state_symbol": at.session_state["iv_dashboard_result"]["symbol"],
    "state_years": at.session_state["iv_dashboard_result"]["years"],
    "result_still_rendered": any("**SPY**" in c.value for c in at.caption),
    "years_shown_anywhere": any("2 an" in (m.value or "") for m in list(at.markdown) + list(at.caption)),
}

print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
