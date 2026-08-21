"""
Subprocess driver for `test_iv_dashboard_render.py` (not collected by pytest).

Renders the 🌡️ Vol Implicite tab twice via AppTest in this pristine interpreter
(seeded payload, then empty state) and prints a single JSON result line.
Outbound sockets are blocked before any app import.
"""

from __future__ import annotations

import datetime as dt
import json
import socket
import sys

repo_root = sys.argv[1]
sys.path.insert(0, repo_root)

RESULT_MARKER = "IVDASH_RESULT "


def _blocked(*args, **kwargs):  # noqa: ANN002, ANN003
    raise RuntimeError("network access is forbidden in the render guard")


socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.create_connection = _blocked  # type: ignore[assignment]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402


def _build_payload():
    from app.model.iv_dashboard import analytics as ivx

    rng = np.random.default_rng(5)
    n = 560
    rets = rng.normal(0.0003, 0.012, n)
    closes = pd.Series(
        100.0 * np.exp(np.cumsum(rets)),
        index=pd.bdate_range("2024-06-03", periods=n),
    )
    rv = ivx.compute_realized_vol(closes, 20)
    pct = ivx.compute_percentile_series(rv, 252)
    series = pd.DataFrame({"close": closes, "vol": rv, "vol_percentile": pct}).dropna(
        subset=["vol"]
    )
    analysis = ivx.analyze_forward_vol(
        series["vol"], forward_window=30, percentile=series["vol_percentile"]
    )
    current_vol = float(series["vol"].iloc[-1])
    current_pct = float(series["vol_percentile"].iloc[-1])
    iv_val = current_vol + 0.03
    iv_pct = ivx.percentile_within(series["vol"].tail(252), iv_val)
    return {
        "symbol": "SPY",
        "source": "alpaca",
        "years": 2.0,
        "rv_window": 20,
        "forward_window": 30,
        "percentile_window": 252,
        "series": series,
        "current_vol": current_vol,
        "current_percentile": current_pct,
        "regime": ivx.classify_regime(current_pct),
        "vol_stats": {
            "min": float(series["vol"].min()),
            "mean": float(series["vol"].mean()),
            "max": float(series["vol"].max()),
        },
        "current_iv": {
            "iv": iv_val,
            "spot": float(closes.iloc[-1]),
            "expiry": dt.date.today() + dt.timedelta(days=30),
            "dte": 30,
            "n_contracts": 6,
            "method": "greeks Alpaca",
            "feed": "indicative",
        },
        "iv_error": None,
        "iv_vs_series_percentile": iv_pct,
        "iv_regime": ivx.classify_regime(iv_pct),
        "iv_minus_rv": iv_val - current_vol,
        "iv_history": pd.DataFrame(
            {
                "date": pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=5),
                "iv": np.linspace(iv_val - 0.02, iv_val, 5),
            }
        ),
        "analysis": analysis,
        "analysis_error": None,
        "log": ["payload de test hors-ligne."],
        "generated_at": "12:00:00",
    }


def _tab_script():
    from app.vue.tabs import tab_iv_dashboard as tab

    tab.render_tab()


def main() -> None:
    # Run 1: seeded payload -> full dashboard
    at = AppTest.from_function(_tab_script, default_timeout=120)
    at.session_state["iv_dashboard_result"] = _build_payload()
    at.run()
    seeded = {
        "exceptions": [str(e.value) for e in at.exception],
        "n_charts": len(at.get("plotly_chart")),
        "n_metrics": len(at.metric),
    }

    # Run 2: no payload -> placeholder only
    at2 = AppTest.from_function(_tab_script, default_timeout=120)
    at2.run()
    empty = {
        "exceptions": [str(e.value) for e in at2.exception],
        "has_info": bool(at2.info),
        "n_charts": len(at2.get("plotly_chart")),
    }

    print(RESULT_MARKER + json.dumps({"seeded": seeded, "empty": empty}))


if __name__ == "__main__":
    main()
