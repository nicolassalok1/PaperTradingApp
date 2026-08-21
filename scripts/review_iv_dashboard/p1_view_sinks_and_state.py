"""
p1_view_sinks_and_state.py — §4.4 View probes (no network: sockets blocked).

1. Hostile symbol: where does it land? (unsafe_allow_html sinks vs escaped sinks)
2. Session-state lifecycle: exception on resubmit keeps the stale result under the error?
3. include_iv=False -> what does the IV warning say?
4. Rendering counts / duplicate keys on a second run.

Run:  .venv/Scripts/python.exe scripts/review_iv_dashboard/p1_view_sinks_and_state.py
"""
from __future__ import annotations

import datetime as dt
import json
import socket
import sys
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

HOSTILE = "<img src=x onerror=alert(1)>"


def build_payload(symbol: str, *, with_iv: bool = True, iv_error=None, years=2.0):
    from app.model.iv_dashboard import analytics as ivx

    rng = np.random.default_rng(5)
    n = 560
    rets = rng.normal(0.0003, 0.012, n)
    closes = pd.Series(100.0 * np.exp(np.cumsum(rets)), index=pd.bdate_range("2024-06-03", periods=n))
    rv = ivx.compute_realized_vol(closes, 20)
    pct = ivx.compute_percentile_series(rv, 252)
    series = pd.DataFrame({"close": closes, "vol": rv, "vol_percentile": pct}).dropna(subset=["vol"])
    analysis = ivx.analyze_forward_vol(series["vol"], forward_window=30, percentile=series["vol_percentile"])
    current_vol = float(series["vol"].iloc[-1])
    current_pct = float(series["vol_percentile"].iloc[-1])
    iv_val = current_vol + 0.03
    iv_pct = ivx.percentile_within(series["vol"].tail(252), iv_val)
    cur_iv = {
        "iv": iv_val, "spot": float(closes.iloc[-1]),
        "expiry": dt.date.today() + dt.timedelta(days=30), "dte": 30,
        "n_contracts": 6, "method": "greeks Alpaca", "feed": "indicative",
    } if with_iv else None
    return {
        "symbol": symbol, "source": "alpaca", "years": years, "rv_window": 20,
        "forward_window": 30, "percentile_window": 252, "series": series,
        "current_vol": current_vol, "current_percentile": current_pct,
        "regime": ivx.classify_regime(current_pct),
        "vol_stats": {"min": float(series["vol"].min()), "mean": float(series["vol"].mean()), "max": float(series["vol"].max())},
        "current_iv": cur_iv, "iv_error": iv_error,
        "iv_vs_series_percentile": iv_pct if with_iv else float("nan"),
        "iv_regime": ivx.classify_regime(iv_pct) if with_iv else None,
        "iv_minus_rv": (iv_val - current_vol) if with_iv else None,
        "iv_history": pd.DataFrame({"date": pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=5),
                                    "iv": np.linspace(iv_val - 0.02, iv_val, 5)}),
        "analysis": analysis, "analysis_error": None,
        "log": [f"Barres alpaca indisponibles : symbole {symbol} inconnu", "payload de test hors-ligne."],
        "generated_at": "12:00:00",
    }


def _script_render():
    from app.vue.tabs import tab_iv_dashboard as tab
    tab.render_tab()


def _script_render_with_failing_ctrl():
    from app.controller import iv_dashboard_controller as ctrl
    from app.vue.tabs import tab_iv_dashboard as tab

    def boom(sym, **kw):
        raise RuntimeError(f"Aucune donnee de prix disponible pour {sym} (Alpaca et fallback en echec).")

    ctrl.get_iv_analysis = boom
    tab.render_tab()


def _collect_sinks(at: AppTest):
    out = {}
    md = [(m.value, bool(getattr(m.proto, "allow_html", False))) for m in at.markdown]
    out["markdown_allow_html_with_hostile"] = [
        v[:80] for v, allow in md if allow and HOSTILE in v
    ]
    out["markdown_escaped_with_hostile"] = [
        v[:80] for v, allow in md if (not allow) and HOSTILE in v
    ]
    out["caption_with_hostile"] = [
        (c.value[:80], bool(getattr(c.proto, "allow_html", False))) for c in at.caption if HOSTILE in c.value
    ]
    out["code_with_hostile"] = [c.value[:120] for c in at.code if HOSTILE in c.value]
    out["warning_with_hostile"] = [w.value[:120] for w in at.warning if HOSTILE in w.value]
    out["error_with_hostile"] = [e.value[:120] for e in at.error if HOSTILE in e.value]
    out["n_allow_html_markdown_total"] = sum(1 for _, a in md if a)
    return out


def main():
    res = {}

    # --- 1. hostile symbol in a seeded result -------------------------------
    at = AppTest.from_function(_script_render, default_timeout=120)
    at.session_state["iv_dashboard_result"] = build_payload(HOSTILE)
    at.run()
    res["hostile_seeded"] = {"exceptions": [str(e.value) for e in at.exception], **_collect_sinks(at)}

    # --- 2. stale result kept under error after failed resubmit --------------
    at2 = AppTest.from_function(_script_render_with_failing_ctrl, default_timeout=120)
    at2.session_state["iv_dashboard_result"] = build_payload("SPY")
    at2.run()
    # change widget values then submit with a different symbol
    at2.text_input[0].set_value(HOSTILE)
    at2.selectbox[0].set_value("5 ans")
    at2.button[0].click()  # form submit button
    at2.run()
    res["stale_after_error"] = {
        "exceptions": [str(e.value) for e in at2.exception],
        "errors": [e.value[:160] for e in at2.error],
        "n_charts_after_error": len(at2.get("plotly_chart")),
        "captions": [c.value[:120] for c in at2.caption][:2],
        "selectbox_now": at2.selectbox[0].value,
        "result_years_in_state": at2.session_state["iv_dashboard_result"]["years"],
        "error_allow_html": [bool(getattr(e.proto, "allow_html", False)) for e in at2.error],
    }

    # --- 3. include_iv False -> iv_error None -> warning text -----------------
    at3 = AppTest.from_function(_script_render, default_timeout=120)
    at3.session_state["iv_dashboard_result"] = build_payload("SPY", with_iv=False, iv_error=None)
    at3.run()
    res["iv_disabled_warning"] = {
        "exceptions": [str(e.value) for e in at3.exception],
        "warnings": [w.value for w in at3.warning],
        "n_metrics": len(at3.metric),
    }

    # --- 4. two consecutive runs (rerun) -> duplicate key? --------------------
    at4 = AppTest.from_function(_script_render, default_timeout=120)
    at4.session_state["iv_dashboard_result"] = build_payload("SPY")
    at4.run()
    at4.run()
    res["rerun_twice"] = {
        "exceptions": [str(e.value) for e in at4.exception],
        "n_charts": len(at4.get("plotly_chart")),
        "chart_ids": [getattr(c.proto, "id", "") for c in at4.get("plotly_chart")],
    }

    # --- 5. metric/caption texts as a trader sees them -----------------------
    at5 = AppTest.from_function(_script_render, default_timeout=120)
    at5.session_state["iv_dashboard_result"] = build_payload("SPY")
    at5.run()
    res["copy"] = {
        "metrics": [(m.label, m.value) for m in at5.metric],
        "captions": [c.value for c in at5.caption],
        "chip_labels": [
            line.strip() for m in at5.markdown if getattr(m.proto, "allow_html", False)
            for line in m.value.splitlines() if line.strip() and "<" not in line
        ],
        "iv_pct_in_payload": at5.session_state["iv_dashboard_result"]["iv_vs_series_percentile"],
        "iv_regime": at5.session_state["iv_dashboard_result"]["iv_regime"],
    }

    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
