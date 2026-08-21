"""The 📡 Kalman Filters tab, driven end to end and offline.

Pinned here is what the 2026-08-21 adversarial review saw wrong in the view:
- "calibration anchored at the tail" was claimed whenever no backtest existed, even
  with a head anchor whose window ate the whole series;
- the live fragment advanced the filter on EVERY rerun (any widget, any tab) and at
  every poll, with the per-bar (phi, Q) — ticks are now gated on the bar clock;
- the live state kept stale (phi, mu, Q) after a recalibration (only R was resynced);
- without Alpaca keys the "spot" was a disk-cached Stooq close of unknown age;
- "Close position" read `qty` before `qty_available` and could be sent twice while
  the first order was pending -> position reversal; fractional closes were sent GTC.

Measured in a clean interpreter, like `test_app_boot.py` (the options bridge stubs the
real streamlit module at import when no ScriptRunContext exists, process-wide).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ENTRY = REPO_ROOT / "streamlit_app.py"

_SECRET_ENV_KEYS = [
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_API_SECRET",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "OPENAI_API_KEY",
]

_RESULT_MARKER = "KALMAN_TAB_RESULT "

_DRIVER = """
import json, socket, sys

repo_root, app_entry = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root)


def _blocked(*args, **kwargs):
    raise RuntimeError("network access is forbidden in this test")


socket.socket.connect = _blocked
socket.create_connection = _blocked

import numpy as np

from app.controller import kalman_controller as ctrl
from app.controller import trading_controller as trading_ctrl


def _synthetic_bars(n=300, phi=0.9, mu=100.0, sig=0.5, seed=1):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + sig * rng.standard_normal()
    return [
        {{"t": "2026-08-21T%02d:%02d:00+00:00" % (10 + i // 60, i % 60), "o": float(v), "h": float(v) + 0.1,
          "l": float(v) - 0.1, "c": float(v)}}
        for i, v in enumerate(x)
    ]


def fake_load_history(ticker, bar_label, *, max_bars=500):
    return {{"success": True, "message": "fake", "bars": _synthetic_bars(), "bar_seconds": 300,
            "interval": "5m", "period": "1mo"}}


spot = {{"px": 103.0}}
broker = {{"positions": [], "orders": []}}
ctrl.load_history = fake_load_history
ctrl.latest_price = lambda ticker: spot["px"]
trading_ctrl.get_spot_positions = lambda: list(broker["positions"])


def fake_send(symbol, qty, side):
    broker["orders"].append([symbol, float(qty), side])
    return {{"id": "fake-%d" % len(broker["orders"])}}


trading_ctrl.send_spot_market_order = fake_send

from streamlit.testing.v1 import AppTest

app = AppTest.from_file(app_entry, default_timeout=600)
app.run()
out = {{"exceptions": [str(e.value) for e in app.exception]}}
d_common_before = app.session_state["d_common"] if "d_common" in app.session_state else None

load_btn = [b for b in app.button if "Charger & calibrer OU" in (b.label or "")][0]
load_btn.click()
app.run()
out["exceptions_after_load"] = [str(e.value) for e in app.exception]
out["kf_keys"] = sorted(k for k in app.session_state.filtered_state if str(k).startswith("kf_"))
out["d_common_unchanged"] = (app.session_state["d_common"] if "d_common" in app.session_state else None) == d_common_before

# --- window >= series with a head anchor: no backtest, and the message must say why
app.number_input(key="kf_calib_window").set_value(2000)
app.run()
out["no_backtest_infos"] = [i.value for i in app.info if "Backtest" in i.value]
app.number_input(key="kf_calib_window").set_value(60)
app.run()

# --- live: first tick, then bar-clock gating across unrelated reruns
app.toggle(key="kf_live_on").set_value(True)
app.run()
live1 = dict(app.session_state["kf_live_state"])
out["live_started"] = "x" in live1
for k in (1.25, 1.5, 1.75):
    app.slider(key="kf_band_k").set_value(k)
    app.run()
live2 = dict(app.session_state["kf_live_state"])
out["live_x_unchanged_across_reruns"] = live1["x"] == live2["x"]
out["live_P_unchanged_across_reruns"] = live1["P"] == live2["P"]

# --- recalibration while live: the whole state follows (phi, mu, Q), not only R
app.radio(key="kf_calib_anchor").set_value("Fin de série (trading live)")
app.run()
live3 = dict(app.session_state["kf_live_state"])
res3 = dict(app.session_state["kf_result_state"])
out["live_resynced_after_recalibration"] = all(live3[k] == res3[k] for k in ("phi", "mu", "Q", "R"))
out["tail_anchor_infos"] = [i.value for i in app.info if "Backtest" in i.value]
out["exceptions_after_live"] = [str(e.value) for e in app.exception]

# --- no live price (no Alpaca keys): an explicit warning, no tick
spot["px"] = None
app.slider(key="kf_band_k").set_value(1.0)
app.run()
out["no_spot_warnings"] = [w.value for w in app.warning if "Alpaca" in w.value]
spot["px"] = 103.0
app.toggle(key="kf_live_on").set_value(False)
app.run()

# --- orders
def _close_btn():
    return [b for b in app.button if b.key == "kf_btn_close"][0]

app.checkbox(key="kf_orders_enabled").set_value(True)
broker["positions"] = [{{"symbol": "AAPL", "qty": "-10", "qty_available": "0"}}]
app.run()
out["close_disabled_when_nothing_available"] = bool(_close_btn().disabled)

broker["positions"] = [{{"symbol": "AAPL", "qty": "-10", "qty_available": "-10"}}]
app.run()
out["close_enabled_when_available"] = not _close_btn().disabled
_close_btn().click()
app.run()
out["orders_after_close"] = list(broker["orders"])
# the broker still reports the (pending) position: the tab must not offer a second Close
out["close_disabled_after_send"] = bool(_close_btn().disabled)
out["long_disabled_after_send"] = bool([b for b in app.button if b.key == "kf_btn_long"][0].disabled)
out["exceptions_after_orders"] = [str(e.value) for e in app.exception]

broker["positions"] = [{{"symbol": "AAPL", "qty": "0.5", "qty_available": "0.5"}}]
unlock = [b for b in app.button if b.key == "kf_btn_unlock"]
out["unlock_button_present"] = bool(unlock)
if unlock:
    unlock[0].click()
    app.run()
out["close_disabled_for_fractional"] = bool(_close_btn().disabled)
out["fractional_captions"] = [c.value for c in app.caption if "fractionnaire" in c.value]

print({marker!r} + json.dumps(out, default=str))
""".format(marker=_RESULT_MARKER)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def outcome():
    env = {k: v for k, v in os.environ.items() if k not in _SECRET_ENV_KEYS}
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(REPO_ROOT), str(APP_ENTRY)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=900,
        cwd=str(REPO_ROOT),
    )
    line = next(
        (ln[len(_RESULT_MARKER) :] for ln in completed.stdout.splitlines() if ln.startswith(_RESULT_MARKER)),
        None,
    )
    assert line is not None, (
        "the driver produced no result line\n"
        f"--- stdout ---\n{completed.stdout[-4000:]}\n--- stderr ---\n{completed.stderr[-4000:]}"
    )
    return json.loads(line)


def test_boots_and_loads_clean(outcome):
    assert not outcome["exceptions"], outcome["exceptions"]
    assert not outcome["exceptions_after_load"], outcome["exceptions_after_load"]
    assert "kf_result_state" in outcome["kf_keys"]
    assert outcome["d_common_unchanged"]  # the tab never touches the Options shared params


def test_missing_backtest_message_names_the_real_cause(outcome):
    msgs = outcome["no_backtest_infos"]
    assert msgs, outcome
    assert all("fin de série" not in m for m in msgs), msgs
    assert any("fenêtre" in m for m in msgs), msgs
    # the tail-anchor explanation is still given when that IS the cause
    assert any("fin de série" in m for m in outcome["tail_anchor_infos"]), outcome["tail_anchor_infos"]


def test_live_ticks_follow_the_bar_clock_not_the_reruns(outcome):
    assert outcome["live_started"]
    assert outcome["live_x_unchanged_across_reruns"]
    assert outcome["live_P_unchanged_across_reruns"]
    assert not outcome["exceptions_after_live"], outcome["exceptions_after_live"]


def test_live_state_follows_a_recalibration(outcome):
    assert outcome["live_resynced_after_recalibration"]


def test_no_live_price_is_said_explicitly(outcome):
    assert outcome["no_spot_warnings"], outcome


def test_close_uses_available_qty_and_cannot_be_sent_twice(outcome):
    assert outcome["close_disabled_when_nothing_available"]
    assert outcome["close_enabled_when_available"]
    assert outcome["orders_after_close"] == [["AAPL", 10.0, "buy"]]
    assert outcome["close_disabled_after_send"]
    assert outcome["long_disabled_after_send"]
    assert outcome["unlock_button_present"]
    assert not outcome["exceptions_after_orders"], outcome["exceptions_after_orders"]


def test_fractional_position_is_not_closed_from_here(outcome):
    assert outcome["close_disabled_for_fractional"]
    assert outcome["fractional_captions"]
