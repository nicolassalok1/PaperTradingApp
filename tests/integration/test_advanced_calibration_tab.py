"""The Calibration avancée tab, driven end to end and offline.

What is pinned here was seen broken live (session 6 review):
- the ticker box was free text: the user had to guess which surfaces Yahoo can serve.
  Now a searchable pick-list — surfaces already in the local cache first, then the
  optionable universe — that still accepts a typed symbol;
- "Surfaces de prix (3D)" never rendered: `go.Surface(zmin=…)` is not a plotly property
  (it is `cmin`), so every calibration ended with a warning instead of three charts;
- the "Envoyer IV modèle vers Options" handoff.

Measured in a clean interpreter, like `test_app_boot.py`: the options bridge stubs the
real streamlit module at import when no ScriptRunContext exists, process-wide.
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

_RESULT_MARKER = "CALIB_TAB_RESULT "

_DRIVER = """
import json, socket, sys, tempfile
from pathlib import Path

repo_root, app_entry = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root)


def _blocked(*args, **kwargs):
    raise RuntimeError("network access is forbidden in this test")


socket.socket.connect = _blocked
socket.create_connection = _blocked

import numpy as np
import pandas as pd

# A fake local cache holding one Yahoo surface, so "cached first" is observable without
# touching the repo's real cache directory.
import app.model.options.data.iv_surface as ivs
fake_cache = Path(tempfile.mkdtemp(prefix="calib_tab_cache_"))
(fake_cache / "iv_surface_yahoo_ZZCACHED.csv").write_text("K,T,S0,iv,type\\n", encoding="utf-8")
ivs.CACHE_YAHOO_OPTION_CHAINS_DIR = fake_cache

# Synthetic smile, on and off the calibration grid (T up to 2y, m 0.8..1.2).
S0 = 250.0
rows = []
for T in (0.25, 0.5, 1.0, 1.5, 2.0):
    for m in np.linspace(0.8, 1.2, 9):
        iv = 0.22 + 0.15 * (m - 1.0) ** 2 - 0.08 * (m - 1.0) + 0.02 * np.sqrt(T)
        for typ in ("call", "put"):
            rows.append({{"K": S0 * m, "T": T, "S0": S0, "iv": float(iv), "type": typ}})
surface = pd.DataFrame(rows)

from streamlit.testing.v1 import AppTest

app = AppTest.from_file(app_entry, default_timeout=600)
app.session_state["adv_calib_yahoo_surface_df"] = surface
app.session_state["adv_calib_yahoo_surface_ticker"] = "AAPL"
app.session_state["adv_calib_yahoo_surface_max_years"] = 2.0
# What the user set in Options beforehand: the handoff must not clobber it.
app.session_state["d_common"] = 0.03
app.session_state["opt_iv_surface_type"] = "Put"
app.session_state["opt_iv_surface_max_years"] = 2.0
app.run()

out = {{"exceptions": [str(e.value) for e in app.exception]}}

sel = [w for w in app.selectbox if w.key == "adv_calib_yahoo_ticker_input"]
txt = [w for w in app.text_input if w.key == "adv_calib_yahoo_ticker_input"]
out["ticker_widget"] = "selectbox" if sel else ("text_input" if txt else None)
if sel:
    w = sel[0]
    out["ticker_value"] = w.value
    out["ticker_n_options"] = len(w.options)
    out["ticker_first_options"] = list(w.options[:3])
    out["ticker_has_spy"] = "SPY" in w.options
    out["ticker_cached_label"] = w.format_func("ZZCACHED")

run_btn = [b for b in app.button if b.key == "adv_calib_run_btn_sabr"][0]
out["run_disabled"] = bool(run_btn.disabled)
run_btn.click()
app.run()
res = app.session_state["last_advanced_calibration_result_sabr"] if "last_advanced_calibration_result_sabr" in app.session_state else None
out["calib_success"] = bool(res and res.get("success"))
out["calib_message"] = (res or {{}}).get("message")
out["calib_ticker"] = (res or {{}}).get("ticker")
out["calib_S0"] = (res or {{}}).get("S0")
out["calib_q"] = (res or {{}}).get("q")
out["metric_labels"] = [m.label for m in app.metric if m.label.startswith("r(")]
out["exceptions_after_calib"] = [str(e.value) for e in app.exception]
out["price_surface_warnings"] = [w.value for w in app.warning if "3D des prix" in w.value]

send_btn = [b for b in app.button if b.key == "adv_calib_send_to_opt_sabr"]
out["send_button_present"] = bool(send_btn)
if send_btn:
    send_btn[0].click()
    app.run()
    ss = app.session_state
    df_cal = ss["calib_model_surface_df"] if "calib_model_surface_df" in ss else None
    out["sent_rows"] = 0 if df_cal is None else int(len(df_cal))
    out["sent_meta"] = ss["calib_model_surface_meta"] if "calib_model_surface_meta" in ss else None
    out["sent_source"] = ss["opt_iv_surface_source"] if "opt_iv_surface_source" in ss else None
    out["sent_tkr_common"] = ss["tkr_common"] if "tkr_common" in ss else None
    out["sent_columns"] = [] if df_cal is None else list(df_cal.columns)
    out["d_common_after_send"] = ss["d_common"] if "d_common" in ss else None
    out["opt_max_years_after_send"] = ss["opt_iv_surface_max_years"] if "opt_iv_surface_max_years" in ss else None
    out["options_empty_surface_infos"] = [i.value for i in app.info if "vide apres filtrage" in i.value]
    out["exceptions_after_send"] = [str(e.value) for e in app.exception]

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


def test_boots_clean(outcome):
    assert not outcome["exceptions"], outcome["exceptions"]


def test_ticker_is_a_pick_list_of_available_surfaces_cached_first(outcome):
    assert outcome["ticker_widget"] == "selectbox"
    assert outcome["ticker_value"] == "AAPL"
    # the optionable universe (data/alpaca_optionable_tickers.csv, ~5.7k symbols)
    assert outcome["ticker_n_options"] > 1000
    assert outcome["ticker_has_spy"]
    # surfaces already in the local cache come first and are marked as such
    # (AppTest exposes the displayed labels, i.e. after format_func)
    assert outcome["ticker_first_options"][0].startswith("ZZCACHED")
    assert "cache" in outcome["ticker_cached_label"].lower()


def test_calibration_runs_and_price_surfaces_render(outcome):
    assert outcome["run_disabled"] is False
    assert outcome["calib_success"], outcome["calib_message"]
    assert outcome["calib_ticker"] == "AAPL"
    assert not outcome["exceptions_after_calib"], outcome["exceptions_after_calib"]
    assert outcome["price_surface_warnings"] == [], outcome["price_surface_warnings"]


def test_calibration_inputs_come_from_the_loaded_surface_and_the_user(outcome):
    # S0 is the surface's own spot column, q the user's Options dividend — not a chain
    # JSON picked by file name from an earlier download.
    assert outcome["calib_S0"] == pytest.approx(250.0)
    assert outcome["calib_q"] == pytest.approx(0.03)
    # r is read on the yield curve at the maturity of the surface actually loaded (2y here).
    assert outcome["metric_labels"] and outcome["metric_labels"][0].startswith("r(T=2.00)")


def test_send_to_options_hands_over_a_surface(outcome):
    assert outcome["send_button_present"]
    assert outcome["sent_rows"] > 0
    assert outcome["sent_source"] == "Calibration"
    assert outcome["sent_tkr_common"] == "AAPL"
    assert (outcome["sent_meta"] or {}).get("ticker") == "AAPL"
    assert not outcome["exceptions_after_send"], outcome["exceptions_after_send"]


def test_send_to_options_respects_what_the_user_set_in_options(outcome):
    # A model IV surface is one sigma(K, T) for calls and puts alike: with "Put" selected in
    # Options it must still display, not come out "vide apres filtrage".
    assert outcome["options_empty_surface_infos"] == [], outcome["options_empty_surface_infos"]
    assert "type" not in outcome["sent_columns"]
    # The user's Options slider and dividend are theirs.
    assert outcome["opt_max_years_after_send"] == pytest.approx(2.0)
    assert outcome["d_common_after_send"] == pytest.approx(0.03)
