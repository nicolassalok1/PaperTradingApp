"""
Orchestrator probe — §4.2 plumbing edge cases (offline): OPRA decode attack set,
cache upsert TZ / corruption / race, alpaca-py 0.12.0 enums, fallback signature,
and the exception-text-in-log path. Deterministic, no network.
"""
from __future__ import annotations

import datetime as dt
import inspect
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from app.model.iv_dashboard import service as svc  # noqa: E402


def section(t):
    print(f"\n=== {t} ===")


section("A. _decode_opra attack set  (expected -> got)")
cases = [
    ("SPY260918C00450000", (450.0, dt.date(2026, 9, 18), "call")),
    ("SPXW260918C05000000", (5000.0, dt.date(2026, 9, 18), "call")),
    ("XSP260918P00640000", (640.0, dt.date(2026, 9, 18), "put")),
    ("BRKB260918C00480000", (480.0, dt.date(2026, 9, 18), "call")),
    ("AAPL1260918C00150000", (150.0, dt.date(2026, 9, 18), "call")),   # adjusted root AAPL1
    ("NDX260918C12345500", (12345.5, dt.date(2026, 9, 18), "call")),   # 5-digit strike
    ("SPY260918c00450000", (450.0, dt.date(2026, 9, 18), "call")),     # lowercase type
    ("SPY260918X00450000", None),                                      # bogus type -> silently 'put'?
    ("SPY26091C00450000", None),                                       # short date
    ("SPY260918C0045000", None),                                       # 7-digit strike
    ("garbage", None),
    ("", None),
    ("SPY2609180C00450000", None),                                    # extra digit in date zone
]
for s, exp in cases:
    got = svc._decode_opra(s)
    ok = (got == exp) if exp else (got == (None, None, None))
    flag = "OK " if ok else "BAD"
    print(f"  {flag} {s!r:26} -> {got}")

section("B. alpaca-py 0.12.0 enums / request fields")
try:
    from alpaca.data.enums import Adjustment, DataFeed
    print("  Adjustment.SPLIT =", Adjustment.SPLIT, "| DataFeed('iex') =", DataFeed("iex"))
except Exception as e:  # noqa: BLE001
    print("  import failed:", e)
from alpaca.data.requests import StockBarsRequest
print("  StockBarsRequest fields:", sorted(StockBarsRequest.model_fields.keys()) if hasattr(StockBarsRequest, "model_fields") else sorted(StockBarsRequest.__fields__.keys()))
try:
    from alpaca.data.requests import OptionSnapshotRequest, OptionChainRequest
    print("  OptionChainRequest fields:", sorted(OptionChainRequest.model_fields.keys()))
    print("  OptionSnapshotRequest fields:", sorted(OptionSnapshotRequest.model_fields.keys()))
except Exception as e:  # noqa: BLE001
    print("  option request import failed:", e)

section("C. fallback signature: download_options_alpaca(sym, feed=..., max_pages=...) valid?")
from app.model.options.logic import download_options_alpaca
sig = inspect.signature(download_options_alpaca)
print("  params:", list(sig.parameters))
try:
    sig.bind("SPY", feed="indicative", max_pages=3)
    print("  bind OK")
except TypeError as e:
    print("  bind FAIL:", e)

section("D. cache upsert: date is local today; reload dtype; empty-file corruption; DataFrame.get")
with tempfile.TemporaryDirectory() as td:
    with mock.patch.object(svc, "CACHE_IV_HISTORY_DIR", Path(td)):
        info = {"iv": 0.18, "dte": 30, "n_contracts": 6, "method": "greeks Alpaca", "spot": 640.0}
        svc.record_iv_observation("SPY", info)
        p = svc._iv_history_path("SPY")
        print("  written:", p.name, "| rows:", p.read_text().strip().splitlines())
        # second write same day -> upsert (1 row)
        svc.record_iv_observation("SPY", {**info, "iv": 0.19})
        print("  after 2nd write same day:", pd.read_csv(p)["iv"].tolist(), "(expect [0.19])")
        # TZ: local date vs US exchange date
        now_utc = dt.datetime.now(dt.timezone.utc)
        sgt = now_utc.astimezone(dt.timezone(dt.timedelta(hours=8))).date()
        ny = now_utc.astimezone(dt.timezone(dt.timedelta(hours=-4))).date()
        print(f"  now: SGT date={sgt} NY date={ny} local date.today()={dt.date.today()} -> differ? {sgt != ny}")
        # dtype on reload
        h = svc.load_iv_history("SPY")
        print("  load_iv_history dtypes:", dict(h.dtypes.astype(str)))
        # corrupt: empty file
        p.write_text("")
        svc.record_iv_observation("SPY", info)
        print("  after write on EMPTY file: content=", repr(p.read_text()[:80]), "(observation lost silently if empty)")
        # DataFrame.get semantics
        df = pd.DataFrame({"date": ["2026-08-20"], "iv": [0.1]})
        print("  DataFrame.get('date') is Series:", isinstance(df.get("date", pd.Series(dtype=str)), pd.Series))
        # missing 'date' column file
        p.write_text("iv\n0.1\n")
        svc.record_iv_observation("SPY", info)
        print("  after write on file w/o date col:", repr(p.read_text()[:120]))

section("E. exception text reaches the log (HTML 401 case) — what does the view render it with?")
import subprocess  # noqa: E402
out = subprocess.run(["grep", "-n", "-E", "st\\.(code|text|markdown|caption|write)\\(|for ln in|log", str(ROOT / "app/vue/tabs/tab_iv_dashboard.py")], capture_output=True, text=True).stdout
print("\n".join(l for l in out.splitlines() if "_render_log" in l or "st.code" in l or "st.text" in l or "log" in l.lower())[:1200])

section("F. get_iv_dashboard_data: symbol with HTML reaches log?")
with mock.patch.object(svc, "fetch_daily_closes", return_value=(pd.DataFrame(), "none", ["Symbole <b>X</b> KO"])):
    try:
        svc.get_iv_dashboard_data("<img src=x onerror=alert(1)>")
    except RuntimeError as e:
        print("  RuntimeError text:", str(e)[:120])
