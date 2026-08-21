"""p4 repro — html-exception-text-in-log (offline, deterministic).

Question: when Alpaca's edge answers 401 with an nginx HTML page, does
fetch_daily_closes() put the raw multi-line HTML into the log list that the
view dumps into st.code ?  And what does fetch_current_atm_iv() put into the
log / iv_error on the same 401 ?

Method: patch requests.Session.request (used by alpaca-py RESTClient) and
requests.get (used by service._fetch_atm_snapshots) with a fake 401 HTML
response. No network, no credentials.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import requests  # noqa: E402

import app.model.iv_dashboard.service as svc  # noqa: E402

HTML_401 = (
    "<html>\r\n<head><title>401 Authorization Required</title></head>\r\n<body>\r\n"
    "<center><h1>401 Authorization Required</h1></center>\r\n<hr><center>nginx</center>\r\n"
    "</body>\r\n</html>\r\n"
)


def fake_response(url: str) -> requests.Response:
    r = requests.Response()
    r.status_code = 401
    r.reason = "Unauthorized"
    r.url = url
    r._content = HTML_401.encode()
    r.headers["Content-Type"] = "text/html"
    return r


# alpaca-py RESTClient -> self._session.request(method, url, **opts)
def fake_session_request(self, method, url, **kwargs):
    return fake_response(url)


requests.Session.request = fake_session_request
requests.get = lambda url, **kw: fake_response(url + "?feed=indicative&limit=1000&expiration_date_gte=2026-09-05")

# credentials: fake, never real
svc._alpaca_keys = lambda: ("k", "s")
svc._alpaca_data_headers = lambda: {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}
# last-resort fallback: empty so the function returns after logging both feeds
svc.fetch_ohlc_history = lambda *a, **k: pd.DataFrame()

print("=== fetch_daily_closes('SPY') with 401 HTML on both Alpaca feeds ===")
df, tag, log = svc.fetch_daily_closes("SPY")
print("source tag:", tag, "| rows:", len(df))
for i, m in enumerate(log):
    print(f"log[{i}]: {len(m)} chars, {m.count(chr(10)) + 1} lines, repr head: {m[:70]!r}")
html_msgs = [m for m in log if "<html>" in m]
print("messages containing raw <html>:", len(html_msgs))
print("max lines in a single log message:", max(m.count("\n") + 1 for m in log))
print("what st.code would show (log part only):")
print("-" * 60)
print("\n".join(log))
print("-" * 60)

print("\n=== fetch_current_atm_iv('SPY') with 401 HTML on the snapshot endpoint ===")
svc.fetch_spot_price = lambda s: 640.0
import app.model.options.logic as logic  # noqa: E402

logic._load_alpaca_credentials = lambda: ("k", "s", None)
logic.time.sleep = lambda *_: None
info, ivlog = svc.fetch_current_atm_iv("SPY")
for i, m in enumerate(ivlog):
    print(f"ivlog[{i}]: {len(m)} chars, {m.count(chr(10)) + 1} lines: {m[:160]!r}")
print("iv_error (= ivlog[-1]) :", repr(ivlog[-1]))
