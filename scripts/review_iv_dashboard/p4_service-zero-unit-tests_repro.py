"""p4 skeptic repro for service-zero-unit-tests.

1. grep tests/ for any direct use of service.py symbols
2. service.py coverage under the real CI gate selection (-m "unit or smoke", whole suite)
3. own minimal offline tests of the service (pagination, opra, cache upsert, fallback chain,
   orchestrator degradation) -- are they feasible and green? Patch-target check included.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
PY = sys.executable
sys.path.insert(0, str(ROOT))
for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
    os.environ.pop(k, None)


def _blocked(*a, **k):  # noqa: ANN002, ANN003
    raise RuntimeError("network forbidden")


socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.create_connection = _blocked  # type: ignore[assignment]

# ---------------------------------------------------------------- 1. grep
names = [
    "get_iv_dashboard_data", "fetch_daily_closes", "fetch_current_atm_iv", "record_iv_observation",
    "load_iv_history", "_fetch_atm_snapshots", "_decode_opra", "_snapshot_mid", "_snapshot_iv",
    "iv_dashboard.service", "iv_dashboard import service",
]
hits = []
for p in (ROOT / "tests").rglob("*.py"):
    txt = p.read_text(encoding="utf-8", errors="replace")
    for i, ln in enumerate(txt.splitlines(), 1):
        if any(n in ln for n in names):
            hits.append(f"{p.relative_to(ROOT)}:{i}: {ln.strip()}")
print("=== 1. service symbol references in tests/:")
print("\n".join(hits) or "  (none)")

# ---------------------------------------------------------------- 2. coverage under the CI gate
print("\n=== 2. service.py coverage under the CI gate (-m 'unit or smoke', full suite)")
cov_json = ROOT / "scripts" / "review_iv_dashboard" / "p4_service_cov.json"
cp = subprocess.run(
    [PY, "-m", "pytest", "-m", "unit or smoke", "-q", "-p", "no:cacheprovider",
     "--cov=app.model.iv_dashboard", f"--cov-report=json:{cov_json}", "--cov-fail-under=0",
     "-o", "addopts=--strict-markers --disable-socket"],
    cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
)
tail = [ln for ln in cp.stdout.splitlines() if "passed" in ln or "failed" in ln or "error" in ln.lower()][-3:]
print("  pytest exit:", cp.returncode, "|", tail)
if cov_json.exists():
    data = json.loads(cov_json.read_text(encoding="utf-8"))
    for f, d in data["files"].items():
        if "iv_dashboard" in f:
            s = d["summary"]
            print(f"  {f}: {s['covered_lines']}/{s['num_statements']} stmts = {s['percent_covered']:.1f}%"
                  f" ; missing ranges: {d['missing_lines'][:3]}..{d['missing_lines'][-3:]} (n={len(d['missing_lines'])})")
    cov_json.unlink()
else:
    print("  no coverage json produced\n", cp.stdout[-2000:], cp.stderr[-2000:])

# ---------------------------------------------------------------- 3. own offline tests
print("\n=== 3. own offline tests against service.py")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from app.model.iv_dashboard import service as svc  # noqa: E402
import app.utils.paths as paths  # noqa: E402

results = {}


def check(name, fn):
    try:
        fn()
        results[name] = "PASS"
    except Exception as exc:  # noqa: BLE001
        results[name] = f"FAIL: {exc!r}\n{traceback.format_exc(limit=3)}"


class _Resp:
    def __init__(s, payload, status=200):
        s._p, s.status_code = payload, status

    def raise_for_status(s):
        if s.status_code >= 400:
            raise RuntimeError(f"HTTP {s.status_code}")

    def json(s):
        return s._p


def t_pagination():
    orig_h, orig_get = svc._alpaca_data_headers, svc.requests.get
    calls = []

    def fake_get(url, headers=None, params=None, timeout=None):
        calls.append(dict(params))
        p = len(calls)
        return _Resp({"snapshots": {f"SPY260918C0045{p}000": {"p": p}}, "next_page_token": f"tok{p}"})

    svc._alpaca_data_headers = lambda: {"k": "v"}
    svc.requests.get = fake_get
    try:
        out = svc._fetch_atm_snapshots("spy", feed="indicative", spot=450.0, dte_min=15, dte_max=60)
        assert len(calls) == svc._SNAPSHOT_MAX_PAGES == 3, calls
        assert "page_token" not in calls[0] and calls[1]["page_token"] == "tok1" and calls[2]["page_token"] == "tok2"
        assert abs(calls[0]["strike_price_gte"] - 405.0) < 1e-9 and abs(calls[0]["strike_price_lte"] - 495.0) < 1e-9
        assert len(out) == 3
        # no token -> single call
        calls.clear()
        svc.requests.get = lambda *a, **k: _Resp({"snapshots": {"SPY260918C00450000": {}}})
        out = svc._fetch_atm_snapshots("spy", feed="indicative", spot=None, dte_min=15, dte_max=60)
        assert len(out) == 1
        # 403 propagates
        svc.requests.get = lambda *a, **k: _Resp({}, 403)
        try:
            svc._fetch_atm_snapshots("spy", feed="indicative", spot=None, dte_min=15, dte_max=60)
            raise AssertionError("expected raise")
        except RuntimeError as e:
            assert "403" in str(e)
        # no creds -> EnvironmentError, requests.get never called
        svc._alpaca_data_headers = lambda: None
        hit = []
        svc.requests.get = lambda *a, **k: hit.append(1)
        try:
            svc._fetch_atm_snapshots("spy", feed="indicative", spot=None, dte_min=15, dte_max=60)
            raise AssertionError("expected EnvironmentError")
        except EnvironmentError:
            assert not hit
    finally:
        svc._alpaca_data_headers, svc.requests.get = orig_h, orig_get


def t_opra():
    assert svc._decode_opra("SPY260918C00450000") == (450.0, dt.date(2026, 9, 18), "call")
    assert svc._decode_opra("SPY260918P00450500") == (450.5, dt.date(2026, 9, 18), "put")
    assert svc._decode_opra("BRKB260918C00450000") == (450.0, dt.date(2026, 9, 18), "call")
    assert svc._decode_opra("garbage") == (None, None, None)
    assert svc._decode_opra("") == (None, None, None)
    assert svc._snapshot_mid({"latestQuote": {"bp": 1.0, "ap": 1.2}}) == 1.1
    assert svc._snapshot_mid({"latestQuote": {"bp": 0, "ap": 1.2}}) == 1.2
    assert svc._snapshot_mid({"latestQuote": {"bp": 0, "ap": 0}, "latestTrade": {"p": 0.9}}) == 0.9
    assert svc._snapshot_mid({}) is None
    assert svc._snapshot_iv({"greeks": {"iv": 0.21}}) == 0.21
    assert svc._snapshot_iv({"latestGreeks": {"impliedVolatility": "0.3"}}) == 0.3
    assert np.isnan(svc._snapshot_iv({"greeks": "x"}))
    assert np.isnan(svc._snapshot_iv({}))


def t_patch_target_and_cache():
    tmp = Path(tempfile.mkdtemp(prefix="p4svc_"))
    # patch-target claim: app.utils.paths does NOT redirect, svc.CACHE_IV_HISTORY_DIR does
    orig_paths, orig_svc = paths.CACHE_IV_HISTORY_DIR, svc.CACHE_IV_HISTORY_DIR
    paths.CACHE_IV_HISTORY_DIR = tmp
    via_paths = str(svc._iv_history_path("SPY")).startswith(str(tmp))
    paths.CACHE_IV_HISTORY_DIR = orig_paths
    svc.CACHE_IV_HISTORY_DIR = tmp
    via_svc = str(svc._iv_history_path("SPY")).startswith(str(tmp))
    print(f"  patch via app.utils.paths redirects: {via_paths} ; via svc module global: {via_svc}")
    assert via_paths is False and via_svc is True
    try:
        yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
        pd.DataFrame([{"date": yesterday, "iv": 0.19, "dte": 29, "n_contracts": 5, "method": "x", "spot": 99.0}]).to_csv(
            tmp / "iv_daily_SPY.csv", index=False
        )
        svc.record_iv_observation(" spy ", {"iv": 0.21, "dte": 30, "n_contracts": 6, "method": "g", "spot": 100.0})
        svc.record_iv_observation("SPY", {"iv": 0.25, "dte": 31, "n_contracts": 4, "method": "m", "spot": 101.0})
        hist = svc.load_iv_history("spy")
        assert hist["iv"].tolist() == [0.19, 0.25], hist
        assert list(svc.load_iv_history("ZZZZ").columns) == ["date", "iv"]
        (tmp / "iv_daily_BAD.csv").write_text("not,a,csv\n\x00", encoding="utf-8")
        assert svc.load_iv_history("BAD").empty
        svc.record_iv_observation("SPY", {"iv": None})  # must not raise
        assert svc.load_iv_history("SPY")["iv"].tolist() == [0.19, 0.25]
    finally:
        svc.CACHE_IV_HISTORY_DIR = orig_svc


def _closes_df(n=600):
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    rng = np.random.default_rng(1)
    return pd.DataFrame({"Date": idx, "Close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})


def t_fallback_chain():
    orig_a, orig_f = svc._fetch_closes_alpaca, svc.fetch_ohlc_history
    feeds = []

    def boom(sym, start, *, feed=None):
        feeds.append(feed)
        raise RuntimeError("down")

    svc._fetch_closes_alpaca = boom
    svc.fetch_ohlc_history = lambda sym, period, interval: _closes_df()
    try:
        df, source, log = svc.fetch_daily_closes("spy", years=2.0)
        assert feeds == [None, "iex"], feeds
        assert source == "fallback (Stooq/Yahoo)", source
        assert sum("indisponibles" in m for m in log) == 2, log
        assert not df.empty and list(df.columns) == ["Date", "Close"]
        svc.fetch_ohlc_history = lambda *a, **k: None
        assert svc.fetch_daily_closes("spy")[1] == "none"
        assert svc.fetch_daily_closes("  ")[1] == "none"
    finally:
        svc._fetch_closes_alpaca, svc.fetch_ohlc_history = orig_a, orig_f


def _bs_put(S, K, T, sig, r=0.0):
    import math
    d1 = (math.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def t_current_iv_mixed():
    orig_spot, orig_snaps = svc.fetch_spot_price, svc._fetch_atm_snapshots
    os.environ.pop("ALPACA_OPTION_DATA_FEED", None)
    today = dt.date.today()
    expiry = today + dt.timedelta(days=30)
    tag = expiry.strftime("%y%m%d")
    put_mid = _bs_put(100.0, 100.0, 30 / 365, 0.25)
    snaps = {
        f"SPY{tag}C00100000": {"greeks": {"iv": 0.21}},
        f"SPY{tag}P00100000": {"latestQuote": {"bp": put_mid - 0.01, "ap": put_mid + 0.01}},
        f"SPY{tag}C00130000": {"greeks": {"iv": 0.9}},
    }
    svc.fetch_spot_price = lambda s: 100.0
    svc._fetch_atm_snapshots = lambda *a, **k: snaps
    try:
        info, log = svc.fetch_current_atm_iv("SPY")
        print(f"  fetch_current_atm_iv -> {info}")
        assert info is not None, log
        assert info["n_contracts"] == 2 and info["dte"] == 30, info
        assert abs(info["iv"] - np.median([0.21, 0.25])) < 2e-3, info["iv"]
        assert "mixte" in info["method"], info["method"]
        svc.fetch_spot_price = lambda s: None
        info2, log2 = svc.fetch_current_atm_iv("SPY")
        assert info2 is None and "Spot" in log2[-1], log2
    finally:
        svc.fetch_spot_price, svc._fetch_atm_snapshots = orig_spot, orig_snaps


def t_orchestrator_degrades():
    orig_dir, orig_fdc, orig_iv = svc.CACHE_IV_HISTORY_DIR, svc.fetch_daily_closes, svc.fetch_current_atm_iv
    svc.CACHE_IV_HISTORY_DIR = Path(tempfile.mkdtemp(prefix="p4svc2_"))
    svc.fetch_daily_closes = lambda sym, **k: (_closes_df(), "alpaca", ["ok"])
    svc.fetch_current_atm_iv = lambda sym: (None, ["Spot indisponible (test)"])
    try:
        out = svc.get_iv_dashboard_data("spy")
        assert out["current_iv"] is None and out["iv_error"] == "Spot indisponible (test)"
        assert out["analysis"] is not None and out["iv_history"].empty and out["iv_minus_rv"] is None
        svc.fetch_daily_closes = lambda sym, **k: (pd.DataFrame(), "none", ["nope"])
        try:
            svc.get_iv_dashboard_data("SPY")
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "Aucune donn" in str(e)
    finally:
        svc.CACHE_IV_HISTORY_DIR, svc.fetch_daily_closes, svc.fetch_current_atm_iv = orig_dir, orig_fdc, orig_iv


for name, fn in [
    ("pagination+params+cap+creds", t_pagination),
    ("opra/mid/iv parsers", t_opra),
    ("patch target + cache upsert round trip", t_patch_target_and_cache),
    ("fallback chain", t_fallback_chain),
    ("fetch_current_atm_iv greeks+parity", t_current_iv_mixed),
    ("orchestrator degradation", t_orchestrator_degrades),
]:
    check(name, fn)

for k, v in results.items():
    print(f"  [{v.split(':')[0]}] {k}" + ("" if v == "PASS" else "\n" + v))
