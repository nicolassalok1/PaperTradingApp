"""Phase-4 skeptic probe (G3_ivmethod, code-reading lens).

Drives the REAL service.fetch_current_atm_iv / _snapshot_mid / _fetch_atm_snapshots
with monkeypatched network + spot, instead of re-implementing the code path.
Offline. Read-only on app/.
"""
from __future__ import annotations

import datetime as dt
import importlib
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model.calibration.implied_vol import bs_call_price  # noqa: E402
from app.model.iv_dashboard import service as svc  # noqa: E402


def bs_put(S, K, t, r, q, vol):
    return bs_call_price(S, K, t, r, q, vol) - S * math.exp(-q * t) + K * math.exp(-r * t)


def opra(sym, expiry, typ, K):
    return f"{sym}{expiry.strftime('%y%m%d')}{'C' if typ == 'call' else 'P'}{int(round(K * 1000)):08d}"


def build_snaps(S, r, q, sigma, dte, *, types=("call", "put"), band=0.10, spread=0.10, with_iv=False):
    today = dt.date.today()
    expiry = today + dt.timedelta(days=dte)
    T = dte / 365.0
    out = {}
    for K in np.arange(math.floor(S * (1 - band)), math.ceil(S * (1 + band)) + 1, 1.0):
        for typ in types:
            px = bs_call_price(S, K, T, r, q, sigma) if typ == "call" else bs_put(S, K, T, r, q, sigma)
            snap = {"latestQuote": {"bp": round(px - spread / 2, 2), "ap": round(px + spread / 2, 2)}}
            if with_iv:
                snap["impliedVolatility"] = sigma
            out[opra("SPY", expiry, typ, K)] = snap
    return out


def run_real(snaps, S, **kw):
    svc.fetch_spot_price = lambda sym: S  # type: ignore[assignment]
    svc._fetch_atm_snapshots = lambda *a, **k: snaps  # type: ignore[assignment]
    info, log = svc.fetch_current_atm_iv("SPY", **kw)
    return info, log


S, r, q, sigma, dte = 640.0, 0.04, 0.013, 0.16, 30
print(f"setup S={S} r={r} q={q} sigma={sigma} dte={dte}")
print("\n=== [A] parity-r-q-zero-bias : real fetch_current_atm_iv on synthetic chain (r=4%, q=1.3%) ===")
for label, types in (("calls+puts", ("call", "put")), ("calls only", ("call",)), ("puts only", ("put",))):
    for spread in (0.0, 0.10):
        info, log = run_real(build_snaps(S, r, q, sigma, dte, types=types, spread=spread), S)
        if info is None:
            print(f"  {label:11s} spread={spread:.2f}: None -> {log[-1]}")
            continue
        print(
            f"  {label:11s} spread={spread:.2f}: iv={info['iv']*100:.3f}% bias={(info['iv']-sigma)*1e4:+.0f} bp "
            f"n={info['n_contracts']} method={info['method']} dte={info['dte']}"
        )
snaps = build_snaps(S, r, q, sigma, dte, spread=0.0)
info, _ = run_real(snaps, S)
n_band = sum(1 for k in snaps if abs(svc._decode_opra(k)[0] / S - 1) <= 0.05)
print(f"  contracts in +-5% band: {n_band}, used: {info['n_contracts']} (dropped {n_band - info['n_contracts']})")
for types in (("call", "put"), ("call",), ("put",)):
    info, _ = run_real(build_snaps(S, r, q, sigma, dte, types=types, spread=0.0), S, r_annual=0.04)
    print(f"  with r_annual=0.04 (q still 0) {'+'.join(types):9s}: bias={(info['iv']-sigma)*1e4:+.0f} bp")
info, _ = run_real(build_snaps(S, r, q, sigma, dte, with_iv=True), S)
print(f"  with greeks present: bias={(info['iv']-sigma)*1e4:+.0f} bp method={info['method']}")
for d in (15, 45, 60):
    for types in (("call",), ("put",)):
        info, _ = run_real(build_snaps(S, r, q, sigma, d, types=types, spread=0.0), S)
        print(f"  dte={d} {types[0]:4s} only: bias={(info['iv']-sigma)*1e4:+.0f} bp (n={info['n_contracts']})")
# per-contract view: which contracts in the band give the extreme values (real inversion path via n=1 chains)
print("  per-contract (real path, single-contract chains):")
today = dt.date.today()
expiry = today + dt.timedelta(days=dte)
T = dte / 365.0
for m in (0.95, 0.97, 1.0, 1.03, 1.05):
    K = round(S * m)
    row = []
    for typ in ("call", "put"):
        px = bs_call_price(S, K, T, r, q, sigma) if typ == "call" else bs_put(S, K, T, r, q, sigma)
        sn = {opra("SPY", expiry, typ, K): {"latestQuote": {"bp": px, "ap": px}}}
        info, log = run_real(sn, S)
        row.append(f"{typ} {'None' if info is None else f'{(info[chr(105)+chr(118)]-sigma)*1e4:+.0f} bp'}")
    print(f"    K/S={m:.2f} K={K}: " + " | ".join(row))

print("\n=== [B] ask_only_and_crossed_mids : real _snapshot_mid ===")
cases = {
    "bid=0 ask=5.20": {"latestQuote": {"bp": 0, "ap": 5.20}},
    "bid missing ask=5.20": {"latestQuote": {"ap": 5.20}},
    "bid=5.30 ask=5.10 (crossed)": {"latestQuote": {"bp": 5.30, "ap": 5.10}},
    "bid=0 ask=0 trade p=4.00 (stale)": {"latestQuote": {"bp": 0, "ap": 0}, "latestTrade": {"p": 4.00, "t": "2026-01-02T15:00:00Z"}},
    "bid=5.10 ask=0": {"latestQuote": {"bp": 5.10, "ap": 0}},
    "no quote no trade": {},
}
for name, snap in cases.items():
    print(f"  {name:34s} -> mid={svc._snapshot_mid(snap)}")
for spread in (0.10, 0.50, 1.00):
    snaps = build_snaps(S, r, q, sigma, dte, spread=spread)
    for s in snaps.values():
        s["latestQuote"]["bp"] = 0
    info, _ = run_real(snaps, S)
    info2, _ = run_real(build_snaps(S, r, q, sigma, dte, spread=spread), S)
    print(
        f"  whole chain bid=0, spread={spread:.2f}: bias={(info['iv']-sigma)*1e4:+.0f} bp (n={info['n_contracts']}) ; "
        f"two-sided same spread: {(info2['iv']-sigma)*1e4:+.0f} bp => ask-only increment {(info['iv']-info2['iv'])*1e4:+.0f} bp"
    )

print("\n=== [C] snapshot-page-cap-silent : real _fetch_atm_snapshots with mocked requests.get ===")


class _Resp:
    def __init__(self, payload):
        self._p = payload
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def make_chain(n_expiries, start_dte=15, step=3, order="asc"):
    keys = []
    for i in range(n_expiries):
        e = today + dt.timedelta(days=start_dte + i * step)
        for K in np.arange(math.floor(S * 0.9), math.ceil(S * 1.1) + 1, 1.0):
            for typ in ("call", "put"):
                keys.append(opra("SPY", e, typ, K))
    keys = sorted(set(keys))
    if order == "desc":
        keys = keys[::-1]
    return keys


def mock_requests(keys):
    calls = []

    def _get(url, headers=None, params=None, timeout=None):
        calls.append(dict(params))
        lim = int(params.get("limit", 100))
        tok = params.get("page_token")
        start = int(tok) if tok else 0
        page = keys[start : start + lim]
        payload = {"snapshots": {k: {"latestQuote": {"bp": 1.0, "ap": 1.1}} for k in page}}
        nxt = start + lim
        if nxt < len(keys):
            payload["next_page_token"] = str(nxt)
        return _Resp(payload)

    return _get, calls


for order in ("asc", "desc"):
    importlib.reload(svc)
    keys = make_chain(19, order=order)
    _get, calls = mock_requests(keys)
    svc.requests.get = _get  # type: ignore[assignment]
    svc.get_secret = lambda k: "x"  # type: ignore[assignment]
    svc.fetch_spot_price = lambda sym: S  # type: ignore[assignment]
    snaps = svc._fetch_atm_snapshots("SPY", feed="indicative", spot=S, dte_min=15, dte_max=60)
    print(
        f"  order={order}: chain={len(keys)} fetched={len(snaps)} http_calls={len(calls)} "
        f"limits={[c.get('limit') for c in calls]} -> truncated={len(snaps) < len(keys)}"
    )
    expiries_fetched = sorted({svc._decode_opra(k)[1] for k in snaps})
    print(f"     expiries fetched: {len(expiries_fetched)} dte range {(expiries_fetched[0]-today).days}..{(expiries_fetched[-1]-today).days}")
    info, log = svc.fetch_current_atm_iv("SPY")
    print("     log:", " | ".join(log))
    flagged = any(("tronqu" in l.lower()) or ("truncat" in l.lower()) or ("page" in l.lower()) for l in log)
    print("     any truncation mention in log:", flagged, "| chosen dte:", None if info is None else info["dte"])
