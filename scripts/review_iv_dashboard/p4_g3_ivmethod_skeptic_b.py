"""Phase-4 skeptic probe addendum (G3_ivmethod): when does the call/put cancellation break?

Real fetch_current_atm_iv, synthetic chain priced with r=4%, q=1.3%.
(1) low vol / short DTE -> ITM puts dropped by the no-arb bound -> imbalance
(2) realistic SPY skew -> headline bias of the mixed median vs true ATM vol
"""
from __future__ import annotations

import datetime as dt
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model.calibration.implied_vol import bs_call_price  # noqa: E402
from app.model.iv_dashboard import service as svc  # noqa: E402

S, r, q = 640.0, 0.04, 0.013
today = dt.date.today()


def bs_put(S, K, t, r, q, vol):
    return bs_call_price(S, K, t, r, q, vol) - S * math.exp(-q * t) + K * math.exp(-r * t)


def opra(expiry, typ, K):
    return f"SPY{expiry.strftime('%y%m%d')}{'C' if typ == 'call' else 'P'}{int(round(K * 1000)):08d}"


def chain(dte, vol_fn, types=("call", "put")):
    expiry = today + dt.timedelta(days=dte)
    T = dte / 365.0
    out = {}
    for K in np.arange(math.floor(S * 0.9), math.ceil(S * 1.1) + 1, 1.0):
        vol = vol_fn(K / S)
        for typ in types:
            px = bs_call_price(S, K, T, r, q, vol) if typ == "call" else bs_put(S, K, T, r, q, vol)
            out[opra(expiry, typ, K)] = {"latestQuote": {"bp": px, "ap": px}}
    return out


def run(snaps):
    svc.fetch_spot_price = lambda sym: S  # type: ignore[assignment]
    svc._fetch_atm_snapshots = lambda *a, **k: snaps  # type: ignore[assignment]
    return svc.fetch_current_atm_iv("SPY")


print("(1) flat vol, count of contracts surviving in +-5% band (126 expected), and median bias")
for sigma in (0.08, 0.10, 0.12, 0.16, 0.25):
    for dte in (15, 30):
        info, log = run(chain(dte, lambda m, s=sigma: s))
        ic, _ = run(chain(dte, lambda m, s=sigma: s, types=("call",)))
        ip, _ = run(chain(dte, lambda m, s=sigma: s, types=("put",)))
        print(
            f"  sigma={sigma:.2f} dte={dte}: n={info['n_contracts']:3d} (calls {ic['n_contracts']}, puts {ip['n_contracts']}) "
            f"median bias={(info['iv']-sigma)*1e4:+.0f} bp | calls-only {(ic['iv']-sigma)*1e4:+.0f} | puts-only {(ip['iv']-sigma)*1e4:+.0f}"
        )

print("(2) realistic SPY skew: atm 16%, -0.6 vol pt per 1% moneyness (put skew), smile 0.03 pt per (1%)^2")
def skew(m):
    x = (m - 1.0) * 100
    return 0.16 - 0.006 * x + 0.0003 * x * x

for dte in (30,):
    info, _ = run(chain(dte, skew))
    # what a perfect inverter would give: median of true IVs across the same contracts
    Ks = [K for K in np.arange(math.floor(S * 0.9), math.ceil(S * 1.1) + 1, 1.0) if abs(K / S - 1) <= 0.05]
    true_med = float(np.median([skew(K / S) for K in Ks for _ in (0, 1)]))
    print(
        f"  dte={dte}: code median={info['iv']*100:.3f}% | median(true IVs in band)={true_med*100:.3f}% | ATM vol=16.000% "
        f"-> code vs true-median {(info['iv']-true_med)*1e4:+.0f} bp, code vs ATM {(info['iv']-0.16)*1e4:+.0f} bp, n={info['n_contracts']}"
    )
    ic, _ = run(chain(dte, skew, types=("call",)))
    ip, _ = run(chain(dte, skew, types=("put",)))
    print(f"     calls-only {ic['iv']*100:.3f}% | puts-only {ip['iv']*100:.3f}%")
