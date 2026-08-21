"""Probe: IV methodology biases of fetch_current_atm_iv's BS-inversion path (offline, synthetic).

Reproduces exactly the code path of service.py L400-L437:
  T = dte/365 ; call -> implied_vol_call(mid, S, K, T, r=0, q=0)
  put  -> C_synth = mid + S - K*exp(-0*T) ; implied_vol_call(C_synth, S, K, T, 0, 0)
True prices come from Black-Scholes with r=4%, q=1.3%.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model.calibration.implied_vol import bs_call_price, implied_vol_call  # noqa: E402


def bs_put_price(S, K, t, r, q, vol):
    c = bs_call_price(S, K, t, r, q, vol)
    return c - S * math.exp(-q * t) + K * math.exp(-r * t)


def code_iv(opt_type, mid, S, K, T, r_annual=0.0):
    if opt_type == "call":
        call_price = float(mid)
    else:
        call_price = float(mid) + S - K * math.exp(-r_annual * T)
    iv = implied_vol_call(call_price, S, K, T, r_annual, 0.0)
    if iv is not None and np.isfinite(iv) and 0.0 < iv < 5.0:
        return float(iv)
    return None


S, r, q, sigma, dte = 640.0, 0.04, 0.013, 0.16, 30
T = dte / 365.0
print(f"Setup: S={S} r={r} q={q} sigma={sigma} dte={dte} T={T:.5f}")
print("\n[1] Parity/discounting bias (q=0, r=0 in code) — recovered IV vs 16.00%")
print(f"{'K/S':>5} {'K':>7} {'call IV':>9} {'bias bp':>8} {'put IV':>9} {'bias bp':>8}")
for m in (0.95, 1.0, 1.05):
    K = S * m
    c_true = bs_call_price(S, K, T, r, q, sigma)
    p_true = bs_put_price(S, K, T, r, q, sigma)
    ivc = code_iv("call", c_true, S, K, T)
    ivp = code_iv("put", p_true, S, K, T)
    fmt = lambda v: f"{v*100:8.3f}%" if v is not None else "   None  "
    bc = f"{(ivc-sigma)*1e4:8.0f}" if ivc is not None else "    n/a"
    bp = f"{(ivp-sigma)*1e4:8.0f}" if ivp is not None else "    n/a"
    print(f"{m:5.2f} {K:7.1f} {fmt(ivc)} {bc} {fmt(ivp)} {bp}")

# ATM vega per $1 for sensitivity
eps = 1e-4
c0 = bs_call_price(S, S, T, 0, 0, sigma)
vega_per_volpt = (bs_call_price(S, S, T, 0, 0, sigma + 0.01) - c0)
print(f"\nATM call (r=q=0) price={c0:.3f}, vega per 1 vol pt = {vega_per_volpt:.3f} $ -> 1 $ of price error = {1/vega_per_volpt*100:.0f} bp of IV")

print("\n[2] _snapshot_mid: ask-only when bid missing/<=0 (L218-219)")
for spread in (0.02, 0.05, 0.10, 0.50, 1.00):
    mid = c0
    ask = mid + spread / 2
    iv_ask = code_iv("call", ask, S, S, T)
    print(f"  spread={spread:4.2f}$ true mid={mid:.3f} ask={ask:.3f} -> IV from ask = {iv_ask*100:.3f}% (bias {(iv_ask-sigma)*1e4:+.0f} bp)")
print("  task case: bid=0 / ask=5.20 (true value unknown). If fair mid were 5.10:",
      f"IV(ask)={code_iv('call', 5.20, S, S, T)*100:.2f}% vs IV(5.10)={code_iv('call', 5.10, S, S, T)*100:.2f}%")
print("  crossed quote bid=5.30 ask=5.10 accepted? code takes 0.5*(bid+ask) when both >0 -> yes, mid=5.20 (no check).")

print("\n[3] T = max(dte,1)/365 ignores intraday time / local-date offset")
for hours_extra in (6.5, 24):
    T_true = (dte * 24 + hours_extra) / (365 * 24)
    c_true = bs_call_price(S, S, T_true, 0, 0, sigma)
    iv_code = code_iv("call", c_true, S, S, T)
    print(f"  true remaining = {dte} d + {hours_extra} h : IV recovered with T=dte/365 -> {iv_code*100:.3f}% (bias {(iv_code-sigma)*1e4:+.0f} bp)")

print("\n[4] Band-median vs true ATM with skew, calls+puts mixed, code inversion path")
for label, skew_fn in (
    ("linear -0.5 vol pt per 1% moneyness", lambda m: sigma - 0.5 * (m - 1.0) * 100 / 100),
    ("linear skew + smile convexity 0.05 pt per (1%)^2", lambda m: sigma - 0.5 * (m - 1.0) + 0.05 * ((m - 1.0) * 100) ** 2 / 100),
):
    for band in (0.05, 0.10):
        ivs_direct, ivs_code = [], []
        n_call_drop = n_put_drop = 0
        strikes = np.arange(math.floor(S * (1 - band)), math.ceil(S * (1 + band)) + 1, 1.0)
        for K in strikes:
            m = K / S
            if abs(m - 1.0) > band:
                continue
            vol_k = skew_fn(m)
            c_true = bs_call_price(S, K, T, r, q, vol_k)
            p_true = bs_put_price(S, K, T, r, q, vol_k)
            ivs_direct += [vol_k, vol_k]
            ivc = code_iv("call", c_true, S, K, T)
            ivp = code_iv("put", p_true, S, K, T)
            if ivc is None:
                n_call_drop += 1
            else:
                ivs_code.append(ivc)
            if ivp is None:
                n_put_drop += 1
            else:
                ivs_code.append(ivp)
        med_direct = float(np.median(ivs_direct))
        med_code = float(np.median(ivs_code))
        print(f"  {label}, band ±{band:.0%}: {len(strikes)} strikes x2 | median(true IVs)={med_direct*100:.3f}% "
              f"(bias {(med_direct-sigma)*1e4:+.0f} bp) | median(code-inverted)={med_code*100:.3f}% "
              f"(bias {(med_code-sigma)*1e4:+.0f} bp) | dropped calls={n_call_drop} puts={n_put_drop}")

print("\n[5] best_expiry tie: dte 28 vs 32 with target 30")
import datetime as dt
today = dt.date(2026, 8, 21)
expiries = sorted({today + dt.timedelta(days=28), today + dt.timedelta(days=32)})
best = min(expiries, key=lambda e: abs((e - today).days - 30))
print("  chosen:", best, "->", (best - today).days, "DTE (min() keeps first of sorted = earlier)")
