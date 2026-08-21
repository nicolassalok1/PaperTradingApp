"""p4 repro — parity-r-q-zero-bias.

Drive the REAL service.fetch_current_atm_iv (monkeypatched spot + snapshots, no network)
with a synthetic chain priced under Black-Scholes with r=4%, q=1.3%, flat sigma, and
measure the recovered headline IV vs. the true sigma, for several call/put mixes.
Also measure the per-contract bias with the exact arithmetic of service.py L411-417.
"""
from __future__ import annotations
import datetime as dt, math, sys
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
import numpy as np
from scipy.stats import norm
from app.model.iv_dashboard import service as svc
from app.model.calibration.implied_vol import implied_vol_call

S, R, Q = 640.0, 0.04, 0.013
DTE = 30
T = DTE / 365.0
TODAY = dt.date.today()
EXPIRY = TODAY + dt.timedelta(days=DTE)

def bs(S, K, T, r, q, sig, typ):
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if typ == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

def opra(K, typ):
    return f"SPY{EXPIRY:%y%m%d}{'C' if typ=='call' else 'P'}{int(round(K*1000)):08d}"

def make_snaps(strikes, types, sig_fn, spread=0.0):
    snaps = {}
    for K in strikes:
        for typ in types:
            px = bs(S, K, T, R, Q, sig_fn(K), typ)
            snaps[opra(K, typ)] = {"latestQuote": {"bp": round(px - spread / 2, 4), "ap": round(px + spread / 2, 4)}}
    return snaps

def run(snaps, band=0.05):
    svc.fetch_spot_price = lambda sym: S
    svc._fetch_atm_snapshots = lambda *a, **k: snaps
    info, log = svc.fetch_current_atm_iv("SPY", moneyness_band=band)
    return info, log

# --- per-contract bias, mirroring service.py L411-417 exactly ---------------
print("== per-contract bias (true sigma 16%, r=4%, q=1.3%, 30 DTE) ==")
sig0 = 0.16
for ks in (0.95, 0.97, 1.00, 1.03, 1.05):
    K = round(S * ks)
    out = []
    for typ in ("call", "put"):
        mid = bs(S, K, T, R, Q, sig0, typ)
        if typ == "call":
            cp = mid
        else:
            cp = mid + S - K * math.exp(-0.0 * T)      # L416 with r_annual=0
        iv = implied_vol_call(cp, S, K, T, 0.0, 0.0)   # L417
        out.append(f"{typ}: {'nan' if not np.isfinite(iv) else f'{(iv-sig0)*1e4:+.0f} bp'}")
    print(f"  K/S={ks:.2f} (K={K}):  " + " | ".join(out))

# --- headline median through the real function ----------------------------
print("\n== headline IV through svc.fetch_current_atm_iv (flat sigma 16%) ==")
strikes = list(range(int(S * 0.95) + 1, int(S * 1.05) + 1))  # $1 strikes strictly inside +-5%
for label, types in (("calls+puts", ("call", "put")), ("calls only", ("call",)), ("puts only", ("put",))):
    info, log = run(make_snaps(strikes, types, lambda K: sig0))
    print(f"  {label:11s}: iv={info['iv']*100:.3f}%  bias={(info['iv']-sig0)*1e4:+.0f} bp  n={info['n_contracts']}  method={info['method']}")

# unbalanced: calls everywhere, puts only at the 5 nearest strikes
snaps = make_snaps(strikes, ("call",), lambda K: sig0)
snaps.update(make_snaps([K for K in strikes if abs(K - S) <= 2], ("put",), lambda K: sig0))
info, _ = run(snaps)
print(f"  calls all + 5 ATM puts: bias={(info['iv']-sig0)*1e4:+.0f} bp n={info['n_contracts']}")

# --- realistic skew: does the cancellation survive? ------------------------
print("\n== headline IV with linear skew (ATM 16%, -0.8 vol pt per +1% moneyness) ==")
skew = lambda K: 0.16 - 0.8 * (K / S - 1.0)
info, _ = run(make_snaps(strikes, ("call", "put"), skew))
true_med = float(np.median([skew(K) for K in strikes]))
print(f"  calls+puts: iv={info['iv']*100:.3f}%  vs ATM 16.000% -> {(info['iv']-0.16)*1e4:+.0f} bp ; vs true band-median {true_med*100:.3f}% -> {(info['iv']-true_med)*1e4:+.0f} bp  n={info['n_contracts']}")

# --- low-vol regime: do ITM puts fall out of the arbitrage bounds? --------
print("\n== low-vol regimes: contracts dropped by implied_vol_call bounds (r=q=0 synthetic call) ==")
for sig in (0.16, 0.12, 0.10, 0.08):
    info, _ = run(make_snaps(strikes, ("call", "put"), lambda K: sig))
    print(f"  sigma={sig*100:.0f}%: iv={info['iv']*100:.3f}% bias={(info['iv']-sig)*1e4:+.0f} bp  n_used={info['n_contracts']}/{2*len(strikes)}")

# --- sensitivity to T (60 DTE) and to r ------------------------------------
print("\n== sensitivity ==")
for dte in (15, 30, 60):
    T_ = dte / 365.0
    K = 640
    c = bs(S, K, T_, R, Q, sig0, "call"); p = bs(S, K, T_, R, Q, sig0, "put")
    ivc = implied_vol_call(c, S, K, T_, 0.0, 0.0); ivp = implied_vol_call(p + S - K, S, K, T_, 0.0, 0.0)
    print(f"  dte={dte}: ATM call {(ivc-sig0)*1e4:+.0f} bp, ATM put {(ivp-sig0)*1e4:+.0f} bp")
