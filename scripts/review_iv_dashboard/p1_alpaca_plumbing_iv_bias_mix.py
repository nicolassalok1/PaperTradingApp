"""Probe: median bias of the BS-inversion path depending on the call/put mix actually quoted (offline)."""
from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.model.calibration.implied_vol import bs_call_price, implied_vol_call  # noqa: E402

def bs_put(S, K, t, r, q, vol):
    return bs_call_price(S, K, t, r, q, vol) - S*math.exp(-q*t) + K*math.exp(-r*t)

def code_iv(kind, mid, S, K, T):
    cp = mid if kind == "call" else mid + S - K
    iv = implied_vol_call(cp, S, K, T, 0.0, 0.0)
    return float(iv) if iv is not None and np.isfinite(iv) and 0 < iv < 5 else None

S, r, q, sigma, T = 640.0, 0.04, 0.013, 0.16, 30/365
band = 0.05
strikes = [k for k in np.arange(600, 681, 1.0) if abs(k/S-1) <= band]
def run(sel):
    ivs = []
    for K in strikes:
        if sel(K, "call"):
            v = code_iv("call", bs_call_price(S, K, T, r, q, sigma), S, K, T)
            if v is not None: ivs.append(v)
        if sel(K, "put"):
            v = code_iv("put", bs_put(S, K, T, r, q, sigma), S, K, T)
            if v is not None: ivs.append(v)
    return len(ivs), float(np.median(ivs))
for name, sel in [
    ("all calls+puts in band", lambda K, t: True),
    ("OTM only (calls K>=S, puts K<=S)", lambda K, t: (K >= S) if t == "call" else (K <= S)),
    ("calls only", lambda K, t: t == "call"),
    ("puts only", lambda K, t: t == "put"),
    ("calls all + puts only 5 nearest ATM", lambda K, t: t == "call" or abs(K-S) <= 2),
]:
    n, med = run(sel)
    print(f"{name:40} n={n:4d} median IV={med*100:.3f}%  bias={(med-sigma)*1e4:+.0f} bp")
