"""p4 repro (part 2) — which contracts are dropped by the r=q=0 inversion in a low-vol regime,
and does implying the forward from parity (config-free fix) remove the bias."""
from __future__ import annotations
import math, sys
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
import numpy as np
from scipy.stats import norm
from app.model.calibration.implied_vol import implied_vol_call

S, R, Q, T = 640.0, 0.04, 0.013, 30 / 365.0

def bs(S, K, T, r, q, sig, typ):
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if typ == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

strikes = list(range(609, 672))
for sig in (0.12, 0.10):
    dropped = []
    for K in strikes:
        p = bs(S, K, T, R, Q, sig, "put")
        cp = p + S - K                      # service.py L416, r_annual = 0
        iv = implied_vol_call(cp, S, K, T, 0.0, 0.0)
        if not np.isfinite(iv):
            dropped.append((K, round(cp, 3), round(max(S - K, 0.0), 3)))
        c = bs(S, K, T, R, Q, sig, "call")
        if not np.isfinite(implied_vol_call(c, S, K, T, 0.0, 0.0)):
            dropped.append(("call", K))
    print(f"sigma={sig:.0%}: dropped puts (K, synthetic call, intrinsic r=q=0): {dropped}")

# config-free fix: forward implied from ATM parity, then invert on the forward (Black-76 via S'=F, r=q=0)
print("\n== fix check: F implied from ATM put-call parity, invert with S'=F, r=q=0 ==")
sig = 0.16
K0 = 640
F_impl = K0 + (bs(S, K0, T, R, Q, sig, "call") - bs(S, K0, T, R, Q, sig, "put")) * math.exp(R * T)
F_true = S * math.exp((R - Q) * T)
print(f"  F_true={F_true:.4f}  F_impl={F_impl:.4f}")
for K in (608, 640, 672):
    c = bs(S, K, T, R, Q, sig, "call"); p = bs(S, K, T, R, Q, sig, "put")
    # undiscounted prices on the forward (Black-76 with r=0)
    c_f = c * math.exp(R * T); p_f = p * math.exp(R * T)
    ivc = implied_vol_call(c_f, F_impl, K, T, 0.0, 0.0)
    ivp = implied_vol_call(p_f + F_impl - K, F_impl, K, T, 0.0, 0.0)
    print(f"  K={K}: call {(ivc-sig)*1e4:+.1f} bp, put {(ivp-sig)*1e4:+.1f} bp")
