"""p4 skeptic probe — ask_only_and_crossed_mids: realistic impact of _snapshot_mid
fallbacks on the DISPLAYED median, and regression risk of the proposed strict fix.

Uses the real _snapshot_mid and implied_vol_call from the repo.
"""
import math
import sys
import numpy as np

sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
from app.model.calibration.implied_vol import implied_vol_call, bs_call_price  # noqa: E402
from app.model.iv_dashboard.service import _snapshot_mid  # noqa: E402

S, sigma, dte = 640.0, 0.16, 30
T = dte / 365.0
K = 640.0
c_true = bs_call_price(S, K, T, 0.0, 0.0, sigma)   # r=q=0 to isolate the quote effect
vega = (bs_call_price(S, K, T, 0.0, 0.0, sigma + 0.01) - c_true)
print(f"ATM call {c_true:.3f}  vega {vega:.3f} $/vol-pt")

print("(1) what _snapshot_mid returns for each quote shape:")
cases = {
    "two-sided 5.10/5.20": {"latestQuote": {"bp": 5.10, "ap": 5.20}},
    "ask-only bid=0 ask=5.20": {"latestQuote": {"bp": 0.0, "ap": 5.20}},
    "crossed 5.30/5.10": {"latestQuote": {"bp": 5.30, "ap": 5.10}},
    "both zero + trade 5.05": {"latestQuote": {"bp": 0.0, "ap": 0.0}, "latestTrade": {"p": 5.05}},
    "both zero, no trade": {"latestQuote": {"bp": 0.0, "ap": 0.0}},
    "no quote at all": {},
}
for name, snap in cases.items():
    print(f"   {name:28s} -> {_snapshot_mid(snap)}")

print("(2) ask-only bias at realistic in-session spreads (SPY ~$0.05-0.15, AAPL ~$0.10-0.30):")
for spread in (0.05, 0.10, 0.30, 1.00):
    iv = implied_vol_call(c_true + spread / 2, S, K, T, 0.0, 0.0)
    print(f"   spread ${spread:.2f}: +{(iv - sigma) * 1e4:.0f} bp")

print("(3) after-hours: ALL quotes bid=0/ask=0, latestTrade from earlier at spot S-3 (stale):")
# last trade happened when spot was 637; we invert it against today's close 640
S_then = 637.0
strikes = [float(k) for k in range(608, 673)]
ivs = []
for Kx in strikes:
    c_then = bs_call_price(S_then, Kx, T, 0.0, 0.0, sigma)
    p_then = c_then - S_then + Kx
    ivc = implied_vol_call(c_then, S, Kx, T, 0.0, 0.0)
    ivp = implied_vol_call(p_then + S - Kx, S, Kx, T, 0.0, 0.0)
    for v in (ivc, ivp):
        if v is not None and np.isfinite(v) and 0 < v < 5:
            ivs.append(v)
ivs = np.array(ivs)
print(f"   usable {len(ivs)}/130, per-contract range {(ivs.min() - sigma) * 1e4:+.0f}..{(ivs.max() - sigma) * 1e4:+.0f} bp, median {(np.median(ivs) - sigma) * 1e4:+.0f} bp")

print("(4) proposed strict fix (bid>0, ask>=bid, spread ratio cap, NO fallback) on the after-hours book:")
n_keep = 0
for snap in [{"latestQuote": {"bp": 0.0, "ap": 0.0}, "latestTrade": {"p": 5.05}}] * 130:
    q = snap["latestQuote"]
    if q["bp"] > 0 and q["ap"] >= q["bp"]:
        n_keep += 1
print(f"   contracts kept: {n_keep}/130 -> service returns None -> tab shows 'IV courante indisponible'")
