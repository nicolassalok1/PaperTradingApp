"""p4 repro — ask_only_and_crossed_mids.

1) Call the REAL service._snapshot_mid on one-sided / crossed / zero quotes.
2) Measure the IV bias of using ask instead of mid through service.fetch_current_atm_iv
   (monkeypatched, no network) for an ATM SPY 30 DTE chain with bid=0 (pre-market style).
"""
from __future__ import annotations
import datetime as dt, math, sys
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
import numpy as np
from scipy.stats import norm
from app.model.iv_dashboard import service as svc
from app.model.calibration.implied_vol import implied_vol_call

print("== _snapshot_mid on degenerate quotes ==")
cases = {
    "bid=0, ask=5.20":            {"latestQuote": {"bp": 0, "ap": 5.20}},
    "bid missing, ask=5.20":      {"latestQuote": {"ap": 5.20}},
    "bid=None, ask=5.20":         {"latestQuote": {"bp": None, "ap": 5.20}},
    "crossed bid=5.30 ask=5.10":  {"latestQuote": {"bp": 5.30, "ap": 5.10}},
    "locked bid=ask=5.20":        {"latestQuote": {"bp": 5.20, "ap": 5.20}},
    "bid=0 ask=0, trade p=4.00":  {"latestQuote": {"bp": 0, "ap": 0}, "latestTrade": {"p": 4.00}},
    "no quote, trade p=4.00":     {"latestTrade": {"p": 4.00}},
    "bid=5.10 ask=0":             {"latestQuote": {"bp": 5.10, "ap": 0}},
    "bid=0.01 ask=50.0 (wide)":   {"latestQuote": {"bp": 0.01, "ap": 50.0}},
}
for name, snap in cases.items():
    print(f"  {name:28s} -> mid={svc._snapshot_mid(snap)}")

# --- IV bias of ask-only at ATM, through the real function ------------------
S, T0 = 640.0, 30 / 365.0
sig = 0.16
TODAY = dt.date.today()
EXPIRY = TODAY + dt.timedelta(days=30)

def bs_call(S, K, T, sig):
    d1 = (math.log(S / K) + 0.5 * sig * sig * T) / (sig * math.sqrt(T)); d2 = d1 - sig * math.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2)
def bs_put(S, K, T, sig):
    return bs_call(S, K, T, sig) - S + K
def opra(K, typ):
    return f"SPY{EXPIRY:%y%m%d}{'C' if typ=='call' else 'P'}{int(round(K*1000)):08d}"

def run(snaps):
    svc.fetch_spot_price = lambda sym: S
    svc._fetch_atm_snapshots = lambda *a, **k: snaps
    info, _ = svc.fetch_current_atm_iv("SPY")
    return info

d1 = (0.5 * sig * sig * T0) / (sig * math.sqrt(T0))
vega = S * norm.pdf(d1) * math.sqrt(T0)
print(f"\nATM vega = {vega:.3f} $/vol pt  (1 $ of price = {100/vega:.0f} bp)")

strikes = list(range(609, 672))
print("== headline IV via fetch_current_atm_iv, r=q=0 market (so only the quote effect is measured) ==")
for spread in (0.10, 0.30, 0.50, 1.00):
    snaps_mid, snaps_ask = {}, {}
    for K in strikes:
        for typ in ("call", "put"):
            px = bs_call(S, K, T0, sig) if typ == "call" else bs_put(S, K, T0, sig)
            snaps_mid[opra(K, typ)] = {"latestQuote": {"bp": px - spread / 2, "ap": px + spread / 2}}
            snaps_ask[opra(K, typ)] = {"latestQuote": {"bp": 0, "ap": px + spread / 2}}
    i_mid = run(snaps_mid); i_ask = run(snaps_ask)
    print(f"  spread ${spread:.2f}: two-sided mid -> {(i_mid['iv']-sig)*1e4:+.0f} bp ; bid=0/ask-only -> {(i_ask['iv']-sig)*1e4:+.0f} bp  (n={i_ask['n_contracts']})")

# crossed market end-to-end: bid > ask by $0.20 everywhere -> accepted?
snaps = {}
for K in strikes:
    for typ in ("call", "put"):
        px = bs_call(S, K, T0, sig) if typ == "call" else bs_put(S, K, T0, sig)
        snaps[opra(K, typ)] = {"latestQuote": {"bp": px + 0.10, "ap": px - 0.10}}
i = run(snaps)
print(f"  crossed by $0.20 everywhere: accepted, iv bias {(i['iv']-sig)*1e4:+.0f} bp, n={i['n_contracts']} (no contract rejected)")
