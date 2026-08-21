"""
Orchestrator probe — §4.2 IV methodology bias of service.fetch_current_atm_iv.

Replays the code's inversion pipeline (service.py L400-418) on synthetic
Black-Scholes prices generated with realistic r/q and reports the IV error in bp.
Offline, deterministic. Run with .venv/Scripts/python.exe.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model.calibration.implied_vol import implied_vol_call  # noqa: E402


def bs(S, K, T, r, q, sig, kind):
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def code_pipeline_iv(mid, S, K, T, kind, r_annual=0.0):
    """Exact replica of service.py L412-417."""
    if kind == "call":
        call_price = float(mid)
    else:
        call_price = float(mid) + S - K * math.exp(-r_annual * T)
    return implied_vol_call(call_price, S, K, T, r_annual, 0.0)


S, r, q, sig, dte = 640.0, 0.04, 0.013, 0.16, 30
T = dte / 365.0
print(f"SPY-like: S={S} r={r:.1%} q={q:.1%} true sigma={sig:.0%} dte={dte} T={T:.4f}")
print("\n=== A. parity q=0 / r=0 bias (bp of vol) by moneyness, per type ===")
print(f"{'K/S':>6} {'K':>7} | {'call mid':>9} {'IV_call':>8} {'err bp':>7} | {'put mid':>9} {'IV_put':>8} {'err bp':>7}")
rows = []
for m in (0.95, 0.97, 0.99, 1.00, 1.01, 1.03, 1.05):
    K = round(S * m)
    c = bs(S, K, T, r, q, sig, "call")
    p = bs(S, K, T, r, q, sig, "put")
    ivc = code_pipeline_iv(c, S, K, T, "call")
    ivp = code_pipeline_iv(p, S, K, T, "put")
    ec = (ivc - sig) * 1e4 if np.isfinite(ivc) else float("nan")
    ep = (ivp - sig) * 1e4 if np.isfinite(ivp) else float("nan")
    rows.append((m, ivc, ivp))
    print(f"{m:>6.2f} {K:>7.0f} | {c:>9.3f} {ivc:>8.4f} {ec:>7.0f} | {p:>9.3f} {ivp:>8.4f} {ep:>7.0f}")

print("\n=== B. headline = np.median over calls AND puts in the ±5% band (service L437) ===")
ivs = [iv for _, ivc, ivp in rows for iv in (ivc, ivp) if np.isfinite(iv)]
print(f"median IV = {np.median(ivs):.4f}  -> error vs true {sig:.2%}: {(np.median(ivs)-sig)*1e4:+.0f} bp  (n={len(ivs)})")
ivs_atm = [iv for m, ivc, ivp in rows if abs(m - 1) < 0.015 for iv in (ivc, ivp) if np.isfinite(iv)]
print(f"median IV (|K/S-1|<1.5%) = {np.median(ivs_atm):.4f} -> {(np.median(ivs_atm)-sig)*1e4:+.0f} bp (n={len(ivs_atm)})")

print("\n=== C. sensitivity of the parity bias to r (q fixed 1.3%) and to q (r fixed 4%) — ATM put only ===")
K = S
for rr in (0.0, 0.02, 0.04, 0.05):
    p = bs(S, K, T, rr, q, sig, "put")
    print(f"  r={rr:.0%}: put-implied IV via code = {code_pipeline_iv(p, S, K, T, 'put'):.4f} ({(code_pipeline_iv(p, S, K, T, 'put')-sig)*1e4:+.0f} bp)")
for qq in (0.0, 0.013, 0.03):
    p = bs(S, K, T, r, qq, sig, "put")
    c = bs(S, K, T, r, qq, sig, "call")
    print(f"  q={qq:.1%}: put {code_pipeline_iv(p, S, K, T, 'put'):.4f} ({(code_pipeline_iv(p, S, K, T, 'put')-sig)*1e4:+.0f} bp) | call {code_pipeline_iv(c, S, K, T, 'call'):.4f} ({(code_pipeline_iv(c, S, K, T, 'call')-sig)*1e4:+.0f} bp)")

print("\n=== D. ask-only mid (service L220-221: bid missing/0 -> ask) — ATM call, true mid c, spread s ===")
c = bs(S, S, T, r, q, sig, "call")
for spread in (0.02, 0.10, 0.30, 1.00):
    ask = c + spread / 2
    iv_ask = code_pipeline_iv(ask, S, S, T, "call")
    iv_mid = code_pipeline_iv(c, S, S, T, "call")
    print(f"  spread ${spread:.2f}: IV(ask)-IV(mid) = {(iv_ask-iv_mid)*1e4:+.0f} bp   (ATM call mid ≈ {c:.2f})")

print("\n=== E. crossed quote accepted? bid=5.50 ask=5.00 -> _snapshot_mid returns 5.25 (no bid<=ask check) ===")
from app.model.iv_dashboard.service import _snapshot_mid  # noqa: E402
print("  _snapshot_mid({'latestQuote': {'bp': 5.5, 'ap': 5.0}}) =", _snapshot_mid({"latestQuote": {"bp": 5.5, "ap": 5.0}}))
print("  _snapshot_mid({'latestQuote': {'bp': 0.0, 'ap': 5.2}}) =", _snapshot_mid({"latestQuote": {"bp": 0.0, "ap": 5.2}}))
print("  _snapshot_mid({'latestQuote': {'bp': 0.0, 'ap': 0.0}, 'latestTrade': {'p': 4.0}}) =", _snapshot_mid({"latestQuote": {"bp": 0.0, "ap": 0.0}, "latestTrade": {"p": 4.0}}))

print("\n=== F. T = max(dte,1)/365 (L400): intraday time ignored — IV error at 16:00 ET vs 09:30 ET for dte=30 ===")
# true price generated with T_true = (30 - 0.27)/365 at close-ish (6.5h of 24h elapsed), inverted with T=30/365
T_true = (30 - 6.5 / 24) / 365.0
c_true = bs(S, S, T_true, 0.0, 0.0, sig, "call")
iv_code = implied_vol_call(c_true, S, S, 30 / 365.0, 0.0, 0.0)
print(f"  IV error from ignoring 6.5h elapsed on dte=30: {(iv_code-sig)*1e4:+.0f} bp ; on dte=15: ", end="")
T_true15 = (15 - 6.5 / 24) / 365.0
c15 = bs(S, S, T_true15, 0.0, 0.0, sig, "call")
print(f"{(implied_vol_call(c15, S, S, 15/365.0, 0.0, 0.0)-sig)*1e4:+.0f} bp")

print("\n=== G. 252 vs 365 — is there an RV/IV annualisation inconsistency? ===")
print("  RV = std(daily log ret) * sqrt(252): annual vol per *trading* year (252 trading days = 1 calendar year).")
print("  BS T = dte/365 calendar: annual vol per calendar year. Both are 'per year' => same unit; NOT an inconsistency.")
print("  (Mixing would only matter if T were expressed in trading days/252 AND RV in calendar days: not the case.)")
