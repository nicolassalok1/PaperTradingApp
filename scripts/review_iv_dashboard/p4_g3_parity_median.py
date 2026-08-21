"""p4 skeptic probe — parity-r-q-zero-bias: impact on the DISPLAYED median.

Replicates service.py L404-419 exactly (r_annual=0, q=0 on spot) on a synthetic
SPY-like ATM band ($1 strikes, both types every strike), then:
  (a) per-contract bias (sanity vs phase-1 numbers),
  (b) median over the +-5% band (balanced set),
  (c) median when one contract / one side is missing (bimodal fragility),
  (d) a parameter-free fix: parity-implied forward from the nearest ATM pair.
"""
import math
import sys
import numpy as np

sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
from app.model.calibration.implied_vol import implied_vol_call, bs_call_price  # noqa: E402

S, r, q, sigma, dte = 640.0, 0.04, 0.013, 0.16, 30
T = dte / 365.0


def true_call(K):
    return bs_call_price(S, K, T, r, q, sigma)


def true_put(K):
    return true_call(K) - S * math.exp(-q * T) + K * math.exp(-r * T)


def service_iv(kind, mid, K, r_annual=0.0):
    # service.py L412-417 verbatim
    if kind == "call":
        call_price = float(mid)
    else:
        call_price = float(mid) + S - K * math.exp(-r_annual * T)
    return implied_vol_call(call_price, S, K, T, r_annual, 0.0)


strikes = [float(k) for k in range(int(S * 0.95), int(S * 1.05) + 1)]  # 608..672 -> 65 strikes
contracts = []
for K in strikes:
    contracts.append(("call", K, service_iv("call", true_call(K), K)))
    contracts.append(("put", K, service_iv("put", true_put(K), K)))

print("(a) per-contract bias vs true 16%:")
for K in (608.0, 640.0, 672.0):
    c = [x for x in contracts if x[1] == K]
    print(f"   K/S={K/S:.3f}: " + ", ".join(f"{k} {(iv - sigma) * 1e4:+.0f} bp" for k, _, iv in c))

ivs_all = np.array([iv for _, _, iv in contracts])
ivs_c = np.array([iv for k, _, iv in contracts if k == "call"])
ivs_p = np.array([iv for k, _, iv in contracts if k == "put"])
print(f"(b) n={len(ivs_all)} balanced median bias: {(np.median(ivs_all) - sigma) * 1e4:+.1f} bp"
      f" | calls-only {(np.median(ivs_c) - sigma) * 1e4:+.0f} bp | puts-only {(np.median(ivs_p) - sigma) * 1e4:+.0f} bp")

# (c) bimodal fragility: drop ONE put (e.g. quote unusable) -> median jumps?
srt = np.sort(ivs_all)
mid_i = len(srt) // 2
print(f"    sorted middle values (bp): {[round((v - sigma) * 1e4) for v in srt[mid_i - 2: mid_i + 2]]}")
drop1 = np.array([iv for i, (k, _, iv) in enumerate(contracts) if i != 1])
print(f"(c) drop one put  -> median bias {(np.median(drop1) - sigma) * 1e4:+.0f} bp (n={len(drop1)})")
drop1c = np.array([iv for i, (k, _, iv) in enumerate(contracts) if i != 0])
print(f"    drop one call -> median bias {(np.median(drop1c) - sigma) * 1e4:+.0f} bp")
band = [iv for k, K, iv in contracts if abs(K / S - 1) <= 0.015]
print(f"    |K/S-1|<=1.5% (n={len(band)}) median bias {(np.median(band) - sigma) * 1e4:+.0f} bp")

# what does the error do to the displayed spread IV-RV (RV=13.12% from the live run)?
rv = 0.1312
print(f"    displayed spread IV-RV true {((sigma - rv)) * 100:+.2f} pts ; calls-only {((np.median(ivs_c) - rv)) * 100:+.2f} ; puts-only {((np.median(ivs_p) - rv)) * 100:+.2f}")

# (d) parameter-free fix: forward from ATM parity pair, then invert on F with r=q=0
K0 = min(strikes, key=lambda k: abs(k - S))
F = K0 + true_call(K0) - true_put(K0)  # undiscounted parity, r unknown
print(f"(d) parity-implied F={F:.3f} (true F={S * math.exp((r - q) * T):.3f})")
fixed = []
for k, K, _ in contracts:
    px = true_call(K) if k == "call" else true_put(K) + F - K
    fixed.append(implied_vol_call(px, F, K, T, 0.0, 0.0))
fixed = np.array(fixed)
print(f"    fixed per-contract bias range: {(fixed.min() - sigma) * 1e4:+.0f} .. {(fixed.max() - sigma) * 1e4:+.0f} bp ; median {(np.median(fixed) - sigma) * 1e4:+.1f} bp")
print(f"    fixed calls-only {(np.median(fixed[::2]) - sigma) * 1e4:+.0f} bp ; puts-only {(np.median(fixed[1::2]) - sigma) * 1e4:+.0f} bp")

# (e) the finding's fix with a WRONG q (q=0 for SPY) and with r left at default 0
wrong = []
for k, K, _ in contracts:
    rr, qq = 0.04, 0.0
    px = true_call(K) if k == "call" else true_put(K) + S * math.exp(-qq * T) - K * math.exp(-rr * T)
    wrong.append(implied_vol_call(px, S, K, T, rr, qq))
wrong = np.array(wrong)
print(f"(e) finding's fix with r=4%, q=0 (SPY q forgotten): calls-only {(np.median(wrong[::2]) - sigma) * 1e4:+.0f} bp ; puts-only {(np.median(wrong[1::2]) - sigma) * 1e4:+.0f} bp")
