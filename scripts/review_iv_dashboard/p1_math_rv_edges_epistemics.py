"""Probe: compute_realized_vol edge cases (zero close, duplicate dates), annualization convention effect,
VRP epistemics of 'Percentile IV vs série RV', and regime-split intersection stability on realistic RV."""
import sys, warnings
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
import numpy as np, pandas as pd
from app.model.iv_dashboard import analytics as A

rng = np.random.default_rng(11)
idx = pd.bdate_range("2024-01-01", periods=300)
clean = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=idx)

# --- 1. one zero close in the middle
bad = clean.copy(); bad.iloc[150] = 0.0
rv_clean = A.compute_realized_vol(clean, 20)
rv_bad = A.compute_realized_vol(bad, 20)
print("1. zero close: len(rv_clean)=", len(rv_clean), "len(rv_bad)=", len(rv_bad), "(returns dropped, not NaN-propagated)")
print("   NaN count clean/bad:", int(rv_clean.isna().sum()), int(rv_bad.isna().sum()))
aligned = pd.concat([rv_clean.rename("clean"), rv_bad.rename("bad")], axis=1)
diff = (aligned["clean"] - aligned["bad"]).abs()
print("   rows where RV differs (>1e-12):", int((diff > 1e-12).sum()), "| max abs diff:", float(diff.max()), "| rows missing in bad:", int(aligned["bad"].isna().sum() - aligned["clean"].isna().sum()))
print("   is the RV on the day of the zero close present?", idx[150] in rv_bad.index, "| day after?", idx[151] in rv_bad.index)
# The true jump across the bad close (t-1 -> t+1) is lost: window now spans 22 rows with 20 returns.

# --- 2. negative close
neg = clean.copy(); neg.iloc[150] = -5.0
print("2. negative close -> same handling as zero:", A.compute_realized_vol(neg, 20).equals(rv_bad))

# --- 3. duplicate dates in closes
dup = pd.concat([clean.iloc[:100], clean.iloc[99:100], clean.iloc[100:]])
print("3. duplicate index: len", len(dup), "unique", dup.index.nunique())
try:
    rv_dup = A.compute_realized_vol(dup, 20)
    print("   compute_realized_vol OK, len", len(rv_dup), "| zero return injected at dup?", float(np.log(dup / dup.shift(1)).iloc[100]))
    pct_dup = A.compute_percentile_series(rv_dup, 252)
    df = pd.DataFrame({"close": dup, "vol": rv_dup, "vol_percentile": pct_dup})
    print("   service-style DataFrame build OK, len", len(df))
except Exception as e:
    print("   RAISE", type(e).__name__, e)

# --- 4. annualization convention: RV sqrt(252) vs IV T=dte/365
# Same option price inverted with T=30/365 vs T=21/252 (30 calendar days ~ 21 trading days)
from app.model.calibration.implied_vol import implied_vol_call
from math import log, sqrt, exp
from scipy.stats import norm
S0, K, r = 100.0, 100.0, 0.0
sigma_true = 0.20
T_cal = 30 / 365.0
d1 = (log(S0 / K) + 0.5 * sigma_true**2 * T_cal) / (sigma_true * sqrt(T_cal)); d2 = d1 - sigma_true * sqrt(T_cal)
price = S0 * norm.cdf(d1) - K * norm.cdf(d2)
iv_cal = implied_vol_call(price, S0, K, T_cal, r, 0.0)
iv_trd = implied_vol_call(price, S0, K, 21 / 252.0, r, 0.0)
print(f"4. same price inverted: T=30/365 -> IV {iv_cal:.5f} | T=21/252 -> IV {iv_trd:.5f} | diff {abs(iv_cal-iv_trd)*100:.3f} vol pts ({abs(iv_cal/iv_trd-1)*100:.2f}% rel)")

# --- 5. Epistemics: SPY-like. RV distribution 12-18%, IV 16-20%.
rv_hist = pd.Series(rng.normal(0.15, 0.02, 252)).clip(0.08, 0.30)
for iv in (0.16, 0.18, 0.20):
    p = A.percentile_within(rv_hist, iv)
    reg = A.classify_regime(p)
    print(f"5. RV~N(15%,2%), IV={iv:.0%}: 'Percentile IV vs série RV' = {p:.1%} -> regime {reg['label']} / Signal (IV) = {reg['signal_label']}")
# lognormal-ish RV with realistic right tail
rv_hist2 = pd.Series(np.exp(rng.normal(np.log(0.14), 0.3, 252)))
print("   RV lognormal(14%, 0.3): quantiles 25/50/75/90 =", [round(float(rv_hist2.quantile(q)), 3) for q in (0.25, 0.5, 0.75, 0.9)])
for iv in (0.16, 0.18, 0.20):
    p = A.percentile_within(rv_hist2, iv)
    print(f"   IV={iv:.0%}: percentile {p:.1%} -> {A.classify_regime(p)['signal_label']}")

# --- 6. intersection stability on realistic (OU log-vol) RV series: how often outside data range?
def synth_rv(seed, n=520):
    r = np.random.default_rng(seed)
    lv = np.empty(n); lv[0] = np.log(0.15)
    for t in range(1, n):
        lv[t] = lv[t-1] + 0.05 * (np.log(0.15) - lv[t-1]) + 0.1 * r.normal()
    rets = np.exp(lv) / np.sqrt(252) * r.normal(size=n)
    px = pd.Series(100 * np.exp(np.cumsum(rets)), index=pd.bdate_range("2023-01-01", periods=n))
    return A.compute_realized_vol(px, 20).dropna()

out_of_range = 0; empties = 0; slopes = []; inters = []
N = 200
for seed in range(N):
    rv = synth_rv(seed)
    res = A.analyze_forward_vol(rv)
    df = res["df"]
    slopes.append(res["reg_forward"]["slope"]); inters.append(res["intersection"])
    if not (df["current_vol"].min() <= res["intersection"] <= df["current_vol"].max()):
        out_of_range += 1
    if res["n_high"] == 0 or res["n_low"] == 0:
        empties += 1
print(f"6. {N} synthetic mean-reverting RV series: slope range [{min(slopes):.3f}, {max(slopes):.3f}] | intersection outside data range: {out_of_range} | empty regime: {empties}")
