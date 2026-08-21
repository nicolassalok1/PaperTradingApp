"""Probe: percentile semantics (rank pct vs percentile_within, min_periods, window alignment)
and forward construction (partial means, look-ahead confinement)."""
import sys
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
import numpy as np, pandas as pd
from app.model.iv_dashboard import analytics as A

rng = np.random.default_rng(7)
idx = pd.bdate_range("2024-01-01", periods=400)
vol = pd.Series(0.15 + 0.05 * rng.standard_normal(400).cumsum() * 0.1, index=idx).abs()

# --- A. rolling rank(pct=True) vs percentile_within on the SAME window incl. current value
pct = A.compute_percentile_series(vol, 252, min_periods=60)
i = 300
win = vol.iloc[i - 251 : i + 1]          # 252-pt window incl. current
cur = vol.iloc[i]
pw_incl = A.percentile_within(win, cur)
pw_excl = A.percentile_within(vol.iloc[i - 251 : i], cur)
print(f"A. rank(pct) = {pct.iloc[i]:.6f} | percentile_within(incl current) = {pw_incl:.6f} | excl current = {pw_excl:.6f}")
# Exact relation: rank(pct, avg) = (below + 0.5*(ties-1) + 1)/n ; percentile_within = below/n + 0.5*ties/n
n = len(win); below = (win < cur).sum(); ties = (win == cur).sum()
print(f"   n={n} below={below} ties={ties}: rank_pct formula {(below + 0.5*(ties-1) + 1)/n:.6f}, within formula {(below + 0.5*ties)/n:.6f}, diff = {(0.5)/n:.6f}")

# --- B. Ties: what does rank(pct) give on all-equal window
allties = pd.Series(np.full(100, 0.2))
print("B. all-ties rank(pct) last:", allties.rolling(100, min_periods=2).rank(pct=True).iloc[-1], "| percentile_within:", A.percentile_within(allties, 0.2))
# min vs max of window
mono = pd.Series(np.arange(100, dtype=float))
print("   monotonic inc: last =", mono.rolling(100).rank(pct=True).iloc[-1], " first-of-window min value rank =", mono.rolling(100).rank(pct=True).iloc[-1] - 0)
print("   rank(pct) of the MIN in a 252 window =", pd.Series(np.r_[np.arange(1,252.0), 0.0]).rolling(252).rank(pct=True).iloc[-1], "(never 0); percentile_within(min)=", A.percentile_within(pd.Series(np.arange(1,252.0)), 0.0))

# --- C. min_periods=60: day 60/61 of history -> percentile within 60 pts
short = vol.iloc[:70]
p_short = A.compute_percentile_series(short, 252, min_periods=60)
print("C. first non-NaN percentile index position:", int(p_short.notna().values.argmax()), "(0-based) | legacy strict-252 would be NaN until position 251")
print("   value at pos 59:", p_short.iloc[59], " = rank among", 60, "pts")

# --- D. Service trailing window for IV percentile: series_df['vol'].tail(252) vs rolling window of RV pct
# series_df is cut at `cutoff` (years) AFTER pct computed on the full (warm-up-inclusive) series.
# So the RV rolling pct at the last row uses the last 252 RV values (incl current); tail(252) of series_df uses the same 252 values as long as series_df has >= 252 rows.
print("D. trailing vs rolling: both include the current row; tail(252) of series_df == last 252 rv values iff len(series_df)>=252 ->", len(vol.tail(252)) == 252)

# --- E. Forward construction: which rows are partial means?
v = pd.Series(np.arange(1.0, 101.0), index=pd.bdate_range("2024-01-01", periods=100))
fw = 30
fwd = v.rolling(fw, min_periods=1).mean().shift(-fw)
# rolling mean at position j uses v[j-29..j]; partial when j<29; shift(-30) moves value at j to j-30
# so row i receives mean at j=i+30: partial iff i+30 < 29 -> never (i>=0). Last fw rows -> NaN.
print("E. fwd NaN count (last rows):", int(fwd.isna().sum()), "| first row fwd:", fwd.iloc[0], "= mean(v[1..31]) =", v.iloc[1:31].mean())
print("   any partial mean among non-NaN forward rows?", any((v.rolling(fw, min_periods=1).count().shift(-fw) < fw).dropna()))
# BUT: rolling mean at j is BACKWARD-looking: row i's 'forward vol' = mean(v[i+1 .. i+30]) -> true forward. OK.
print("   row i forward = mean(v[i+1..i+30]) check:", np.isclose(fwd.iloc[10], v.iloc[11:41].mean()))
# With dropna() inside analyze_forward_vol on a gappy series: forward is computed on ROW offsets, not calendar.
gap = vol.copy(); gap.iloc[100:160] = np.nan
res = A.analyze_forward_vol(gap)
print("   gappy series: n rows in regression =", res["reg_forward"]["n"], "(forward uses row offsets across the 60-row gap)")

# --- F. look-ahead confined? current_vol/current_percentile/regime come from series_df, analysis only feeds charts/log.
print("F. non mesuré ici — vérifié par lecture: service.py L541-545 (current_*) ne lisent pas `analysis`; analysis construit L583.")
