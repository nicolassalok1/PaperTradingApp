"""P4 skeptic probe (G2_epistemics): independent oracle for
 - percentile-label-vs-effective-window : count of non-NaN RV points inside the last rolling window
   (pandas rolling(...).count(), not a reconstruction) for every UI-reachable (years, pct_window).
 - iv-signal-vrp-bias : IV offset above a constant-sigma GBM RV distribution that flips Signal (IV)
   to 'down' ; also on a lognormal RV distribution.
Network stubbed. No tracked file modified."""
import sys
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
import numpy as np, pandas as pd
from app.model.iv_dashboard import service as S, analytics as A

rng = np.random.default_rng(11)
today = pd.Timestamp.now().normalize()
full_idx = pd.bdate_range(end=today, periods=4000)
SIG = 0.18
full_close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, SIG/np.sqrt(252), len(full_idx)))), index=full_idx)

def stub_fetch(sym, years=2.0, extra_days=60):
    lookback_days = int(float(years) * 365.25) + max(0, int(extra_days))
    start = today - pd.Timedelta(days=lookback_days)
    c = full_close[full_close.index >= start]
    return pd.DataFrame({"Date": c.index, "Close": c.values}), "stub", ["stub"]

S.fetch_daily_closes = stub_fetch
S.load_iv_history = lambda sym: pd.DataFrame(columns=["date", "iv"])

print("== A. effective points in the last rolling percentile window (label = pwin) ==")
print(f"{'years':>5} {'pwin':>5} {'rv_rows_fetched':>15} {'count_in_last_win':>17} {'iv_tail_pts(post-cutoff)':>24} {'label_ok':>8}")
for years in (1.0, 2.0, 3.0, 5.0):
    for pwin in (60, 252, 504, 756):
        df, _, _ = stub_fetch("ZZZ", years=years, extra_days=int(20 * 1.6) + 15)
        closes = df.set_index("Date")["Close"]
        rv = A.compute_realized_vol(closes, 20)
        cnt = int(rv.rolling(pwin, min_periods=60).count().iloc[-1])
        out = S.get_iv_dashboard_data("ZZZ", years=years, percentile_window=pwin, include_current_iv=False)
        iv_tail = int(out["series"]["vol"].tail(pwin).shape[0])
        assert out["percentile_window"] == pwin
        print(f"{years:>5} {pwin:>5} {int(rv.notna().sum()):>15} {cnt:>17} {iv_tail:>24} {str(cnt >= pwin):>8}")

print("\n== B. Signal (IV) flip on constant-sigma GBM (sigma=18%), pwin=252, years=2 ==")
out = S.get_iv_dashboard_data("ZZZ", years=2.0, percentile_window=252, include_current_iv=False)
trailing = out["series"]["vol"].tail(252)
med = float(trailing.median())
print(f"RV median={med:.4f} q80={trailing.quantile(0.8):.4f} q90={trailing.quantile(0.9):.4f}")
for off in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
    iv = med + off
    p = A.percentile_within(trailing, iv)
    r = A.classify_regime(p)
    print(f"IV = median + {off*100:.0f} pts = {iv:.4f} -> pct {p:.3f} -> {r['label']} / {r['signal_label']}")

print("\n== C. Lognormal RV (median 15%, sdlog 0.2), IV fixed at 18%/20% ==")
rv_ln = pd.Series(np.exp(rng.normal(np.log(0.15), 0.2, 252)))
for iv in (0.18, 0.20):
    p = A.percentile_within(rv_ln, iv)
    print(f"IV={iv:.2f}: pct {p:.3f} -> {A.classify_regime(p)['signal_label']}")

print("\n== D. label string ==")
print(repr(A.classify_regime(0.3)["label"]), repr(A.classify_regime(0.7)["label"]))
