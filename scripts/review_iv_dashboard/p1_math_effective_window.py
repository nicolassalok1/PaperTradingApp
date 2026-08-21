"""Probe: effective number of points behind the displayed 'Percentile (N j)' for each (years, percentile_window)
combination reachable from the UI (durations 1/2/3/5 ans, window 60..756). Network stubbed with a
synthetic close series that covers exactly the lookback the service requests."""
import sys
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
import datetime as dt
import numpy as np, pandas as pd
from app.model.iv_dashboard import service as S, analytics as A

rng = np.random.default_rng(3)
today = pd.Timestamp.now().normalize()
full_idx = pd.bdate_range(end=today, periods=3000)
full_close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(full_idx)))), index=full_idx)

def stub_fetch(sym, years=2.0, extra_days=60):
    lookback_days = int(float(years) * 365.25) + max(0, int(extra_days))
    start = today - pd.Timedelta(days=lookback_days)
    c = full_close[full_close.index >= start]
    return pd.DataFrame({"Date": c.index, "Close": c.values}), "stub", ["stub"]

S.fetch_daily_closes = stub_fetch
S.load_iv_history = lambda sym: pd.DataFrame(columns=["date", "iv"])

print(f"{'years':>5} {'pwin':>5} {'rv_rows_total':>13} {'rows_displayed':>14} {'eff_pts_last_pct':>16} {'first_pct_row_nan':>17}")
for years in (1.0, 2.0, 3.0, 5.0):
    for pwin in (60, 252, 504, 756):
        out = S.get_iv_dashboard_data("ZZZ", years=years, percentile_window=pwin, include_current_iv=False)
        # recompute what the service did to count effective points at the last row
        df, _, _ = stub_fetch("ZZZ", years=years, extra_days=int(20 * 1.6) + 15)
        closes = df.set_index("Date")["Close"]
        rv = A.compute_realized_vol(closes, 20)
        eff = int(rv.dropna().tail(pwin).shape[0])
        series = out["series"]
        n_nan_pct = int(series["vol_percentile"].isna().sum())
        print(f"{years:>5} {pwin:>5} {int(rv.notna().sum()):>13} {len(series):>14} {eff:>16} {n_nan_pct:>17}")
