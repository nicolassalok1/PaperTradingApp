"""p4 repro — percentile-label-vs-effective-window.

Claim: `extra_days` (service.py L521) only covers the RV warm-up; when the requested
percentile window exceeds the number of RV rows fetched, the last percentile is ranked among
all available rows while the view still labels it "Percentile (<requested> j)" (tab L190).

Independent oracle:
- network stubbed with the REAL AAPL trading calendar (Stooq file, holidays included), dates
  shifted so the last bar is today -> realistic bar counts per calendar span;
- effective points = number of non-NaN RV rows that the rolling window could see at the last
  row, cross-checked by recomputing the last percentile by hand on exactly that many rows and
  asserting equality with the service output (so the number is not an estimate);
- the label string is rebuilt the way the view builds it.
Offline, deterministic.
"""
from __future__ import annotations

import sys

sys.path.insert(
    0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca"
)

import numpy as np
import pandas as pd

from app.model.iv_dashboard import analytics as A
from app.model.iv_dashboard import service as S

ROOT = r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca"
CSV = ROOT + "/cache/OHLC/stooq_aapl.us_start_end_d.csv"

raw = pd.read_csv(CSV, parse_dates=["date"]).set_index("date").sort_index()["close"].astype(float)
today = pd.Timestamp.now().normalize()
shift = today - raw.index[-1]
full = pd.Series(raw.values, index=raw.index + shift)  # real trading-day gaps, ends today

captured = {}


def stub_fetch(sym, *, years=2.0, extra_days=60):
    lookback_days = int(float(years) * 365.25) + max(0, int(extra_days))
    start = today - pd.Timedelta(days=lookback_days)
    c = full[full.index >= start]
    captured["n_closes"] = len(c)
    captured["extra_days"] = extra_days
    return pd.DataFrame({"Date": c.index, "Close": c.values}), "stub", ["stub"]


S.fetch_daily_closes = stub_fetch
S.load_iv_history = lambda sym: pd.DataFrame(columns=["date", "iv"])


def main() -> None:
    print("UI reachable: Durée in {1,2,3,5} ans ; Fenêtre percentile 60..756 (default 252)")
    hdr = f"{'years':>5} {'pwin':>5} {'closes':>7} {'rv_rows':>8} {'eff_pts':>8} {'label shown':>20} {'match':>6} {'chk':>4}"
    print(hdr)
    for years in (1.0, 2.0, 3.0, 5.0):
        for pwin in (60, 252, 504, 756):
            out = S.get_iv_dashboard_data("AAPL", years=years, percentile_window=pwin, include_current_iv=False)
            # Recompute what the service did, from the same stubbed closes
            df, _, _ = stub_fetch("AAPL", years=years, extra_days=captured["extra_days"])
            rv = A.compute_realized_vol(df.set_index("Date")["Close"], 20).dropna()
            eff = min(pwin, len(rv))
            # hand oracle: percentile of last RV among the last `eff` RV rows (pandas rank pct, avg ties)
            tail = rv.tail(eff)
            hand = float(tail.rank(pct=True).iloc[-1])
            svc = float(out["current_percentile"])
            ok = np.isclose(hand, svc, atol=1e-12)
            label = f"Percentile ({out['percentile_window']} j)"  # tab_iv_dashboard.py L190
            match = "yes" if eff == pwin else "NO"
            print(f"{years:>5} {pwin:>5} {captured['n_closes']:>7} {len(rv):>8} {eff:>8} {label:>20} {match:>6} {'ok' if ok else 'FAIL':>4}")
    print(f"\nextra_days used by service for rv_window=20: {captured['extra_days']} calendar days")

    # Direct pandas oracle: rolling(756, min_periods=60) on a 265-row series == rank among 265 rows
    rng = np.random.default_rng(0)
    s = pd.Series(rng.lognormal(-1.8, 0.3, 265))
    roll = A.compute_percentile_series(s, 756).iloc[-1]
    plain = float(s.rank(pct=True).iloc[-1])
    print(f"\npandas oracle: rolling(756,min_periods=60).rank(pct) last = {roll:.6f} ; rank among all 265 = {plain:.6f} ; equal={np.isclose(roll, plain)}")

    # Also: the IV-vs-RV percentile (service L569) uses series_df AFTER the cutoff -> even fewer rows
    out = S.get_iv_dashboard_data("AAPL", years=1.0, percentile_window=756, include_current_iv=False)
    print(f"1 an / 756 : series rows after cutoff (what .tail(756) would see for the IV percentile) = {len(out['series'])}")


if __name__ == "__main__":
    main()
