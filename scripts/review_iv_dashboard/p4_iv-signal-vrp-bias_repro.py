"""p4 repro — iv-signal-vrp-bias.

Question: does `classify_regime(percentile_within(RV_tail, IV))` (service.py L569-571) fire
"MEAN REVERSION ↓ ATTENDUE" under an ordinary, *constant* volatility-risk premium, i.e. is
the "Signal (IV)" chip structurally biased rather than informative?

Independent oracle: real AAPL daily closes cached offline (Stooq, 1984-2026), the repo's own
`compute_realized_vol` / `percentile_within` / `classify_regime`. For every day of the last
~10 years we place a hypothetical IV at RV_today + k pts (constant VRP, no information about
IV richness at all) and count how often the chip says "down". If a *constant* premium fires
the chip most of the time, the chip does not measure IV richness.

Also: a sanity check of the GBM numbers quoted by the finder (sigma = 18 %, N = 252).
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

ROOT = r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca"
CSV = ROOT + "/cache/OHLC/stooq_aapl.us_start_end_d.csv"

PWIN = A.DEFAULT_PERCENTILE_WINDOW  # 252 — what the service passes to .tail()
RVW = A.DEFAULT_RV_WINDOW  # 20


def main() -> None:
    df = pd.read_csv(CSV, parse_dates=["date"]).set_index("date").sort_index()
    closes = df["close"].astype(float)
    rv = A.compute_realized_vol(closes, RVW).dropna()
    rv = rv[rv.index >= "2016-01-01"]  # ~10y of modern AAPL dynamics
    print(f"AAPL RV({RVW}) sample: {len(rv)} days, {rv.index[0].date()} -> {rv.index[-1].date()}")
    print(f"RV median={rv.median():.3%}  q10={rv.quantile(.1):.3%}  q90={rv.quantile(.9):.3%}")

    # --- Part A: constant additive VRP (IV = RV_today + k pts) ---------------------------
    print("\nA) IV = RV_today + k pts  (constant premium; chip should carry NO information)")
    print(f"{'k pts':>6} {'P(sig=down)':>12} {'P(sig=up)':>10} {'P(neutral)':>11} {'median pct':>11}")
    days = rv.index[PWIN:]  # need a full trailing window behind each day
    for k in (0.0, 0.02, 0.03, 0.04, 0.05, 0.08):
        sigs = []
        pcts = []
        for d in days:
            loc = rv.index.get_loc(d)
            trailing = rv.iloc[loc - PWIN + 1 : loc + 1]  # same as series_df['vol'].tail(252)
            iv = float(rv.iloc[loc]) + k
            p = A.percentile_within(trailing, iv)
            pcts.append(p)
            sigs.append(A.classify_regime(p)["signal_key"])
        s = pd.Series(sigs)
        print(
            f"{k*100:>5.0f}  {(s=='down').mean():>11.1%} {(s=='up').mean():>9.1%} "
            f"{(s=='neutral').mean():>10.1%} {np.median(pcts):>10.1%}"
        )

    # --- Part B: constant multiplicative VRP (IV = RV_today * m) -------------------------
    print("\nB) IV = RV_today * m")
    print(f"{'m':>6} {'P(sig=down)':>12} {'P(sig=up)':>10} {'median pct':>11}")
    for m in (1.0, 1.15, 1.25, 1.4):
        sigs = []
        pcts = []
        for d in days:
            loc = rv.index.get_loc(d)
            trailing = rv.iloc[loc - PWIN + 1 : loc + 1]
            p = A.percentile_within(trailing, float(rv.iloc[loc]) * m)
            pcts.append(p)
            sigs.append(A.classify_regime(p)["signal_key"])
        s = pd.Series(sigs)
        print(f"{m:>6.2f} {(s=='down').mean():>11.1%} {(s=='up').mean():>9.1%} {np.median(pcts):>10.1%}")

    # --- Part C: contrast — a true IV percentile on IV's own history (legacy construct) ----
    # Under a constant premium, IV's own 252d percentile == RV's own percentile, i.e. the
    # chip would fire 'down' only when RV itself is in its top quintile (~20 % of days).
    print("\nC) Contrast: percentile of (RV+k) within ITS OWN trailing history (legacy IV-rank construct)")
    for k in (0.03,):
        ivs = rv + k
        own = []
        for d in days:
            loc = ivs.index.get_loc(d)
            trailing = ivs.iloc[loc - PWIN + 1 : loc + 1]
            own.append(A.classify_regime(A.percentile_within(trailing.iloc[:-1], float(ivs.iloc[loc])))["signal_key"])
        s = pd.Series(own)
        print(f"k={k*100:.0f} pts  P(down)={(s=='down').mean():.1%}  P(up)={(s=='up').mean():.1%}")

    # --- Part D: check finder's GBM numbers (sigma=18 %, 252 pts, IV = median RV + k) --------
    print("\nD) GBM sigma=18 % (finder's setup), median over 200 seeds")
    res = {k: [] for k in (0.02, 0.03, 0.04, 0.05)}
    for seed in range(200):
        rng = np.random.default_rng(seed)
        n = PWIN + RVW + 1
        px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.18 / np.sqrt(252), n))))
        rvs = A.compute_realized_vol(px, RVW).dropna().tail(PWIN)
        med = float(rvs.median())
        for k in res:
            res[k].append(A.percentile_within(rvs, med + k))
    for k, v in res.items():
        print(f"IV = median RV + {k*100:.0f} pts -> pct median {np.median(v):.3f} (q10 {np.quantile(v,.1):.3f}, q90 {np.quantile(v,.9):.3f})")


if __name__ == "__main__":
    main()
