"""
Phase-4 impact probe (G2_epistemics) — offline, real AAPL closes from cache/OHLC.

1. iv-signal-vrp-bias: how often would the « Signal (IV) » chip say
   "MEAN REVERSION ↓ ATTENDUE" if IV sits at a realistic premium over RV20?
2. percentile-label-vs-effective-window: effective points per (years, pwin)
   combo through the real service path (fetch stubbed), and the numeric gap
   between a 252-pt percentile and the true 756-pt one on real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.model.iv_dashboard import analytics as ivx  # noqa: E402
from app.model.iv_dashboard import service as svc  # noqa: E402

csv = ROOT / "cache" / "OHLC" / "stooq_aapl.us_start_end_d.csv"
raw = pd.read_csv(csv, parse_dates=["date"]).sort_values("date")
closes = raw.set_index("date")["close"].astype(float)
closes = closes[closes.index >= "2015-01-01"]
print(f"AAPL closes: {len(closes)} rows, {closes.index[0].date()} -> {closes.index[-1].date()}")

rv = ivx.compute_realized_vol(closes, 20).dropna()

# ---------------------------------------------------------------- 1. VRP bias
print("\n=== 1. Signal (IV) chip firing frequency, IV proxy = f(RV20_t), trailing 252 RV ===")
PWIN = 252
idx = rv.index[PWIN:]
trail = {t: rv.loc[:t].tail(PWIN) for t in idx}


def fire_rate(iv_fn):
    down = up = neutral = 0
    for t in idx:
        p = ivx.percentile_within(trail[t], iv_fn(float(rv.loc[t])))
        k = ivx.classify_regime(p)["signal_key"]
        if k == "down":
            down += 1
        elif k == "up":
            up += 1
        else:
            neutral += 1
    n = len(idx)
    return down / n, neutral / n, up / n


print(f"{'IV proxy':<28} {'down(MR↓)':>10} {'neutral':>9} {'up(MR↑)':>9}")
for lbl, fn in [
    ("IV = RV (baseline)", lambda r: r),
    ("IV = RV x1.10", lambda r: r * 1.10),
    ("IV = RV x1.20", lambda r: r * 1.20),
    ("IV = RV x1.30", lambda r: r * 1.30),
    ("IV = RV + 2 pts", lambda r: r + 0.02),
    ("IV = RV + 4 pts", lambda r: r + 0.04),
    ("IV = RV + 6 pts", lambda r: r + 0.06),
]:
    d, nn, u = fire_rate(fn)
    print(f"{lbl:<28} {d:>10.1%} {nn:>9.1%} {u:>9.1%}")

# Conditional: on CALM days (RV20 below its own trailing median), how often does the chip fire?
print("\n  conditional on calm days (RV20 <= trailing-252 median of RV20):")
for lbl, fn in [("IV = RV x1.20", lambda r: r * 1.20), ("IV = RV + 4 pts", lambda r: r + 0.04)]:
    calm = [t for t in idx if float(rv.loc[t]) <= float(trail[t].median())]
    d = np.mean(
        [ivx.classify_regime(ivx.percentile_within(trail[t], fn(float(rv.loc[t]))))["signal_key"] == "down" for t in calm]
    )
    print(f"  {lbl:<26} calm days={len(calm)}  MR↓ fires on {d:.1%} of them")

# ------------------------------------------------- 2. effective percentile window
print("\n=== 2. Effective percentile window through get_iv_dashboard_data (fetch stubbed) ===")
END = closes.index[-1]


def _stub_fetch(symbol, *, years=2.0, extra_days=60):
    lookback = int(float(years) * 365.25) + max(0, int(extra_days))
    start = END - pd.Timedelta(days=lookback)
    df = closes[closes.index >= start].rename("Close").reset_index().rename(columns={"date": "Date"})
    return df, "stub", [f"{len(df)} barres (stub)"]


svc.fetch_daily_closes = _stub_fetch
# service uses Timestamp.now() for the cutoff; anchor to data end so the trim matches the stub
_real_now = pd.Timestamp.now


class _FakeTs(pd.Timestamp):
    @classmethod
    def now(cls, tz=None):  # noqa: D401
        return END


svc.pd = type("pdshim", (), {})()
for name in dir(pd):
    if not name.startswith("_"):
        setattr(svc.pd, name, getattr(pd, name))
svc.pd.Timestamp = _FakeTs
svc.pd.Timedelta = pd.Timedelta
svc.pd.DataFrame = pd.DataFrame
svc.pd.notna = pd.notna

# full-history "true" percentile series for comparison
rv_full = ivx.compute_realized_vol(closes, 20)
print(f"{'years':>5} {'pwin':>5} {'label':>12} {'eff pts':>8} {'shown pct':>10} {'true pct(pwin)':>15} {'delta':>7}")
for years, pwin in [(2.0, 252), (1.0, 252), (1.0, 504), (1.0, 756), (2.0, 504), (2.0, 756), (3.0, 756), (5.0, 756)]:
    out = svc.get_iv_dashboard_data("AAPL", years=years, rv_window=20, percentile_window=pwin, include_current_iv=False)
    s = out["series"]
    eff = int(s["vol"].tail(pwin).shape[0])
    shown = float(out["current_percentile"])
    true_p = float(rv_full.rolling(pwin, min_periods=pwin).rank(pct=True).iloc[-1])
    print(f"{years:>5} {pwin:>5} {'Percentile ('+str(pwin)+' j)':>12} {eff:>8} {shown:>10.3f} {true_p:>15.3f} {shown-true_p:>+7.3f}")

# distribution of |pct_252 - pct_756| over history: how misleading is a 1-year rank sold as 3-year?
p252 = rv_full.rolling(252, min_periods=252).rank(pct=True)
p756 = rv_full.rolling(756, min_periods=756).rank(pct=True)
both = pd.concat([p252, p756], axis=1, keys=["p252", "p756"]).dropna()
gap = (both["p252"] - both["p756"]).abs()
reg_changes = np.mean(
    [ivx.classify_regime(a)["key"] != ivx.classify_regime(b)["key"] for a, b in zip(both["p252"], both["p756"])]
)
sig_changes = np.mean(
    [ivx.classify_regime(a)["signal_key"] != ivx.classify_regime(b)["signal_key"] for a, b in zip(both["p252"], both["p756"])]
)
print(
    f"\n|pct252 - pct756| over {len(both)} days: median={gap.median():.3f} p90={gap.quantile(0.9):.3f} max={gap.max():.3f}"
    f"\nregime bucket differs on {reg_changes:.1%} of days; mean-reversion signal differs on {sig_changes:.1%} of days"
)

# effective points for 1 an / 252 on real calendar (holidays): label 252 vs actual
out = svc.get_iv_dashboard_data("AAPL", years=1.0, rv_window=20, percentile_window=252, include_current_iv=False)
print(f"\n1 an / 252 j : series rows = {len(out['series'])} (label says 252)")
