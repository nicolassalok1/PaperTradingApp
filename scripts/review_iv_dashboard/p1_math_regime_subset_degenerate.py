"""Probe: can a regime subset with >10 identical x (RV=0 from a halt) kill an otherwise valid analysis?
Also: what does get_iv_dashboard_data surface as analysis_error (service.py L586) — stubbed network."""
import sys, warnings
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
import numpy as np, pandas as pd
from app.model.iv_dashboard import analytics as A

idx = pd.bdate_range("2023-01-01", periods=600)
rng = np.random.default_rng(1)
found = None
for seed in range(200):
    rng = np.random.default_rng(seed)
    # halted 60 trading days (RV=0 for 40 rows after 20-day warmup) then mean-reverting noisy closes
    n_halt = 60
    n_live = 400
    # OU log-vol process -> heteroskedastic returns -> RV mean-reverts (slope<1, intercept>0)
    lv = np.empty(n_live); lv[0] = np.log(0.3)
    for t in range(1, n_live):
        lv[t] = lv[t-1] + 0.1 * (np.log(0.3) - lv[t-1]) + 0.15 * rng.normal()
    rets = np.exp(lv) / np.sqrt(252) * rng.normal(size=n_live)
    closes = np.concatenate([np.full(n_halt, 10.0), 10.0 * np.exp(np.cumsum(rets))])
    px = pd.Series(closes, index=idx[: len(closes)])
    rv = A.compute_realized_vol(px, 20).dropna()
    try:
        res = A.analyze_forward_vol(rv)
    except ValueError as e:
        found = (seed, str(e), int((rv == 0).sum()), rv.nunique())
        break
print("first seed raising ValueError inside analyze_forward_vol:", found)

# Deterministic construction: zeros + a block where all non-zero RV values sit above the intersection
v = pd.Series(np.concatenate([np.zeros(40), rng.uniform(0.30, 0.50, 120)]), index=idx[:160])
try:
    res = A.analyze_forward_vol(v, forward_window=10)
    print("deterministic: OK intersection=", res["intersection"], "n_low", res["n_low"], "n_high", res["n_high"], "reg_low", res["reg_low"] is not None)
except ValueError as e:
    print("deterministic: RAISE", e)

# Now: service-level wiring (monkeypatch network)
from app.model.iv_dashboard import service as S
df_closes = pd.DataFrame({"Date": idx[:160], "Close": np.concatenate([np.full(60, 10.0), 10.0 * np.exp(np.cumsum(rng.normal(0, 0.025, 100)))])})
S.fetch_daily_closes = lambda sym, years=2.0, extra_days=60: (df_closes, "stub", ["stub"])
S.fetch_current_atm_iv = lambda sym: (None, ["stub iv none"])
S.load_iv_history = lambda sym: pd.DataFrame(columns=["date", "iv"])
out = S.get_iv_dashboard_data("ZZZ", years=2.0, include_current_iv=False)
print("service analysis is None:", out["analysis"] is None, "| analysis_error:", out["analysis_error"])
print("log tail:", out["log"][-1])
