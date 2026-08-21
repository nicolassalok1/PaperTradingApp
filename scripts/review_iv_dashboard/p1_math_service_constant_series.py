"""Probe: what the service/view surface when the RV series is constant (tick-bound closes), network stubbed."""
import sys
sys.path.insert(0, r"C:/Users/Nathalie Asus/Dev/PaperTradingApp-fix/.claude/worktrees/feature+iv-dashboard-alpaca")
import numpy as np, pandas as pd
from app.model.iv_dashboard import service as S

end = pd.Timestamp.now().normalize()
idx = pd.bdate_range(end=end, periods=400)
closes = np.where(np.arange(400) % 2 == 0, 10.00, 10.01)  # SPAC-like / tick-bound
df_closes = pd.DataFrame({"Date": idx, "Close": closes})
S.fetch_daily_closes = lambda sym, years=2.0, extra_days=60: (df_closes, "stub", ["stub"])
S.fetch_current_atm_iv = lambda sym: (None, ["stub iv none"])
S.load_iv_history = lambda sym: pd.DataFrame(columns=["date", "iv"])
out = S.get_iv_dashboard_data("ZZZ", years=1.0, include_current_iv=False)
print("series len:", len(out["series"]), "vol nunique:", out["series"]["vol"].nunique())
print("current_vol:", out["current_vol"], "current_percentile:", out["current_percentile"], "regime:", out["regime"])
print("analysis is None:", out["analysis"] is None)
print("analysis_error (displayed in st.info):", repr(out["analysis_error"]))
print("log tail:", out["log"][-1])

# Also: fully constant closes -> RV = 0 everywhere -> percentile of ties
closes2 = np.full(400, 10.0)
S.fetch_daily_closes = lambda sym, years=2.0, extra_days=60: (pd.DataFrame({"Date": idx, "Close": closes2}), "stub", ["stub"])
out2 = S.get_iv_dashboard_data("ZZZ", years=1.0, include_current_iv=False)
print("const closes: current_vol", out2["current_vol"], "pct", out2["current_percentile"], "regime", out2["regime"]["label"], "signal", out2["regime"]["signal_label"])
print("analysis_error:", repr(out2["analysis_error"]))
