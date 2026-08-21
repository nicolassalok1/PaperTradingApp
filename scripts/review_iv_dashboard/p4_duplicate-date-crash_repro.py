"""p4 repro — duplicate-date-crash (offline, deterministic).

Replays service.get_iv_dashboard_data L532-536 with a closes DataFrame containing
one duplicated Date (as fetch_daily_closes would return it: sorted, no dedup).
  (1) same close on the duplicate -> return 0 injected ? and DataFrame build -> ValueError ?
  (2) different close on the duplicate -> what return is injected ?
  (3) duplicate at the LAST row (Yahoo 'today twice' pattern) -> same crash ?
  (4) full service path with fetch_daily_closes monkeypatched (no network) -> what does
      the tab receive ? (RuntimeError/ValueError propagates -> view shows the raw message)
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from app.model.iv_dashboard import analytics as A
from app.model.iv_dashboard import service as S

def make_closes(n=300, dup_pos=150, dup_close_delta=0.0, seed=0):
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=1), periods=n)
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Date": idx, "Close": px})
    dup = df.iloc[[dup_pos]].copy(); dup["Close"] += dup_close_delta
    df = pd.concat([df, dup]).sort_values("Date", kind="stable").reset_index(drop=True)
    return df

def replay_service_tail(closes_df, label):
    closes = closes_df.set_index("Date")["Close"]
    print(f"\n[{label}] rows={len(closes)} unique dates={closes.index.nunique()}")
    rets = A.compute_log_returns(closes)
    dup_date = closes.index[closes.index.duplicated()][0]
    print(f"  returns at duplicated date {dup_date.date()}: {rets.loc[[dup_date]].round(6).tolist()}")
    rv = A.compute_realized_vol(closes, 20)
    pct = A.compute_percentile_series(rv, 252)
    print(f"  compute_realized_vol OK (len={len(rv)}), percentile OK (len={len(pct)})")
    try:
        series_df = pd.DataFrame({"close": closes, "vol": rv, "vol_percentile": pct})
        print(f"  DataFrame build OK -> len={len(series_df)}")
    except Exception as exc:
        print(f"  DataFrame build RAISES {type(exc).__name__}: {exc}")

replay_service_tail(make_closes(dup_close_delta=0.0), "1: dup mid, same close")
replay_service_tail(make_closes(dup_close_delta=1.0), "2: dup mid, close +1.0")
replay_service_tail(make_closes(dup_pos=299), "3: dup on last row (Yahoo 'today twice')")

# (4) full service path, offline
df_dup = make_closes(dup_pos=299)
S.fetch_daily_closes = lambda sym, **kw: (df_dup, "fake", ["301 barres daily reçues via fake."])
try:
    out = S.get_iv_dashboard_data("FAKE", include_current_iv=False)
    print("\n[4] service OK ?!", out["series"].shape)
except Exception as exc:
    print(f"\n[4] get_iv_dashboard_data RAISES {type(exc).__name__}: {exc}")
    print("    -> not caught inside the service (only analyze_forward_vol is try/except'ed, L583-588);")
    print("       propagates to the controller/tab as an opaque pandas message.")
