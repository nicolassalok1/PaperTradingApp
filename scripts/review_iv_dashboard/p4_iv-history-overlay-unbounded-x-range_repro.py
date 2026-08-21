"""
p4 skeptic repro — finding `iv-history-overlay-unbounded-x-range`.

Independent harness: the IV-history cache file is produced by the REAL
`record_iv_observation` (with `dt.date.today` patched to walk through past
days), read back by the REAL `load_iv_history` (no date filter?), then the REAL
`_render_series_chart` is called with `st.plotly_chart` intercepted so the
figure can be inspected. We measure the x-extent of every trace, the share of
the autoscaled x-axis actually covered by the RV series, and the overlay mode.

Scenarios:
  S0 — all observations inside the window (control)
  S1 — finder-style: one observation 2023-03-01 + 5 recent, window "1 an"
  S2 — realistic: weekly analyses 14..12.5 months ago, then today, window "1 an"
  S3 — realistic at default "2 ans": weekly analyses 26..24.5 months ago, then today

Run: .venv/Scripts/python.exe scripts/review_iv_dashboard/p4_iv-history-overlay-unbounded-x-range_repro.py
"""
from __future__ import annotations

import datetime as dt
import json
import socket
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _blocked(*a, **k):  # noqa: ANN002, ANN003
    raise RuntimeError("network forbidden in probe")


socket.socket.connect = _blocked  # type: ignore[method-assign]
socket.create_connection = _blocked  # type: ignore[assignment]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.model.iv_dashboard import analytics as ivx  # noqa: E402
from app.model.iv_dashboard import service as svc  # noqa: E402
from app.vue.tabs import tab_iv_dashboard as tab  # noqa: E402

TODAY = pd.Timestamp.today().normalize()


class _FakeDate(dt.date):
    _today = dt.date.today()

    @classmethod
    def today(cls):
        return cls._today


def _write_history(symbol: str, days: list[pd.Timestamp], tmp: Path) -> None:
    svc.CACHE_IV_HISTORY_DIR = tmp
    orig_date = svc.dt.date
    try:
        svc.dt.date = _FakeDate  # type: ignore[attr-defined]
        for i, d in enumerate(days):
            _FakeDate._today = d.date()
            svc.record_iv_observation(
                symbol, {"iv": 0.18 + 0.01 * (i % 4), "dte": 30, "n_contracts": 6, "method": "x", "spot": 100.0}
            )
    finally:
        svc.dt.date = orig_date


def _series(years: float) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = int(years * 252) + 60
    closes = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.011, n))),
        index=pd.bdate_range(end=TODAY, periods=n),
    )
    rv = ivx.compute_realized_vol(closes, 20)
    df = pd.DataFrame({"close": closes, "vol": rv}).dropna()
    cutoff = TODAY - pd.Timedelta(days=int(years * 365.25))
    return df[df.index >= cutoff]


def _capture_fig():
    box = {}

    def fake_plotly_chart(fig, *a, **k):  # noqa: ANN001, ANN002, ANN003
        box["fig"] = fig

    tab.st.plotly_chart = fake_plotly_chart
    return box


def measure(series: pd.DataFrame, hist: pd.DataFrame):
    box = _capture_fig()
    tab._render_series_chart(
        {"series": series, "rv_window": 20, "iv_history": hist, "current_iv": None}
    )
    fig = box["fig"]
    xs_all, overlay = [], None
    for tr in fig.data:
        x = pd.to_datetime(pd.Series(list(tr.x)))
        xs_all.append((x.min(), x.max()))
        if tr.name == "IV ATM (historique local)":
            overlay = tr
    xmin = min(a for a, _ in xs_all)
    xmax = max(b for _, b in xs_all)
    axis_days = (xmax - xmin).days
    rv_days = (series.index[-1] - series.index[0]).days
    n_out = int((hist["date"] < series.index[0]).sum()) if not hist.empty else 0
    gaps = hist["date"].diff().dt.days.dropna() if len(hist) > 1 else pd.Series(dtype=float)
    return {
        "rv_series": f"{series.index[0].date()} -> {series.index[-1].date()} ({rv_days} d)",
        "hist_first_obs": None if hist.empty else str(hist["date"].min().date()),
        "n_hist_obs": int(len(hist)),
        "n_hist_obs_before_window": n_out,
        "x_axis_extent_days": axis_days,
        "rv_share_of_x_axis_pct": round(100.0 * rv_days / axis_days, 1) if axis_days else None,
        "overlay_mode": None if overlay is None else overlay.mode,
        "overlay_connectgaps": None if overlay is None else overlay.connectgaps,
        "largest_gap_between_connected_obs_days": None if gaps.empty else int(gaps.max()),
        "xaxis_range_set_explicitly": fig.layout.xaxis.range,
    }


def main():
    out = {"load_iv_history_filters_by_date": "load_iv_history(symbol) takes no date/window argument"}
    recent5 = list(pd.bdate_range(end=TODAY, periods=5))

    # S0 control: all inside window
    tmp0 = Path(tempfile.mkdtemp(prefix="p4_ivx0_"))
    _write_history("S0", recent5, tmp0)
    out["S0_control_inside_window"] = measure(_series(1.0), svc.load_iv_history("S0"))

    # S1 finder-style
    tmp1 = Path(tempfile.mkdtemp(prefix="p4_ivx1_"))
    _write_history("S1", [pd.Timestamp("2023-03-01")] + recent5, tmp1)
    out["S1_finder_style_1y_plus_2023_obs"] = measure(_series(1.0), svc.load_iv_history("S1"))

    # S2 realistic: weekly 14..12.5 months ago then today, window 1 an
    tmp2 = Path(tempfile.mkdtemp(prefix="p4_ivx2_"))
    old = list(pd.bdate_range(end=TODAY - pd.Timedelta(days=380), periods=8, freq="W-WED"))
    _write_history("S2", old + [TODAY], tmp2)
    out["S2_realistic_1y_after_14_months_use"] = measure(_series(1.0), svc.load_iv_history("S2"))

    # S3 realistic at default 2 ans
    tmp3 = Path(tempfile.mkdtemp(prefix="p4_ivx3_"))
    old2 = list(pd.bdate_range(end=TODAY - pd.Timedelta(days=745), periods=8, freq="W-WED"))
    _write_history("S3", old2 + [TODAY], tmp3)
    out["S3_realistic_2y_after_26_months_use"] = measure(_series(2.0), svc.load_iv_history("S3"))

    # S4: in-window but sparse monthly use (no x-range issue, only the connecting segments)
    tmp4 = Path(tempfile.mkdtemp(prefix="p4_ivx4_"))
    monthly = list(pd.bdate_range(end=TODAY, periods=6, freq="BMS"))
    _write_history("S4", monthly, tmp4)
    out["S4_inside_window_monthly_use"] = measure(_series(1.0), svc.load_iv_history("S4"))

    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
