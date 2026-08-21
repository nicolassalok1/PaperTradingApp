"""
Pure analytics for the 🌡️ Vol Implicite dashboard (no I/O, no UI, no network).

Ported from the legacy Tkinter "Implied Volatility Trading Dashboard" (IB/TWS):
- rolling percentile of a vol series -> volatility regime classification
- 30d forward vol regressions (mean reversion vs momentum)
- regime-split regressions (high vs low vol) around the y=x intersection

Differences vs the legacy script (intentional fixes):
- No `* sqrt(252)` rescaling of an already-annualized implied vol (legacy bug).
  Realized vol is annualized here from daily log returns with `sqrt(252)`.
- The forward-vol regression sample only drops rows lacking current/forward vol
  (the legacy version also dropped the percentile warm-up rows).
- The legacy ranked the IV inside its own 252-day IV history. Alpaca has no IV
  history, so the service ranks the current IV inside the *realized-vol*
  distribution instead; that rank is reported as-is (variance risk premium) and
  `classify_regime` is NOT applied to it — no mean-reversion signal on the IV.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS_PER_YEAR = 252
DEFAULT_RV_WINDOW = 20
DEFAULT_FORWARD_WINDOW = 30
DEFAULT_PERCENTILE_WINDOW = 252
DEFAULT_PERCENTILE_MIN_PERIODS = 60
MIN_ANALYSIS_POINTS = 30
MIN_REGIME_POINTS = 10  # legacy: strictly more than 10 points per regime


def compute_log_returns(closes: pd.Series) -> pd.Series:
    """Daily log returns from a close series (index preserved, first row dropped)."""
    px = pd.to_numeric(closes, errors="coerce").astype(float)
    px = px.where(px > 0)
    return np.log(px / px.shift(1)).dropna()


def compute_realized_vol(
    closes: pd.Series,
    window: int = DEFAULT_RV_WINDOW,
    *,
    annualization: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Rolling realized volatility, annualized (e.g. 0.18 = 18%).

    `window` is in trading days; std uses the pandas default ddof=1.
    """
    window = max(2, int(window))
    rets = compute_log_returns(closes)
    if rets.empty:
        return pd.Series(dtype=float)
    return rets.rolling(window=window, min_periods=window).std() * np.sqrt(float(annualization))


def compute_percentile_series(
    vol: pd.Series,
    window: int = DEFAULT_PERCENTILE_WINDOW,
    *,
    min_periods: int = DEFAULT_PERCENTILE_MIN_PERIODS,
) -> pd.Series:
    """
    Rolling percentile rank (0..1) of the latest vol value within its trailing window.

    Same construct as the legacy dashboard (`rolling(252).rank(pct=True)`), with a
    `min_periods` relaxation so shorter histories still produce a value.
    """
    v = pd.to_numeric(vol, errors="coerce").astype(float)
    window = max(2, int(window))
    min_periods = max(2, min(int(min_periods), window))
    return v.rolling(window=window, min_periods=min_periods).rank(pct=True)


def percentile_within(history: pd.Series, value: float) -> float:
    """
    Percentile (0..1) of `value` within the distribution of `history`
    (mean of {< value} plus half the ties, i.e. a mid-rank percentile).
    Returns NaN when inputs are unusable.
    """
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(val):
        return float("nan")
    h = pd.to_numeric(history, errors="coerce").dropna().astype(float)
    if h.empty:
        return float("nan")
    below = float((h < val).mean())
    ties = float((h == val).mean())
    return below + 0.5 * ties


def classify_regime(percentile: float) -> Dict[str, str]:
    """
    Map a percentile (0..1) to the legacy regime buckets + mean-reversion signal.

    Returns {"key", "label", "signal_key", "signal_label"}; keys are stable
    identifiers for the view ("high"/"above"/"normal"/"below"/"low"/"unknown"
    and "down"/"up"/"neutral"/"unknown").
    """
    try:
        p = float(percentile)
    except (TypeError, ValueError):
        p = float("nan")

    if not np.isfinite(p):
        return {
            "key": "unknown",
            "label": "N/A",
            "signal_key": "unknown",
            "signal_label": "N/A",
        }

    if p > 0.8:
        key, label = "high", "VOL ÉLEVÉE"
    elif p > 0.6:
        key, label = "above", "AU-DESSUS DE LA MOYENNE"
    elif p > 0.4:
        key, label = "normal", "NORMALE"
    elif p > 0.2:
        key, label = "below", "EN-DESSOUS DE LA MOYENNE"
    else:
        key, label = "low", "VOL FAIBLE"

    if p > 0.8:
        signal_key, signal_label = "down", "MEAN REVERSION ↓ ATTENDUE"
    elif p < 0.2:
        signal_key, signal_label = "up", "MEAN REVERSION ↑ ATTENDUE"
    else:
        signal_key, signal_label = "neutral", "NEUTRE"

    return {"key": key, "label": label, "signal_key": signal_key, "signal_label": signal_label}


def _linreg(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    res = stats.linregress(x.astype(float), y.astype(float))
    return {
        "slope": float(res.slope),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue) ** 2,
        "p_value": float(res.pvalue),
        "std_err": float(res.stderr) if res.stderr is not None else float("nan"),
        "n": int(len(x)),
    }


def analyze_forward_vol(
    vol: pd.Series,
    *,
    forward_window: int = DEFAULT_FORWARD_WINDOW,
    percentile: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Legacy "Analyze Implied Vol" step, on any vol series (RV or IV):

    - forward vol = rolling mean over `forward_window`, shifted `forward_window`
      into the future (the last `forward_window` rows are therefore NaN),
    - regression 1: forward vol ~ current vol,
    - regression 2: (forward - current) ~ current vol,
    - regime split at the intersection of regression 1 with y=x,
    - per-regime regressions of the vol difference when enough points exist.

    Raises ValueError when fewer than MIN_ANALYSIS_POINTS usable rows remain.
    """
    fw = max(1, int(forward_window))
    v = pd.to_numeric(vol, errors="coerce").astype(float).dropna()

    forward = v.rolling(window=fw, min_periods=1).mean().shift(-fw)
    df = pd.DataFrame(
        {
            "current_vol": v,
            "forward_vol": forward,
            "vol_diff": forward - v,
        }
    )
    if percentile is not None:
        df["vol_percentile"] = pd.to_numeric(percentile, errors="coerce").reindex(df.index)

    df = df.dropna(subset=["current_vol", "forward_vol", "vol_diff"])
    if len(df) < MIN_ANALYSIS_POINTS:
        raise ValueError(
            f"Série insuffisante pour l'analyse: {len(df)} points utilisables "
            f"(minimum {MIN_ANALYSIS_POINTS})."
        )

    reg_forward = _linreg(df["current_vol"], df["forward_vol"])
    reg_diff = _linreg(df["current_vol"], df["vol_diff"])

    slope1 = reg_forward["slope"]
    if np.isfinite(slope1) and abs(1.0 - slope1) > 1e-12:
        intersection = reg_forward["intercept"] / (1.0 - slope1)
    else:
        intersection = float(df["current_vol"].median())
    if not np.isfinite(intersection):
        intersection = float(df["current_vol"].median())

    high_mask = df["current_vol"] > intersection
    low_mask = ~high_mask

    reg_high = (
        _linreg(df.loc[high_mask, "current_vol"], df.loc[high_mask, "vol_diff"])
        if int(high_mask.sum()) > MIN_REGIME_POINTS
        else None
    )
    reg_low = (
        _linreg(df.loc[low_mask, "current_vol"], df.loc[low_mask, "vol_diff"])
        if int(low_mask.sum()) > MIN_REGIME_POINTS
        else None
    )

    insights: list[str] = []
    if np.isfinite(slope1):
        if slope1 < 1.0:
            insights.append("La vol forward tend à mean-reverter (pente < 1).")
        else:
            insights.append("La vol forward tend à suivre la tendance (pente ≥ 1).")
    if np.isfinite(reg_diff["slope"]):
        if reg_diff["slope"] < 0.0:
            insights.append(
                "Une vol courante élevée prédit une vol future plus basse (mean reversion)."
            )
        else:
            insights.append(
                "Une vol courante élevée prédit une vol future plus haute (momentum)."
            )

    return {
        "df": df,
        "forward_window": fw,
        "reg_forward": reg_forward,
        "reg_diff": reg_diff,
        "intersection": float(intersection),
        "reg_high": reg_high,
        "reg_low": reg_low,
        "n_high": int(high_mask.sum()),
        "n_low": int(low_mask.sum()),
        "insights": insights,
    }


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "DEFAULT_RV_WINDOW",
    "DEFAULT_FORWARD_WINDOW",
    "DEFAULT_PERCENTILE_WINDOW",
    "MIN_ANALYSIS_POINTS",
    "compute_log_returns",
    "compute_realized_vol",
    "compute_percentile_series",
    "percentile_within",
    "classify_regime",
    "analyze_forward_vol",
]
