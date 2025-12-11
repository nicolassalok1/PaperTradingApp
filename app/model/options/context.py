from __future__ import annotations

import pandas as pd

from app.model.market_data.market_data import fetch_closing_prices, fetch_spot_price


def _session_spot_value() -> float | None:
    """Grab common_spot_value from Streamlit session if available (no direct import)."""
    try:
        st = __import__("streamlit")
        val = st.session_state.get("common_spot_value")
        return float(val) if val is not None else None
    except Exception:
        return None


def _extract_close_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if len(df.columns) == 0:
        return pd.Series(dtype=float)
    date_col = next((c for c in df.columns if str(c).lower() == "date"), df.columns[0])
    close_col = next((c for c in df.columns if str(c).lower() == "close"), None)
    if close_col is None and len(df.columns) > 1:
        close_col = df.columns[1]
    try:
        series = pd.Series(df[close_col].values, index=pd.to_datetime(df[date_col], errors="coerce"))
    except Exception:
        return pd.Series(dtype=float)
    return series.dropna()


def build_option_context(ticker: str) -> dict:
    """
    Build an option context with close series and inferred spot.
    """
    tk = (ticker or "").strip().upper()
    if not tk:
        s0_session = _session_spot_value()
        s0_default = float(s0_session) if s0_session is not None else None
        return {
            "S0": s0_default,
            "ticker": "",
            "close_series": pd.Series(dtype=float),
            "close_available": False,
            "_k": lambda name: f"__EMPTY__{name}",
        }
    try:
        close_series = load_close_series_for_ticker(tk)
    except Exception:
        close_series = pd.Series(dtype=float)
    close_available = close_series is not None and not close_series.empty
    if close_series is None or len(close_series) == 0:
        closes_df = fetch_closing_prices(tk, period="2y", interval="1d")
        close_series = _extract_close_series(closes_df)
        close_available = close_series is not None and not close_series.empty

    s0 = _session_spot_value()
    if s0 is None and not close_series.empty:
        try:
            s0 = float(close_series.iloc[-1])
        except Exception:
            s0 = None
    if s0 is None:
        s0 = fetch_spot_price(tk)
    if s0 is None and (close_series is None or close_series.empty):
        s0 = 0.0
    if (close_series is None or close_series.empty) and s0 is not None:
        # Keep a fallback series for downstream UI sliders but mark availability flag False.
        close_series = pd.Series([float(s0)], index=pd.Index([pd.Timestamp.today()]))

    ctx = {
        "S0": float(s0) if s0 is not None else None,
        "ticker": tk,
        "close_series": close_series,
        "close_available": bool(close_available),
        "_k": lambda name: f"{tk}_{name}",
    }
    return ctx


def get_option_context_from_state(state_dict=None):
    """
    Compatibility wrapper that accepts a state mapping from the UI.
    """
    state = state_dict or {}
    tk = (
        state.get("common_underlying")
        or state.get("tkr_common")
        or state.get("heston_cboe_ticker")
        or state.get("ticker")
        or ""
    )
    ctx = build_option_context(tk)
    if "common_spot_value" in state and state.get("common_spot_value") is not None:
        try:
            ctx["S0"] = float(state.get("common_spot_value"))
        except Exception:
            pass
    if "_k" in state:
        ctx["_k"] = state["_k"]
    return ctx


def load_close_series_for_ticker(ticker: str, fallback_value=None):
    """
    Compatibility shim expected by some scripts/tests.
    """
    try:
        from app.model.market_data.history import load_close_series_for_ticker as _legacy

        return _legacy(ticker, fallback_value=fallback_value)
    except Exception:
        return pd.Series(dtype=float)


def get_option_context(state_dict=None):
    """Backward-compatible alias."""
    return get_option_context_from_state(state_dict)


__all__ = [
    "build_option_context",
    "get_option_context_from_state",
    "get_option_context",
    "load_close_series_for_ticker",
]
