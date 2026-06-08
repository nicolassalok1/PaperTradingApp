"""
Context / market / rate helpers extracted from controller_bridge (Step-6 refactor).

State-derived option-context helpers the bridge defines locally. controller_bridge
re-exports them unchanged (public façade + signatures guarded by tests/characterization).
View layer: imports streamlit + the options controller, never model/utils.
"""

from __future__ import annotations

import streamlit as st

from app.controller import options_controller as oc

load_close_series_for_ticker = oc.load_close_series_for_ticker


def ensure_close_history(ctx: dict) -> bool:
    """
    Ensure closing prices are available for the current global ticker.
    When unavailable, show a notice and skip rendering.
    """
    available = bool(ctx.get("close_available"))
    if available:
        return True
    ticker = ctx.get("ticker") or resolve_common_underlying()
    tkr_display = (ticker or "").strip().upper() or "N/A"
    st.info(
        f"Clotures introuvables pour le ticker global ({tkr_display}). "
        "Renseigne un ticker valide puis recharge les clÙtures pour afficher ce panneau."
    )
    return False


def current_ticker(ctx: dict) -> str:
    """Return the active ticker used for charts/labels."""
    return (ctx.get("ticker") or resolve_common_underlying() or "").strip().upper()


def current_spot(ctx: dict) -> float:
    """
    Derive a spot anchor from the latest close when available,
    then fallback to S0/context defaults.
    """
    close_series = ctx.get("close_series")
    spot = None
    try:
        if hasattr(close_series, "dropna") and len(close_series.dropna()) > 0:
            spot = float(close_series.dropna().iloc[-1])
    except Exception:
        spot = None
    if spot is None and ctx.get("S0") is not None:
        try:
            spot = float(ctx.get("S0"))
        except Exception:
            spot = None
    if spot is None:
        spot = float(st.session_state.get("common_spot_value", 100.0))
    return float(spot)


def get_common_maturity_value(default: float = 1.0) -> float:
    try:
        return float(st.session_state.get("common_maturity_value", default))
    except Exception:
        return float(default)


def get_common_rate_value(default: float = 0.01) -> float:
    try:
        return float(st.session_state.get("common_rate_value", default))
    except Exception:
        return float(default)


def get_common_sigma_value(default: float = 0.2) -> float:
    try:
        return float(st.session_state.get("common_sigma_value", default))
    except Exception:
        return float(default)


def get_common_div_yield(default: float = 0.0) -> float:
    try:
        return float(st.session_state.get("d_common", default))
    except Exception:
        return float(default)


def get_rate_for_ttm(T: float, default: float = 0.01) -> float:
    """
    Resolve the risk-free rate for a given maturity:
    - If Options is configured to use Yield Curve, query yc.get_risk_free_rate(T).
    - Otherwise, fall back to the global manual rate (common_rate_value).
    """
    try:
        use_yc = bool(st.session_state.get("opt_use_yield_curve_rate", True))
    except Exception:
        use_yc = False

    if use_yc:
        currency = (st.session_state.get("yc_currency") or "USD").strip().upper()
        try:
            from app.controller import yieldcurve_controller as yc  # local import (UI helper)

            return float(
                yc.get_risk_free_rate(T_ref=float(T), currency=currency, ensure_cache=True)
            )
        except Exception:
            pass

    return float(get_common_rate_value(default))


def resolve_common_underlying() -> str:
    """Return the shared ticker set by the user (empty string if unset)."""
    ticker = (
        st.session_state.get("tkr_common")
        or st.session_state.get("common_underlying")
        or st.session_state.get("ticker_default")
        or ""
    )
    return str(ticker or "").strip().upper()


def load_shared_close_series(fallback_value: float):
    """
    Load close series for the shared ticker only if the user provided one.
    Returns (ticker, series|None).
    """
    ticker = resolve_common_underlying()
    if not ticker or load_close_series_for_ticker is None:
        return ticker, None
    try:
        return ticker, load_close_series_for_ticker(ticker, fallback_value=fallback_value)
    except Exception:
        return ticker, None
