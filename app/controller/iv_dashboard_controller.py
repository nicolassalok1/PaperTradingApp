"""
Controller for the 🌡️ Vol Implicite tab.

Thin glue: sanitize the view's inputs, delegate to the iv_dashboard model
service, and hand back a payload ready for rendering. No domain logic here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.model.iv_dashboard import service as _svc


def _clamp_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = int(default)
    return max(lo, min(v, hi))


def _clamp_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float(default)
    return max(lo, min(v, hi))


def get_iv_analysis(
    symbol: str,
    *,
    years: float = 2.0,
    rv_window: int = 20,
    forward_window: int = 30,
    percentile_window: int = 252,
    include_current_iv: bool = True,
) -> Dict[str, Any]:
    """
    Full payload for the tab: RV series + regime + forward-vol regressions,
    current Alpaca ATM IV (optional) and the locally accumulated IV history.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        raise ValueError("Symbole requis.")

    return _svc.get_iv_dashboard_data(
        sym,
        years=_clamp_float(years, 2.0, 0.5, 10.0),
        rv_window=_clamp_int(rv_window, 20, 5, 120),
        forward_window=_clamp_int(forward_window, 30, 5, 90),
        percentile_window=_clamp_int(percentile_window, 252, 60, 756),
        include_current_iv=bool(include_current_iv),
    )


__all__ = ["get_iv_analysis"]
