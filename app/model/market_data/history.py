"""
Thin wrappers maintained for backward compatibility.
The actual API logic now lives in app.model.market_data.market_data.
"""

from app.model.market_data.market_data import (
    load_or_fetch_closing_history,
    clear_closing_history_cache,
)
# Legacy helper expected by options_controller and UI components.
try:  # noqa: SIM105 - soft import to avoid hard dependency
    from app.model.options.logic import load_close_series_for_ticker as _legacy_load_close_series
except Exception:  # pragma: no cover - optional dependency
    _legacy_load_close_series = None


def load_close_series_for_ticker(ticker: str, fallback_value=None):
    """
    Compatibility wrapper to load a close series for a ticker.
    Delegates to the legacy options.logic helper if available,
    otherwise falls back to the generic load_or_fetch_closing_history.
    """
    if _legacy_load_close_series is not None:
        try:
            return _legacy_load_close_series(ticker, fallback_value=fallback_value)
        except Exception:
            pass

    df, _, _ = load_or_fetch_closing_history(
        ticker, period="1y", interval="1d"
    )
    if df is not None and not df.empty:
        close_col = df.columns[-1]
        return df[close_col]
    return None


__all__ = [
    "load_or_fetch_closing_history",
    "clear_closing_history_cache",
    "load_close_series_for_ticker",
]
