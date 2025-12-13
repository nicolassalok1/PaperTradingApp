"""
Controller for Options domain (thin wrappers over model services).
"""

from __future__ import annotations

from app.model.market_data.history import (
    clear_closing_history_cache,
    load_close_series_for_ticker,
    load_or_fetch_closing_history,
)
from app.model.market_data.realtime import get_data
from app.model.options.engines.black_scholes import (
    black_scholes_price,
    price_asset_or_nothing,
    price_butterfly,
    price_calendar_spread,
    price_call_spread,
    price_chooser,
    price_condor,
    price_diagonal_spread,
    price_digital,
    price_forward_start,
    price_iron_butterfly,
    price_iron_condor,
    price_put_spread,
    price_quanto,
    price_rainbow,
    price_straddle,
    price_strangle,
)
from app.model.options.engines.crr import price_american_crr, price_bermuda_crr
from app.model.options.data.iv_surface import (
    fetch_iv_surface,
    interpolate_surface,
    load_iv_from_csv,
)
from app.model.options.heatmaps import compute_crr_heatmaps, heatmap_axis, render_heatmap
from app.model.options.helpers import _get_cached_iv_for
from app.model.options.core.iv import (
    CACHE_OPTIONS_CALLS_FILE,
    CACHE_OPTIONS_HISTORY_FILE,
    CACHE_OPTIONS_META_FILE,
    CACHE_OPTIONS_PUTS_FILE,
    CLOSING_CACHE_FILE,
    fetch_option_history_to_cache,
    load_market_data,
    load_cached_option_chain,
    load_cached_option_history,
    load_options_meta,
    save_cached_option_chain,
    save_cached_option_history,
    save_options_meta,
)
from app.model.options.core.pricing_lib import price_iron_butterfly_bs
from app.model.options.ui_gateway.pricing_ui import (
    view_asian_arith,
    view_asian_geom,
    view_asset_or_nothing,
    view_barrier,
    view_butterfly,
    view_call_spread,
    view_calendar_spread,
    view_chooser,
    view_cliquet,
    view_condor,
    view_diagonal_spread,
    view_digital,
    view_european,
    view_american,
    view_bermudan,
    view_forward_start,
    view_iron_butterfly,
    view_iron_condor,
    view_lookback,
    view_lookback_fixed,
    view_put_spread,
    view_quanto,
    view_rainbow,
    view_straddle,
    view_strangle,
)
from app.model.options.service import (
    compute_greeks,
    compute_payoff_surface,
    compute_price,
    load_iv_surface,
    price_and_greeks,
    price_option_mc,
    price_option_mc_unified,
    price_european_from_market,
)
from app.model.options.logic import download_options_alpaca as _download_options_alpaca
from app.model.options.engines.tree import build_crr_tree, plot_crr_tree
from app.model.options.core.trees import (
    price_asian_geo_mc,
    price_asian_mc,
    price_barrier_digital,
    price_barrier_vanilla,
    price_cliquet,
    price_lookback_fixed_mc,
    price_lookback_mc,
)
from app.model.options.core import shared as _shared
from app.model.options.context import (
    get_option_context_from_state as _get_option_context_from_state,
)
from app.utils.math_utils import floor_n as _floor_n


def refresh_underlying_cache(ticker: str):
    """
    Download option chains and history for a ticker, then persist through the IV cache helpers.
    """
    symbol = (ticker or "").strip().upper()
    if not symbol:
        raise ValueError("Ticker manquant pour rafraichir le cache.")

    calls_df, puts_df, S0_ref, rf_rate, div_yield = load_market_data(symbol)
    save_cached_option_chain(symbol, calls_df, puts_df, S0_ref, rf_rate, div_yield)
    fetch_option_history_to_cache(symbol)
    return {
        "ticker": symbol,
        "S0_ref": S0_ref,
        "r": rf_rate,
        "q": div_yield,
    }


def floor_n(x, n=2):
    """Expose floor_n to the UI without a direct utils dependency."""
    return _floor_n(x, n)


def build_option_context(state_dict):
    """Build an options context from a state mapping (UI wrapper)."""
    return _get_option_context_from_state(state_dict)


def get_shared():
    """Expose shared module for cached IV helpers."""
    return _shared


def compute_price_mc(option_dict, mc_model: str = "bs", n_paths: int = 10000, n_steps: int = 252):
    """
    Compute a Monte Carlo price for a European option using the MC engine.
    Expects option_dict to contain at least: ticker, K, T, sigma.
    """
    ticker = option_dict["ticker"]
    K = float(option_dict["K"])
    T = float(option_dict["T"])
    sigma = float(option_dict["sigma"])
    return price_option_mc(
        ticker=ticker,
        K=K,
        T=T,
        sigma=sigma,
        model=mc_model,
        n_paths=n_paths,
        n_steps=n_steps,
    )


def download_alpaca_options_chain(
    ticker: str,
    *,
    feed: str = "indicative",
    max_pages: int | None = None,
    max_contracts: int | None = None,
    min_days_to_expiry: int | None = 1,
    include_spot: bool = True,
    cache_to_csv: bool = True,
):
    """
    Thin wrapper to expose Alpaca options snapshots to the UI via the controller.

    Alpaca snapshots are paginated; by default we pull all pages.
    Use `max_pages` / `max_contracts` only to cap the fetch.
    """
    return _download_options_alpaca(
        ticker,
        feed=feed,
        max_pages=max_pages,
        max_contracts=max_contracts,
        min_days_to_expiry=min_days_to_expiry,
        include_spot=include_spot,
        cache_to_csv=cache_to_csv,
    )


__all__ = [
    # Core service wrappers
    "compute_price",
    "compute_greeks",
    "compute_payoff_surface",
    "load_iv_surface",
    "price_and_greeks",
    "compute_price_mc",
    "price_european_from_market",
    "price_option_mc_unified",
    "download_alpaca_options_chain",
    # Black-Scholes / spreads
    "black_scholes_price",
    "price_asset_or_nothing",
    "price_butterfly",
    "price_calendar_spread",
    "price_call_spread",
    "price_chooser",
    "price_condor",
    "price_diagonal_spread",
    "price_digital",
    "price_forward_start",
    "price_iron_butterfly",
    "price_iron_condor",
    "price_put_spread",
    "price_quanto",
    "price_rainbow",
    "price_straddle",
    "price_strangle",
    # Trees / MC
    "price_american_crr",
    "price_bermuda_crr",
    "price_asian_geo_mc",
    "price_asian_mc",
    "price_barrier_digital",
    "price_barrier_vanilla",
    "price_cliquet",
    "price_lookback_fixed_mc",
    "price_lookback_mc",
    # Layout helpers
    "compute_crr_heatmaps",
    "heatmap_axis",
    "render_heatmap",
    "build_crr_tree",
    "plot_crr_tree",
    # Pricing lib extras
    "price_iron_butterfly_bs",
    # View/payoff builders
    "view_asset_or_nothing",
    "view_barrier",
    "view_butterfly",
    "view_call_spread",
    "view_calendar_spread",
    "view_chooser",
    "view_cliquet",
    "view_condor",
    "view_diagonal_spread",
    "view_digital",
    "view_european",
    "view_american",
    "view_bermudan",
    "view_forward_start",
    "view_iron_butterfly",
    "view_iron_condor",
    "view_lookback",
    "view_lookback_fixed",
    "view_put_spread",
    "view_quanto",
    "view_rainbow",
    "view_straddle",
    "view_strangle",
    "view_asian_arith",
    "view_asian_geom",
    # Helper exports
    "_get_cached_iv_for",
    "floor_n",
    "build_option_context",
    "get_shared",
    # Market data
    "get_data",
    "load_or_fetch_closing_history",
    "load_close_series_for_ticker",
    "clear_closing_history_cache",
    # IV surface helpers
    "fetch_iv_surface",
    "load_iv_from_csv",
    "interpolate_surface",
    # Cache helpers
    "CACHE_OPTIONS_CALLS_FILE",
    "CACHE_OPTIONS_HISTORY_FILE",
    "CACHE_OPTIONS_META_FILE",
    "CACHE_OPTIONS_PUTS_FILE",
    "CLOSING_CACHE_FILE",
    "fetch_option_history_to_cache",
    "load_market_data",
    "load_cached_option_chain",
    "load_cached_option_history",
    "load_options_meta",
    "save_cached_option_chain",
    "save_cached_option_history",
    "save_options_meta",
    "refresh_underlying_cache",
]
