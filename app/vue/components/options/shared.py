import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.controller import options_controller as oc

CACHE_OPTIONS_CALLS_FILE = oc.CACHE_OPTIONS_CALLS_FILE
CACHE_OPTIONS_HISTORY_FILE = oc.CACHE_OPTIONS_HISTORY_FILE
CACHE_OPTIONS_META_FILE = oc.CACHE_OPTIONS_META_FILE
CACHE_OPTIONS_PUTS_FILE = oc.CACHE_OPTIONS_PUTS_FILE
CLOSING_CACHE_FILE = oc.CLOSING_CACHE_FILE


def _norm_ticker(ticker: str) -> str:
    return (ticker or "").strip().upper()


def _file_age_hours(path: Path) -> float | None:
    try:
        ts = path.stat().st_mtime
        return (
            datetime.datetime.now() - datetime.datetime.fromtimestamp(ts)
        ).total_seconds() / 3600
    except Exception:
        return None


def load_options_meta() -> dict:
    """Load options meta cache via controller."""
    return oc.load_options_meta()


def save_options_meta(meta: dict) -> None:
    oc.save_options_meta(meta)


def load_cached_option_history() -> tuple[str | None, pd.DataFrame | None]:
    """Load cached 1y close history (closing_cache.csv)."""
    return oc.load_cached_option_history()


def save_cached_option_history(ticker: str, df: pd.DataFrame) -> None:
    """Persist 1y close history into closing_cache.csv (append/merge)."""
    oc.save_cached_option_history(ticker, df)


def save_cached_option_chain(
    ticker: str, calls_df: pd.DataFrame, puts_df: pd.DataFrame, S0_ref: float, r: float, q: float
) -> None:
    """Persist the latest option chains and meta information."""
    oc.save_cached_option_chain(ticker, calls_df, puts_df, S0_ref, r, q)


def load_cached_option_chain(
    ticker: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, float | None, float | None, float | None]:
    """Load cached option chain if it matches the requested ticker."""
    return oc.load_cached_option_chain(ticker)


def fetch_option_history_to_cache(ticker: str) -> pd.DataFrame:
    """
    Download 1y daily closes via CLI helper and persist to cache CSV.
    Returns the DataFrame (may be empty on failure).
    """
    return oc.fetch_option_history_to_cache(ticker)


def refresh_underlying_cache(ticker: str):
    """Download option chains and closing history for a ticker, then cache them."""
    tkr = _norm_ticker(ticker)
    if not tkr:
        st.warning("Merci de saisir un ticker pour rafraichir le cache.")
        return
    try:
        meta = oc.refresh_underlying_cache(tkr)
        st.success(f"Cache options mis a jour pour {tkr}. S0~{float(meta.get('S0_ref', 0.0)):.2f}")
    except Exception as exc:
        st.error(f"Echec de la mise a jour du cache pour {tkr}: {exc}")


def show_cache_status(ticker: str):
    """Display cache status for the selected ticker."""
    tkr = _norm_ticker(ticker)
    meta = load_options_meta()
    meta_tkr = _norm_ticker(meta.get("ticker", ""))
    chain_age = _file_age_hours(CACHE_OPTIONS_META_FILE)
    hist_age = _file_age_hours(CLOSING_CACHE_FILE)
    chain_present = CACHE_OPTIONS_CALLS_FILE.exists() or CACHE_OPTIONS_PUTS_FILE.exists()
    hist_present = CLOSING_CACHE_FILE.exists()

    if tkr and meta_tkr == tkr and chain_present:
        age_txt = f" (~{chain_age:.1f} h)" if chain_age is not None else ""
        st.success(f"Chaines options en cache pour {tkr}{age_txt}.")
    elif chain_present and not tkr:
        st.info("Chaines options en cache (ticker non renseigne).")
    elif hist_present:
        age_txt = f" (~{hist_age:.1f} h)" if hist_age is not None else ""
        st.warning(
            f"Aucune chaine options en cache pour {tkr or 'ticker ?'}. Historique clotures disponible{age_txt}."
        )
    else:
        st.warning(
            f"Aucun cache disponible pour {tkr or 'ticker ?'}. Clique sur Refresh pour telecharger les donnees."
        )


def load_market_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """
    Download market call/put chains (Yahoo) and return (calls_df, puts_df, S0_ref, r, q).
    Caches are handled by the caller.
    """
    return oc.load_market_data(symbol)
