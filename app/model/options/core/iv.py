from __future__ import annotations

import datetime
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# Unified cache locations under cache/ and data/.
from app.utils.paths import CACHE_CSV_DIR, JSON_DIR, ROOT_DIR

SCRIPTS_DIR = ROOT_DIR / "app" / "scripts"

for _p in (CACHE_CSV_DIR, JSON_DIR):
    _p.mkdir(parents=True, exist_ok=True)

CACHE_OPTIONS_HISTORY_FILE = CACHE_CSV_DIR / "options_last_history.csv"
CACHE_OPTIONS_CALLS_FILE = CACHE_CSV_DIR / "options_last_calls.csv"
CACHE_OPTIONS_PUTS_FILE = CACHE_CSV_DIR / "options_last_puts.csv"
CACHE_OPTIONS_META_FILE = JSON_DIR / "options_page_cache.json"
CLOSING_CACHE_FILE = CACHE_CSV_DIR / "closing_cache.csv"
OPTIONS_BOOK_FILE = JSON_DIR / "options_portfolio.json"
CACHED_EXPIRED_FILE = JSON_DIR / "options_expired.json"
EXPIRED_OPTIONS_FILE = CACHED_EXPIRED_FILE
OPTIONS_PORTFOLIO_FILE = OPTIONS_BOOK_FILE
CUSTOM_OPTIONS_FILE = JSON_DIR / "custom_options.json"


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


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def get_option_expiries(ticker: str):
    tk = yf.Ticker(ticker)
    return tk.options or []


def get_option_surface_from_yf(ticker: str, expiry: str):
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)

    frames = []
    for frame in [chain.calls, chain.puts]:
        tmp = frame[["strike", "impliedVolatility"]].rename(
            columns={"strike": "K", "impliedVolatility": "iv"}
        )
        tmp["T"] = 0.0
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["K", "iv"])
    return df


def load_options_meta() -> dict:
    """Load options meta cache (no ancienne fallbacks)."""
    meta: dict = {}
    try:
        if CACHE_OPTIONS_META_FILE.exists():
            obj = _load_json(CACHE_OPTIONS_META_FILE)
            if isinstance(obj, dict):
                meta = obj
    except Exception:
        meta = {}

    return meta if isinstance(meta, dict) else {}


def save_options_meta(meta: dict) -> None:
    try:
        _write_json(CACHE_OPTIONS_META_FILE, meta if isinstance(meta, dict) else {})
    except Exception:
        pass


def load_cached_option_history() -> tuple[str | None, pd.DataFrame | None]:
    """Load cached 1y close history (closing_cache.csv)."""
    if not CLOSING_CACHE_FILE.exists():
        return None, None
    try:
        df_raw = pd.read_csv(CLOSING_CACHE_FILE, parse_dates=["Date"])
        date_col = next((c for c in df_raw.columns if str(c).lower() == "date"), None)
        price_cols = [c for c in df_raw.columns if str(c).lower() != "date"]
        if date_col and price_cols:
            df = df_raw[[date_col] + price_cols].copy()
            df.set_index(date_col, inplace=True)
            tkr_guess = price_cols[0] if len(price_cols) == 1 else None
            return tkr_guess, df
    except Exception:
        return None, None
    return None, None


def save_cached_option_history(ticker: str, df: pd.DataFrame) -> None:
    """Persist 1y close history into closing_cache.csv (append/merge)."""
    try:
        ticker_clean = _norm_ticker(ticker) or "TICKER"
        series = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        from app.model.options import (
            logic as opt_logic,
        )  # local import to avoid heavy deps at import time

        opt_logic._update_closing_cache_series(ticker_clean, series)
    except Exception:
        pass


def save_cached_option_chain(
    ticker: str, calls_df: pd.DataFrame, puts_df: pd.DataFrame, S0_ref: float, r: float, q: float
) -> None:
    """Persist the latest CBOE chains and meta information."""
    try:
        CACHE_OPTIONS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        if calls_df is not None and not calls_df.empty:
            calls_df.to_csv(CACHE_OPTIONS_CALLS_FILE, index=False)
        if puts_df is not None and not puts_df.empty:
            puts_df.to_csv(CACHE_OPTIONS_PUTS_FILE, index=False)
        meta = {"ticker": _norm_ticker(ticker), "S0_ref": S0_ref, "r": r, "q": q}
        save_options_meta(meta)
    except Exception:
        pass


def load_cached_option_chain(
    ticker: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, float | None, float | None, float | None]:
    """Load cached CBOE chain if it matches the requested ticker."""
    tkr = _norm_ticker(ticker)
    if not tkr:
        return None, None, None, None, None
    try:
        meta = load_options_meta()
        if _norm_ticker(meta.get("ticker", "")) != tkr:
            return None, None, None, None, None
        calls_df = (
            pd.read_csv(CACHE_OPTIONS_CALLS_FILE) if CACHE_OPTIONS_CALLS_FILE.exists() else None
        )
        puts_df = pd.read_csv(CACHE_OPTIONS_PUTS_FILE) if CACHE_OPTIONS_PUTS_FILE.exists() else None
        return (
            calls_df,
            puts_df,
            float(meta.get("S0_ref") or 0.0),
            float(meta.get("r") or 0.0),
            float(meta.get("q") or 0.0),
        )
    except Exception:
        return None, None, None, None, None


def fetch_option_history_to_cache(ticker: str) -> pd.DataFrame:
    """
    Download 1y daily closes via CLI helper and persist to cache CSV.
    Returns the DataFrame (may be empty on failure).
    """
    tkr = _norm_ticker(ticker)
    if not tkr:
        return pd.DataFrame()
    cli_path = SCRIPTS_DIR / "fetch_history_cli.py"
    hist_df = pd.DataFrame()
    try:
        result = subprocess.run(
            [sys.executable, str(cli_path), "--ticker", tkr, "--period", "1y", "--interval", "1d"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            hist_df = pd.read_csv(io.StringIO(result.stdout))
            if "Date" in hist_df.columns:
                hist_df["Date"] = pd.to_datetime(hist_df["Date"])
                hist_df.set_index("Date", inplace=True)
            save_cached_option_history(tkr, hist_df)
    except Exception:
        pass
    return hist_df


def load_cboe_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """
    Download CBOE call/put chains and return (calls_df, puts_df, S0_ref, r, q).
    Caches are handled by the caller.
    """
    from app.model.options import logic as opt_logic  # local import to keep optional deps lazy

    calls_df, spot_calls, rf_calls, div_calls = opt_logic.download_options_cboe(symbol, "call")
    puts_df, spot_puts, rf_puts, div_puts = opt_logic.download_options_cboe(symbol, "put")
    S0_ref = float(np.nanmean([spot_calls, spot_puts]))
    risk_free = float(np.nanmean([rf_calls, rf_puts]))
    dividend_yield = float(np.nanmean([div_calls, div_puts]))
    return calls_df, puts_df, S0_ref, risk_free, dividend_yield


def get_last_cached_option_ticker() -> str | None:
    """Return last ticker used for options meta cache."""
    meta = load_options_meta()
    tkr = meta.get("ticker")
    if tkr:
        return str(tkr).strip().upper()
    return None
