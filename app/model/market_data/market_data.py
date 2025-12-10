"""
Market Data Engine (PRO Version)
- SPOT   : Alpaca → fallback yfinance
- OHLC   : Alpaca → fallback yfinance.download
- OPTIONS: CBOE (calls + puts)
- CSV cache: cache/

This file removes ALL yfinance metadata usage
→ no more 'currentTradingPeriod' errors.
"""

from __future__ import annotations

import os
import json
import logging
import contextlib
import io
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import yfinance as yf
from app.utils.math_utils import floor_4
from app.utils.paths import CACHE_CSV_DIR

# Alpaca is optional; fall back to yfinance if not installed or not configured.
try:  # pragma: no cover - optional dependency
    from alpaca_trade_api.rest import REST as AlpacaREST, TimeFrame
except Exception:  # noqa: BLE001
    AlpacaREST = None  # type: ignore
    TimeFrame = None  # type: ignore


# ======================================================================
def save_csv(df: pd.DataFrame, filename: str):
    """Write DataFrame to cache folder (<cache>/<filename>), ignore empty."""
    try:
        if df is not None and not df.empty:
            (CACHE_CSV_DIR / filename).write_text(df.to_csv(index=False))
    except Exception as exc:
        logging.warning(f"[cache] Failed to save {filename}: {exc}")


# ======================================================================
# 1) ALPACA CREDENTIALS + NORMALIZATION
# ======================================================================


def _load_env_fallback():
    """Load .env into os.environ if present (lightweight parser)."""
    env_file = Path(".env")
    if not env_file.exists():
        return
    try:
        for line in env_file.read_text().splitlines():
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def _load_alpaca_credentials():
    """Load Alpaca creds from env/.env or json."""
    _load_env_fallback()
    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL") or "https://paper-api.alpaca.markets"

    if key and secret:
        return key, secret, base

    for fp in [".secrets/alpaca_keys.json", "alpaca_keys.json"]:
        f = Path(fp)
        if f.exists():
            try:
                payload = json.loads(f.read_text())
                key = payload.get("APCA_API_KEY_ID") or payload.get("key_id")
                secret = payload.get("APCA_API_SECRET_KEY") or payload.get("secret_key")
                base = payload.get("APCA_API_BASE_URL") or base
                return key, secret, base
            except Exception:
                pass

    return None, None, base


def _norm(sym: str) -> str:
    """Basic normalization: strip $, hyphens, and uppercase."""
    return (sym or "").strip().upper().replace("-", "").replace("$", "")


def _yfinance_disabled() -> bool:
    """Guardrail to disable yfinance globally via env flag."""
    return str(os.getenv("DISABLE_YFINANCE") or os.getenv("YFINANCE_DISABLED") or "").lower() in {
        "1",
        "true",
        "yes",
    }


def _load_cached_spot(sym: str) -> float | None:
    """Try to read last cached spot from CSV to avoid noisy fallbacks."""
    fp = CACHE_CSV_DIR / f"spot_{sym}.csv"
    if not fp.exists():
        return None
    try:
        df = pd.read_csv(fp)
        if not df.empty and "spot" in df.columns:
            return float(df["spot"].iloc[-1])
    except Exception:
        return None
    return None


def _yfinance_ohlc_allowed() -> bool:
    """
    Whether yfinance is allowed for OHLC. Default: False (Alpaca-only) unless explicitly enabled.
    Set ALLOW_YFINANCE_OHLC=1/true to permit fallback, or DISABLE_YFINANCE=1 to force off.
    """
    if _yfinance_disabled():
        return False
    return str(os.getenv("ALLOW_YFINANCE_OHLC") or "").lower() in {"1", "true", "yes"}


# ======================================================================
# 2) SPOT PRICE = Alpaca → YF fast_info → YF history
# ======================================================================


def fetch_spot_price(symbol: str) -> float | None:
    sym = _norm(symbol)
    if not sym:
        return None

    key, secret, base = _load_alpaca_credentials()

    # --- 1) Alpaca latest trade ---
    if key and secret and AlpacaREST is not None:
        try:
            client = AlpacaREST(key, secret, base, api_version="v2")
            t = client.get_latest_trade(sym)
            px = getattr(t, "price", None)
            if px is None and hasattr(client, "get_latest_quote"):
                try:
                    q = client.get_latest_quote(sym)
                    px = getattr(q, "bidprice", None) or getattr(q, "askprice", None)
                except Exception:
                    px = None
            if px is not None:
                px = floor_4(px)
                save_csv(pd.DataFrame([{"symbol": sym, "spot": px}]), f"spot_{sym}.csv")
                return px
        except Exception as exc:
            logging.warning(f"[spot] Alpaca failed for {sym}: {exc}")

    # --- 2) yfinance fast_info ---
    if not _yfinance_disabled():
        try:
            yt = yf.Ticker(sym)
            fi = getattr(yt, "fast_info", {}) or {}
            for k in ("lastPrice", "last_price", "last_close", "previousClose"):
                v = fi.get(k)
                if v not in (None, ""):
                    v = floor_4(v)
                    save_csv(pd.DataFrame([{"symbol": sym, "spot": v}]), f"spot_{sym}.csv")
                    return v
        except Exception as exc:
            logging.warning(f"[spot] yfinance fast_info failed for {sym}: {exc}")

    # --- 3) yfinance history (safe: never touches metadata) ---
    if not _yfinance_disabled():
        try:
            df = yf.download(sym, period="1mo", interval="1d", progress=False, threads=False)
            if df is not None and not df.empty and "Close" in df.columns:
                px = floor_4(df["Close"].iloc[-1])
                save_csv(pd.DataFrame([{"symbol": sym, "spot": px}]), f"spot_{sym}.csv")
                return px
        except Exception as exc:
            logging.warning(f"[spot] yfinance history failed for {sym}: {exc}")

    logging.warning(f"[spot] no data for {sym}")
    cached = _load_cached_spot(sym)
    if cached is not None:
        logging.info(f"[spot] using cached price for {sym}: {cached}")
        return cached
    return None


# ======================================================================
# 3) OHLC HISTORY = Alpaca → fallback yfinance.download
# ======================================================================


def _fetch_ohlc_alpaca(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch OHLC from Alpaca, daily bars."""
    if AlpacaREST is None or TimeFrame is None:
        return pd.DataFrame()

    key, secret, base = _load_alpaca_credentials()
    if not key or not secret:
        return pd.DataFrame()

    try:
        client = AlpacaREST(key, secret, base, api_version="v2")
        bars = client.get_bars(symbol, TimeFrame.Day, limit=days).df
        if bars is None or bars.empty:
            return pd.DataFrame()

        bars = bars.reset_index()
        bars["symbol"] = symbol
        return bars
    except Exception as exc:
        logging.warning(f"[OHLC] Alpaca failed for {symbol}: {exc}")
        return pd.DataFrame()


def _fetch_ohlc_yf(symbol: str, period="1y", interval="1d") -> pd.DataFrame:
    """Fallback OHLC from yfinance using safe download()."""
    if not _yfinance_ohlc_allowed():
        logging.info(f"[OHLC] yfinance OHLC disabled via env, skip for {symbol}")
        return pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval)
        if df is not None and not df.empty:
            df = df.reset_index()
            df["symbol"] = symbol
            return df
    except Exception as exc:
        logging.error(f"[OHLC] yfinance download failed for {symbol}: {exc}")

    return pd.DataFrame()


def fetch_closing_prices(
    tickers: str | Iterable[str], period: str = "1y", interval: str = "1d"
) -> pd.DataFrame:
    """
    OHLC fetcher:
    Alpaca → fallback yfinance.download.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    frames = []

    for sym in tickers:
        sym = _norm(sym)

        # 1) Alpaca
        df = _fetch_ohlc_alpaca(sym, days=365)
        if df.empty and _yfinance_ohlc_allowed():
            # 2) fallback yfinance (explicitly allowed)
            df = _fetch_ohlc_yf(sym, period=period, interval=interval)

        if df.empty:
            logging.warning(f"[closing] no data for {sym}")
            continue

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    name = tickers[0] if len(tickers) == 1 else "_".join(tickers)
    suffix = f"_{period.replace(' ', '_')}_{interval.replace(' ', '_')}"
    save_csv(out, f"closing_{name}{suffix}.csv")
    return out


# ----------------------------------------------------------------------
# Unified closing history with cache (used by UI)
# ----------------------------------------------------------------------


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    safe = f"{ticker}_{period}_{interval}".replace("/", "_").replace(" ", "_")
    return CACHE_CSV_DIR / f"closing_{safe}.csv"


def _normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize YF downloads: flatten columns, coerce numeric, drop empties."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ("Close", "Open", "High", "Low", "Adj Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    drop_cols = []
    if "Date" in df.columns:
        drop_cols.append("Date")
    if "Close" in df.columns:
        drop_cols.append("Close")
    if drop_cols:
        df = df.dropna(subset=drop_cols)
    return df.reset_index(drop=True)


def load_or_fetch_closing_history(
    ticker: str,
    *,
    period: str = "1y",
    interval: str = "1d",
):
    """Return closing history from cache or download; returns (df, cache_path, from_cache)."""
    tk = (ticker or "").strip().upper()
    if not tk:
        return None, None, False
    path = _cache_path(tk, period, interval)
    if path.exists():
        try:
            df = pd.read_csv(path)
            df = _normalize_history_df(df)
            if df.empty:
                raise ValueError("cached history empty after normalization")
            try:
                df.to_csv(path, index=False)
            except Exception:
                pass
            return df, path, True
        except Exception:
            pass

    buf = io.StringIO()
    df = pd.DataFrame()
    # Attempt download with suppressed stdout/stderr
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            df = yf.download(tk, period=period, interval=interval, progress=False, threads=False)
    except Exception:
        df = pd.DataFrame()

    # Fallback: history() may succeed when download fails
    if df is None or df.empty:
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                tkr = yf.Ticker(tk)
                df = tkr.history(period=period, interval=interval, raise_errors=False)
        except Exception:
            df = pd.DataFrame()

    if df is not None and not df.empty:
        try:
            df = df.reset_index()
            df = _normalize_history_df(df)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            return df, path, False
        except Exception:
            pass

    return None, path, False


def clear_closing_history_cache(ticker: str, *, period: str = "1y", interval: str = "1d") -> None:
    """Remove cached closing history for a ticker."""
    norm = (ticker or "").strip().upper()
    path = _cache_path(norm, period, interval)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


# ======================================================================
# 4) OPTIONS = CBOE (calls + puts)
# ======================================================================


def fetch_options_details(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    """
    Returns:
        calls_df, puts_df, spot, rf, div_yield
    """
    try:
        from app.model.options import (
            logic as opt_logic,
        )  # lazy import to avoid hard dependency on scipy/alpaca
    except Exception as exc:  # noqa: BLE001
        logging.warning(f"[options] dependencies missing; options data unavailable: {exc}")
        return pd.DataFrame(), pd.DataFrame(), float("nan"), float("nan"), float("nan")

    sym = _norm(symbol)
    if not sym:
        return pd.DataFrame(), pd.DataFrame(), float("nan"), float("nan"), float("nan")

    try:
        calls_df, spot_c, rf_c, div_c = opt_logic.download_options_cboe(sym, "call")
        puts_df, spot_p, rf_p, div_p = opt_logic.download_options_cboe(sym, "put")

        # Meta selection
        spot = spot_c if spot_c else spot_p
        rf = rf_c if rf_c else rf_p
        div = div_c if (div_c == div_c) else div_p  # NaN check

        # Cache CSV at this level too (optional)
        save_csv(calls_df, f"options_calls_{sym}.csv")
        save_csv(puts_df, f"options_puts_{sym}.csv")

        return calls_df, puts_df, spot, rf, div

    except Exception as exc:
        logging.warning(f"[options] CBOE failed for {sym}: {exc}")
        return pd.DataFrame(), pd.DataFrame(), float("nan"), float("nan"), float("nan")


# ======================================================================
# 5) Manual client creation
# ======================================================================


def make_alpaca_client():
    key, secret, base = _load_alpaca_credentials()
    if not key or not secret:
        return None
    try:
        return AlpacaREST(key, secret, base, api_version="v2")
    except Exception:
        return None


__all__ = [
    "fetch_spot_price",
    "fetch_closing_prices",
    "fetch_options_details",
    "make_alpaca_client",
]
