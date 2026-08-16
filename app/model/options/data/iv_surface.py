from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from app.model.market_data.market_data import fetch_options_details_yahoo, fetch_spot_price
from app.utils.paths import CACHE_CSV_DIR, CACHE_YAHOO_OPTION_CHAINS_DIR


def _build_iv_surface_from_yahoo(ticker: str, max_maturity_years: float = 2.0) -> pd.DataFrame:
    """
    Build IV surface from Yahoo option chain (calls + puts).
    Returns a normalized DataFrame with columns: K, T, S0, iv, type.
    """
    calls_df, puts_df, spot, _, _ = fetch_options_details_yahoo(
        ticker, max_maturity_years=max_maturity_years
    )
    s0 = spot if pd.notna(spot) else fetch_spot_price(ticker)
    sym = (ticker or "").strip().upper()

    records: List[dict] = []

    def _append_rows(df: pd.DataFrame, opt_type: str) -> None:
        if df is None or df.empty:
            return

        cols = {c.lower(): c for c in df.columns}
        strike_col = cols.get("strike") or cols.get("k")
        iv_col = cols.get("iv") or cols.get("iv_market") or cols.get("sigma")
        t_col = cols.get("t") or cols.get("ttm") or cols.get("tau") or cols.get("maturity")
        if strike_col is None or iv_col is None:
            return

        df_clean = df.dropna(subset=[strike_col, iv_col]).copy()
        df_clean[strike_col] = pd.to_numeric(df_clean[strike_col], errors="coerce")
        df_clean[iv_col] = pd.to_numeric(df_clean[iv_col], errors="coerce")

        if t_col and t_col in df_clean.columns:
            df_clean[t_col] = pd.to_numeric(df_clean[t_col], errors="coerce")
            df_clean = df_clean.dropna(subset=[t_col])
            df_clean = df_clean[(df_clean[t_col] > 0) & (df_clean[t_col] <= max_maturity_years)]
        elif "T" in df_clean.columns:
            df_clean["T"] = pd.to_numeric(df_clean["T"], errors="coerce")
            df_clean = df_clean.dropna(subset=["T"])
            df_clean = df_clean[(df_clean["T"] > 0) & (df_clean["T"] <= max_maturity_years)]
            t_col = "T"
        else:
            return

        df_clean = df_clean.dropna(subset=[strike_col, iv_col, t_col])
        if df_clean.empty:
            return

        for _, row in df_clean.iterrows():
            K = row[strike_col]
            T = row[t_col]
            iv = row[iv_col]
            if pd.isna(K) or pd.isna(T) or pd.isna(iv):
                continue
            records.append(
                {
                    "K": float(K),
                    "T": float(T),
                    "S0": float(s0) if s0 is not None and not pd.isna(s0) else float("nan"),
                    "iv": float(iv),
                    "type": opt_type,
                }
            )

    _append_rows(calls_df, "call")
    _append_rows(puts_df, "put")

    surface = pd.DataFrame(records, columns=["K", "T", "S0", "iv", "type"])
    path = CACHE_YAHOO_OPTION_CHAINS_DIR / f"iv_surface_yahoo_{sym}.csv"
    # Avoid overwriting a previously valid cache with an empty fetch.
    if surface is not None and not surface.empty:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            surface.to_csv(path, index=False)
        except Exception:
            pass
    return surface


def fetch_iv_surface(ticker: str, max_maturity_years: float = 2.0) -> pd.DataFrame:
    """Public entrypoint building an IV surface from Yahoo option chain."""
    return _build_iv_surface_from_yahoo(ticker, max_maturity_years=max_maturity_years)


def interpolate_surface(df: pd.DataFrame):
    """
    Placeholder interpolator: returns the raw surface for downstream consumers.
    """
    if df is None or df.empty:
        return None, None, None
    return None, None, df


def load_iv_from_csv(path: str | Path) -> pd.DataFrame:
    """Best-effort CSV loader for cached IV surfaces."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def list_cached_iv_surface_tickers(cache_dir: Path | None = None) -> List[str]:
    """
    Tickers for which a Yahoo IV surface is already on disk (loadable offline).

    Two file families live in the cache: `iv_surface_yahoo_<SYM>.csv` (built surface)
    and `yahoo_chain_<SYM>_Y<years>_E<n>.csv` (raw chain, one per max_years setting).
    Yahoo publishes no catalogue of optionable underlyings, so this is the only
    "surfaces available" list that can be answered without a network call.
    """
    root = Path(cache_dir) if cache_dir is not None else CACHE_YAHOO_OPTION_CHAINS_DIR
    if not root.exists():
        return []
    found: set[str] = set()
    for path in root.glob("iv_surface_yahoo_*.csv"):
        sym = path.stem[len("iv_surface_yahoo_"):]
        if sym:
            found.add(sym.strip().upper())
    for path in root.glob("yahoo_chain_*_Y*_E*.csv"):
        body = path.stem[len("yahoo_chain_"):]
        sym = body.rsplit("_Y", 1)[0] if "_Y" in body else ""
        if sym:
            found.add(sym.strip().upper())
    return sorted(found)


__all__ = [
    "fetch_iv_surface",
    "_build_iv_surface_from_yahoo",
    "interpolate_surface",
    "list_cached_iv_surface_tickers",
    "load_iv_from_csv",
]
