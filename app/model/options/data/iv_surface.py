from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List

import numpy as np

import pandas as pd

from app.model.market_data.market_data import (
    fetch_options_chain_yahoo,
    fetch_options_details,
    fetch_spot_price,
)
from app.utils.paths import CACHE_CSV_DIR


def _build_iv_surface_from_yahoo(ticker: str, max_maturity_years: float = 2.0) -> pd.DataFrame:
    """
    Build IV surface using Yahoo Finance options chain JSON endpoints.
    """
    calls_df, puts_df, spot = fetch_options_chain_yahoo(ticker)
    s0 = spot if pd.notna(spot) else fetch_spot_price(ticker)
    today = dt.date.today()

    records: List[dict] = []

    def _append(df: pd.DataFrame, opt_type: str) -> None:
        if df is None or df.empty:
            return
        for _, row in df.iterrows():
            try:
                expiry = row.get("expiry")
                if not expiry:
                    continue
                if isinstance(expiry, dt.datetime):
                    expiry = expiry.date()
                days = (expiry - today).days
                if days <= 0:
                    continue
                T = days / 365.0
                if T > max_maturity_years:
                    continue
                K = float(row.get("strike"))
                iv = float(row.get("iv"))
                if pd.isna(K) or pd.isna(iv) or iv <= 0:
                    continue
                records.append(
                    {
                        "K": K,
                        "T": T,
                        "S0": float(s0)
                        if s0 is not None and not pd.isna(s0)
                        else float("nan"),
                        "iv": iv,
                        "type": opt_type,
                    }
                )
            except Exception:
                continue

    _append(calls_df, "call")
    _append(puts_df, "put")

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out = out.sort_values(["type", "T", "K"]).reset_index(drop=True)
    return out


def _decode_opra_expiry(opra: str) -> dt.date | None:
    """
    Extract expiry date from OPRA code (…YYMMDDCTTTTTTTT).
    """
    if not opra or len(opra) < 15:
        return None
    try:
        expiry_str = opra[-15:-9]
        return dt.datetime.strptime(expiry_str, "%y%m%d").date()
    except Exception:
        return None


def _build_iv_surface_from_cboe(ticker: str, max_maturity_years: float = 2.0) -> pd.DataFrame:
    """
    Build IV surface from CBOE call/put chains.
    """
    calls_df, puts_df, spot, _, _ = fetch_options_details(ticker)
    s0 = spot if pd.notna(spot) else fetch_spot_price(ticker)
    sym = (ticker or "").strip().upper()
    today = dt.date.today()

    records: List[dict] = []

    def _append_rows(df: pd.DataFrame, opt_type: str) -> None:
        if df is None or df.empty:
            return
        code_col = next(
            (c for c in df.columns if str(c).lower() in {"opra", "symbol", "option_symbol", "code"}),
            None,
        )
        expiry_col = next((c for c in df.columns if "exp" in str(c).lower()), None)
        strike_col = next((c for c in df.columns if str(c).lower() == "strike"), None)
        iv_col = next((c for c in df.columns if str(c).lower() == "iv"), None)
        if strike_col is None or iv_col is None:
            return
        for _, row in df.iterrows():
            opra_code = str(row[code_col]) if code_col else ""
            expiry = _decode_opra_expiry(opra_code) if opra_code else None
            if expiry is None and expiry_col:
                try:
                    expiry = pd.to_datetime(row[expiry_col]).date()
                except Exception:
                    expiry = None
            if expiry is None:
                continue
            T = (expiry - today).days / 365.0
            if T < 0 or T > max_maturity_years:
                continue
            K = row[strike_col]
            iv = row[iv_col]
            if pd.isna(K) or pd.isna(iv):
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
    path = CACHE_CSV_DIR / f"iv_surface_cboe_{sym}.csv"

    # Fallback 1: if CBOE is empty, try cached CSV
    if surface.empty and path.exists():
        try:
            cached = pd.read_csv(path)
            if cached is not None and not cached.empty:
                return cached
        except Exception:
            pass

    # Fallback 2: try Alpaca options snapshot if available
    if surface.empty:
        try:
            from app.model.options.logic import download_options_alpaca

            df_alt = download_options_alpaca(sym)
            if df_alt is not None and not df_alt.empty:
                cols = {c.lower(): c for c in df_alt.columns}
                k_col = cols.get("k") or cols.get("strike")
                t_col = cols.get("t") or cols.get("maturity")
                iv_col = cols.get("iv") or cols.get("sigma") or cols.get("vol")
                type_col = cols.get("type")
                s_col = cols.get("s0") or cols.get("spot") or cols.get("underlying")
                if k_col and t_col and iv_col:
                    df_alt = df_alt.dropna(subset=[k_col, t_col, iv_col]).copy()
                    df_alt["type"] = df_alt[type_col] if type_col else "call"
                    df_alt["S0"] = df_alt[s_col] if s_col else s0
                    df_alt = df_alt.rename(columns={k_col: "K", t_col: "T", iv_col: "iv"})
                    df_alt = df_alt[df_alt["T"] <= max_maturity_years]
                    surface = df_alt[["K", "T", "S0", "iv", "type"]]
        except Exception:
            pass

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        surface.to_csv(path, index=False)
    except Exception:
        pass
    return surface


def fetch_iv_surface(
    ticker: str,
    max_maturity_years: float = 2.0,
    cache: bool = True,
    max_cache_age_hours: float = 12.0,
) -> pd.DataFrame:
    """
    Public entrypoint. Builds IV surface via Yahoo options chain (default) with CSV cache.
    Falls back to existing CBOE/Alpaca paths if Yahoo is empty.
    """
    sym = (ticker or "").strip().upper()
    if not sym:
        return pd.DataFrame()

    cache_path = CACHE_CSV_DIR / f"iv_surface_yahoo_{sym}.csv"
    if cache and cache_path.exists():
        try:
            age_hours = None
            if max_cache_age_hours is not None:
                age_seconds = dt.datetime.now().timestamp() - cache_path.stat().st_mtime
                age_hours = age_seconds / 3600.0
            if age_hours is None or age_hours <= max_cache_age_hours:
                cached = pd.read_csv(cache_path)
                if cached is not None and not cached.empty:
                    return cached
        except Exception:
            pass

    surface = _build_iv_surface_from_yahoo(sym, max_maturity_years=max_maturity_years)

    if (surface is None or surface.empty) and cache_path.exists():
        try:
            cached = pd.read_csv(cache_path)
            if cached is not None and not cached.empty:
                surface = cached
        except Exception:
            pass

    if surface is None or surface.empty:
        surface = _build_iv_surface_from_cboe(sym, max_maturity_years=max_maturity_years)

    if cache and surface is not None and not surface.empty:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            surface.to_csv(cache_path, index=False)
        except Exception:
            pass
    return surface if surface is not None else pd.DataFrame()


def interpolate_surface(df: pd.DataFrame):
    """
    Turn a flat IV DataFrame (K, T, iv) into grid arrays (maturities, strikes, matrix).
    """
    if df is None or df.empty:
        return None, None, None
    cols = {c.lower(): c for c in df.columns}
    k_col = cols.get("k") or cols.get("strike")
    t_col = cols.get("t") or cols.get("maturity") or cols.get("tau")
    iv_col = cols.get("iv") or cols.get("sigma") or cols.get("vol")
    if not (k_col and t_col and iv_col):
        return None, None, None

    df_clean = df[[k_col, t_col, iv_col]].dropna()
    if df_clean.empty:
        return None, None, None

    strikes = sorted(df_clean[k_col].unique())
    maturities = sorted(df_clean[t_col].unique())
    grid = pd.DataFrame(index=maturities, columns=strikes, dtype=float)
    for _, row in df_clean.iterrows():
        grid.at[row[t_col], row[k_col]] = row[iv_col]
    iv_matrix = grid.to_numpy(dtype=float)
    return np.array(maturities, dtype=float), np.array(strikes, dtype=float), iv_matrix


def load_iv_from_csv(file_obj) -> pd.DataFrame:
    """
    Load IV surface from uploaded CSV-like object.
    """
    try:
        df = pd.read_csv(file_obj)
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


__all__ = [
    "fetch_iv_surface",
    "_build_iv_surface_from_yahoo",
    "_build_iv_surface_from_cboe",
    "interpolate_surface",
    "load_iv_from_csv",
]
