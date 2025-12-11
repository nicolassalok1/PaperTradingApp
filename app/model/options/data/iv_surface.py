from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List

import numpy as np

import pandas as pd

from app.model.market_data.market_data import fetch_options_details, fetch_spot_price
from app.utils.paths import CACHE_CSV_DIR


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
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        surface.to_csv(path, index=False)
    except Exception:
        pass
    return surface


def fetch_iv_surface(ticker: str, max_maturity_years: float = 2.0) -> pd.DataFrame:
    """
    Public entrypoint. Rebuilds IV surface from CBOE and writes CSV cache.
    """
    return _build_iv_surface_from_cboe(ticker, max_maturity_years=max_maturity_years)


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


__all__ = ["fetch_iv_surface", "_build_iv_surface_from_cboe", "interpolate_surface", "load_iv_from_csv"]
