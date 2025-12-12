from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List

import pandas as pd

from app.model.market_data.market_data import fetch_options_details, fetch_spot_price
from app.utils.paths import CACHE_CSV_DIR


def _decode_opra_expiry(opra: str) -> dt.date | None:
    """
    Extract expiry date from OPRA code (…YYMMDDCTTTTTTTT).
    """
    if not opra:
        return None
    code = str(opra)
    if len(code) < 15:
        return None
    try:
        expiry_str = code[-15:-9]
        return dt.datetime.strptime(expiry_str, "%y%m%d").date()
    except Exception:
        return None


def _build_iv_surface_from_cboe(ticker: str, max_maturity_years: float = 2.0) -> pd.DataFrame:
    """
    Build IV surface from CBOE call/put chains.
    """
    calls_df, puts_df, spot, _, _ = fetch_options_details(ticker)
    s0 = spot if pd.notna(spot) else fetch_spot_price(ticker)
    today = dt.date.today()
    sym = (ticker or "").strip().upper()

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
    """Public entrypoint building an IV surface from CBOE chains."""
    return _build_iv_surface_from_cboe(ticker, max_maturity_years=max_maturity_years)


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


__all__ = ["fetch_iv_surface", "_build_iv_surface_from_cboe", "interpolate_surface", "load_iv_from_csv"]
