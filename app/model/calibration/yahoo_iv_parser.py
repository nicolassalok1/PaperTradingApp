from __future__ import annotations

import datetime as dt
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_STRIKE_ALIASES = {"strike", "k", "strike_price"}
_EXPIRY_ALIASES = {"expiration", "expiry", "maturity", "expirationdate"}
_IV_ALIASES = {"impliedvolatility", "implied_volatility", "iv"}
_S0_ALIASES = {"underlyingprice", "s0", "spot"}


def _read_csv(csv_bytes_or_path: Any) -> pd.DataFrame:
    if isinstance(csv_bytes_or_path, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(csv_bytes_or_path))
    if isinstance(csv_bytes_or_path, (str, Path)):
        return pd.read_csv(csv_bytes_or_path)
    if isinstance(csv_bytes_or_path, pd.DataFrame):
        return csv_bytes_or_path.copy()
    raise ValueError("Unsupported CSV input type.")


def _find_col(df: pd.DataFrame, aliases: set[str]) -> str | None:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias in cols:
            return cols[alias]
    return None


def _coerce_date(val: Any) -> pd.Timestamp:
    try:
        return pd.to_datetime(val, errors="coerce")
    except Exception:
        return pd.NaT


def parse_yahoo_iv_csv(csv_bytes_or_path, asof_date=None, S0: float | None = None) -> pd.DataFrame:
    """
    Parse a Yahoo-style options IV CSV and normalize to columns: moneyness, ttm, iv.
    Requires strike, expiration, impliedVolatility (or implied_volatility).
    If S0 is not found in the CSV, provide it via the S0 argument.
    """
    df_raw = _read_csv(csv_bytes_or_path)
    strike_col = _find_col(df_raw, _STRIKE_ALIASES)
    expiry_col = _find_col(df_raw, _EXPIRY_ALIASES)
    iv_col = _find_col(df_raw, _IV_ALIASES)
    s0_col = _find_col(df_raw, _S0_ALIASES)

    if not (strike_col and expiry_col and iv_col):
        raise ValueError("CSV must include strike, expiration, and impliedVolatility columns.")

    s0_val = S0
    if s0_val is None and s0_col:
        try:
            s0_val = float(pd.to_numeric(df_raw[s0_col], errors="coerce").dropna().iloc[0])
        except Exception:
            s0_val = None

    if s0_val is None:
        raise ValueError("Underlying price (S0) missing; provide it in payload or CSV.")

    try:
        asof_ts = _coerce_date(asof_date) if asof_date is not None else pd.Timestamp(dt.date.today())
    except Exception:
        asof_ts = pd.Timestamp(dt.date.today())

    strikes = pd.to_numeric(df_raw[strike_col], errors="coerce")
    expiries = df_raw[expiry_col].apply(_coerce_date)
    ivs = pd.to_numeric(df_raw[iv_col], errors="coerce")

    df = pd.DataFrame({"strike": strikes, "expiration": expiries, "iv": ivs})
    df = df.dropna(subset=["strike", "expiration", "iv"])
    if df.empty:
        return pd.DataFrame(columns=["moneyness", "ttm", "iv"])

    df["moneyness"] = df["strike"].astype(float) / float(s0_val)
    df["ttm"] = (df["expiration"] - asof_ts) / np.timedelta64(365, "D")
    df = df.dropna(subset=["moneyness", "ttm", "iv"])
    df = df[df["ttm"] > 0]

    return df[["moneyness", "ttm", "iv"]].reset_index(drop=True)


__all__ = ["parse_yahoo_iv_csv"]
