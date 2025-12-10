"""
IV Surface Loader
- Fetch IV from yfinance if available
- Load from CSV
- Interpolate on strike/maturity grid
- Return clean, numeric grids
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.interpolate import griddata


def fetch_iv_surface(ticker: str):
    """
    Fetch implied vol surface from yfinance.
    Returns dataframe or None.
    """
    try:
        opt = yf.Ticker(ticker)
        chain = opt.option_chain
        # In yfinance >= 0.2, option_chain returns function-like
        expiries = opt.options
        rows = []
        for exp in expiries:
            chain = opt.option_chain(exp)
            calls = chain.calls
            for _, row in calls.iterrows():
                rows.append(
                    {
                        "expiration": exp,
                        "strike": float(row["strike"]),
                        "iv": float(row.get("impliedVolatility") or 0.0),
                    }
                )
        return pd.DataFrame(rows)
    except Exception:
        return None


def load_iv_from_csv(path: str):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


def interpolate_surface(df: pd.DataFrame):
    """
    Take a raw IV dataframe:
      columns: expiration, strike, iv
    Output: (maturities, strikes, iv_matrix)
    """
    if df is None or df.empty:
        return None, None, None

    df = df.copy()
    df["T"] = pd.to_datetime(df["expiration"])
    df["T"] = (df["T"] - pd.Timestamp.today()).dt.days / 365.0
    df = df[df["T"] > 0]

    maturities = np.sort(df["T"].unique())
    strikes = np.sort(df["strike"].unique())

    grid_T, grid_K = np.meshgrid(maturities, strikes, indexing="ij")
    points = df[["T", "strike"]].values
    values = df["iv"].values

    iv_grid = griddata(points, values, (grid_T, grid_K), method="linear")

    return maturities, strikes, iv_grid


__all__ = [
    "fetch_iv_surface",
    "load_iv_from_csv",
    "interpolate_surface",
]
