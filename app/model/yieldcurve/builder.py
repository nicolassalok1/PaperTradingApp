"""
Yield Curve Downloader (PRO Version)
- Downloads US Treasury rates from FRED (no API key needed)
- Builds latest yield curve (instantaneous)
- Builds full historical dataset
- Saves yield_curve.csv to ./ and ./data/

Output columns:
date, 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
"""

from __future__ import annotations

import logging
from pathlib import Path
from io import StringIO
import time
import pandas as pd
import requests

# ============================================================
# TARGET PATHS
# ============================================================

from app.utils.paths import CACHE_CSV_DIR

ROOT_FILE = Path("yield_curve.csv")
DATA_FILE = CACHE_CSV_DIR / "yield_curve.csv"
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_FILE


# ============================================================
# FRED SERIES
# ============================================================

FRED_SERIES = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
MAX_RETRIES = 3
TIMEOUT = 20  # seconds
BACKOFF = 2  # seconds


# ============================================================
# DOWNLOAD ONE SERIES
# ============================================================


def download_fred_series(series_id: str) -> pd.DataFrame:
    """Download a single FRED time series CSV with simple retry/backoff."""
    url = FRED_URL + series_id
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            break
        except Exception as exc:
            if attempt >= MAX_RETRIES:
                logging.error(f"[FRED] Download failed for {series_id}: {exc}")
                return pd.DataFrame()
            logging.warning(f"[FRED] Retry {attempt}/{MAX_RETRIES} for {series_id}: {exc}")
            time.sleep(BACKOFF * attempt)

    try:
        df = pd.read_csv(StringIO(r.text))
        df.columns = ["date", series_id]
        return df
    except Exception as exc:
        logging.error(f"[FRED] Parsing error for {series_id}: {exc}")
        return pd.DataFrame()


# ============================================================
# BUILD FULL YIELD CURVE HISTORY
# ============================================================


def build_yield_curve() -> pd.DataFrame:
    """Download all maturities and merge them into a single dataframe."""
    frames: list[pd.DataFrame] = []

    for maturity, fred_id in FRED_SERIES.items():
        df = download_fred_series(fred_id)
        if df.empty:
            logging.warning(f"[YieldCurve] No data for {maturity} ({fred_id})")
            continue
        df = df.rename(columns={fred_id: maturity})
        frames.append(df)

    if not frames:
        logging.error("[YieldCurve] All downloads failed.")
        return pd.DataFrame()

    # Merge on date
    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on="date", how="outer")

    # Sort by date and clean
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # numeric cleanup
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save CSVs
    try:
        ROOT_FILE.write_text(df.to_csv(index=False))
        DATA_FILE.write_text(df.to_csv(index=False))
        CACHE_FILE.write_text(df.to_csv(index=False))
        logging.info(f"[YieldCurve] Saved yield curve to {ROOT_FILE}, {DATA_FILE} and {CACHE_FILE}")
    except Exception as exc:
        logging.warning(f"[YieldCurve] Failed to save CSVs: {exc}")

    return df


def _curve_from_points(points: list[tuple[float, float]]):
    clean = [(float(m), float(r)) for m, r in points if m is not None]
    clean = sorted(clean, key=lambda x: x[0])
    if not clean:
        return lambda _maturity: 0.0

    def _interp(maturity: float):
        if maturity is None:
            return 0.0
        x = float(maturity)
        if x <= clean[0][0]:
            return clean[0][1]
        if x >= clean[-1][0]:
            return clean[-1][1]
        for (m1, r1), (m2, r2) in zip(clean[:-1], clean[1:]):
            if m1 <= x <= m2 and m2 != m1:
                w = (x - m1) / (m2 - m1)
                return r1 + w * (r2 - r1)
        return clean[-1][1]

    return _interp


def _tenor_to_years(label: str) -> float | None:
    lbl = str(label).strip().upper()
    if lbl.endswith("M"):
        try:
            return float(lbl[:-1]) / 12.0
        except Exception:
            return None
    if lbl.endswith("Y"):
        try:
            return float(lbl[:-1])
        except Exception:
            return None
    try:
        return float(lbl)
    except Exception:
        return None


def build_curve(raw_points: list[tuple[float, float]] | None = None):
    """
    Return a simple interpolation function r = f(maturity).
    If raw_points are provided (list of (maturity, rate)), use them directly.
    Otherwise, download the latest yield curve and build from the last row.
    """
    if raw_points:
        return _curve_from_points(list(raw_points))

    df = build_yield_curve()
    if df is None or df.empty:
        return _curve_from_points([])

    last = df.iloc[-1]
    points: list[tuple[float, float]] = []
    for col, val in last.items():
        if col == "date":
            continue
        m = _tenor_to_years(col)
        try:
            rate = float(val)
        except Exception:
            continue
        if m is not None and not pd.isna(rate):
            points.append((m, rate))

    return _curve_from_points(points)


# ============================================================
# MAIN SCRIPT ENTRY
# ============================================================

if __name__ == "__main__":
    df = build_yield_curve()
    print(df.tail())
    print("Yield curve built successfully.")
