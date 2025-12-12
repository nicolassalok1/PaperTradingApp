"""
Utilities to fetch risk-free rates and dividend yield.

- get_r(maturity_years): interpolates r(T) from FRED series when available, otherwise DEFAULT_RF.
- get_q(ticker): returns 0.0 (placeholder).
"""

from __future__ import annotations

import os
import time
import math
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np
import requests
from pathlib import Path

# Symbols for risk-free proxies
RATE_SYMBOLS: Dict[str, float] = {
    "^IRX": 0.25,  # 13-week T-Bill ≈ 0.25 years
    "^FVX": 5.0,  # 5-year
    "^TNX": 10.0,  # 10-year
}
FRED_SERIES: Dict[str, str] = {
    "^IRX": "DGS3MO",  # 3-month constant maturity
    "^FVX": "DGS5",  # 5-year
    "^TNX": "DGS10",  # 10-year
}

MAX_RETRIES = 3
SLEEP_BETWEEN = 1.0
DEFAULT_RF = float(os.getenv("DEFAULT_RF_RATE", "0.02"))


def _fetch_from_fred(series_id: str) -> float:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError("Réponse FRED vide")
    last_val = lines[-1].split(",")[-1]
    if last_val.strip() == ".":
        raise RuntimeError("Pas de valeur FRED exploitable")
    return float(last_val)


def _fetch_last_close(symbol: str) -> float:
    """
    Fallback helper kept for completeness; returns default if unavailable.
    """
    return DEFAULT_RF * 100.0


def get_r(maturity_years: float) -> float:
    """
    Interpolate risk-free rate r(T) in decimal from ^IRX (≈0.25y), ^FVX (5y), ^TNX (10y).
    If USE_STATIC_RF_RATE=1, returns DEFAULT_RF_RATE. Otherwise calls the CLI helper
    fetch_r_cli.py (subprocess) to avoid import side-effects inside Streamlit.
    """
    if os.getenv("USE_STATIC_RF_RATE", "0").lower() in {"1", "true", "yes"}:
        return DEFAULT_RF

    cli_path = Path(__file__).resolve().parent / "fetch_r_cli.py"
    if cli_path.exists():
        try:
            res = subprocess.run(
                [sys.executable, str(cli_path), str(float(maturity_years))],
                capture_output=True,
                text=True,
                check=True,
                timeout=8,
            )
            val = float(res.stdout.strip())
            if math.isfinite(val) and val > 0:
                return val
        except Exception:
            pass

    points: List[Tuple[float, float]] = []
    for sym, mat in RATE_SYMBOLS.items():
        fred_id = FRED_SERIES.get(sym)
        if fred_id:
            try:
                val = _fetch_from_fred(fred_id) / 100.0
                if math.isfinite(val) and val > 0:
                    points.append((mat, val))
                    continue
            except Exception:
                continue

    if not points:
        return DEFAULT_RF  # fallback

    # Sort by maturity
    points.sort(key=lambda x: x[0])
    maturities = np.array([p[0] for p in points], dtype=float)
    rates = np.array([p[1] for p in points], dtype=float)

    T = float(maturity_years)
    if len(points) == 1:
        val = float(rates[0])
        return val if math.isfinite(val) and val > 0 else DEFAULT_RF
    # Clamp to available range then linear interpolation
    T_clamped = np.clip(T, maturities.min(), maturities.max())
    val = float(np.interp(T_clamped, maturities, rates))
    return val if math.isfinite(val) and val > 0 else DEFAULT_RF


def get_q(ticker: str) -> float:
    """
    Return dividend yield (continuous approx) for the given ticker.
    Fallback to 0.0.
    """
    return 0.0


if __name__ == "__main__":
    r = get_r(0.5)
    q = get_q("AAPL")
    print(r, q)
