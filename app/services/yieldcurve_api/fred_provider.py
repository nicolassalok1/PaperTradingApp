from __future__ import annotations

import os
from typing import Dict, List

from app.services.yieldcurve_api.base import safe_get_json

FRED_SERIES: Dict[str, str] = {
    "1M": "DTB1M",
    "3M": "DTB3",
    "6M": "DTB6",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
}

TENOR_YEARS: Dict[str, float] = {"1M": 1 / 12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "5Y": 5.0, "10Y": 10.0}


def _fetch_series(series_id: str, api_key: str | None) -> float | None:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key or "",
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
    }
    data = safe_get_json(url, params=params, timeout=5)
    if data is None:
        return None
    try:
        obs = data.get("observations", [])
        if not obs:
            return None
        val_str = obs[0].get("value")
        val = float(val_str)
        return val / 100.0  # convert to decimal
    except Exception:
        return None


def fetch_usd_nodes_from_fred() -> List[dict]:
    """
    Fetch USD curve nodes from FRED (optional). Returns list of node dicts.
    """
    api_key = os.getenv("FRED_API_KEY")
    nodes: List[dict] = []
    for tenor, series in FRED_SERIES.items():
        val = _fetch_series(series, api_key)
        t_years = TENOR_YEARS.get(tenor)
        if val is None or t_years is None:
            continue
        nodes.append(
            {
                "tenor": tenor,
                "t_years": t_years,
                "zero_rate": val,
                "discount_factor": None,  # computed downstream
            }
        )
    return nodes
