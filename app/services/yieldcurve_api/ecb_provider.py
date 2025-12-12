from __future__ import annotations

from typing import List

from app.services.yieldcurve_api.base import safe_get_json

# ECB Statistical Data Warehouse: daily par yield curve (example for EUR, AAA-rated)
# We use a lightweight endpoint; if unavailable, this provider returns empty.
ECB_BASE = "https://data-api.ecb.europa.eu/service/data"
SERIES = {
    # maturity code suffix in ECB YC dataset (years)
    "0.5Y": ("YC/B.U2.EUR.4F.G_N.A.SV_C_YM.SR_6M", 0.5),
    "1Y": ("YC/B.U2.EUR.4F.G_N.A.SV_C_YM.SR_1Y", 1.0),
    "2Y": ("YC/B.U2.EUR.4F.G_N.A.SV_C_YM.SR_2Y", 2.0),
    "5Y": ("YC/B.U2.EUR.4F.G_N.A.SV_C_YM.SR_5Y", 5.0),
    "10Y": ("YC/B.U2.EUR.4F.G_N.A.SV_C_YM.SR_10Y", 10.0),
}


def _fetch_single_series(series_code: str) -> float | None:
    url = f"{ECB_BASE}/{series_code}"
    params = {"lastNObservations": 1, "format": "jsondata"}
    data = safe_get_json(url, params=params, timeout=5)
    if data is None:
        return None
    try:
        obs = data.get("data", {}).get("dataSets", [{}])[0].get("series", {})
        # series key "0:0:0:0:0" typically holds the observation
        first_series = next(iter(obs.values()))
        values = first_series.get("observations", {})
        first_obs = next(iter(values.values()))
        val = float(first_obs[0])
        return val / 100.0
    except Exception:
        return None


def fetch_eur_nodes_from_ecb() -> List[dict]:
    nodes: List[dict] = []
    for tenor, (series_code, t_years) in SERIES.items():
        val = _fetch_single_series(series_code)
        if val is None:
            continue
        nodes.append(
            {
                "tenor": tenor,
                "t_years": float(t_years),
                "zero_rate": float(val),
                "discount_factor": None,
            }
        )
    return nodes
