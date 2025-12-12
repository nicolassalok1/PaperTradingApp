from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def tenor_to_years(tenor: str | None) -> float | None:
    if tenor is None:
        return None
    t = tenor.strip().upper()
    if not t:
        return None
    try:
        if t.endswith("Y"):
            return float(t[:-1])
        if t.endswith("M"):
            return float(t[:-1]) / 12.0
        if t.endswith("D"):
            return float(t[:-1]) / 365.0
    except Exception:
        return None
    try:
        return float(t)
    except Exception:
        return None


def _normalize_row(row: Dict) -> Dict:
    tenor = row.get("tenor") or row.get("Tenor")
    t_years = row.get("t_years") or row.get("years") or row.get("T")
    zero_rate = row.get("zero_rate_cc") or row.get("zero_rate") or row.get("zero")
    discount_factor = row.get("discount_factor") or row.get("df")
    if t_years is None:
        t_years = tenor_to_years(tenor)
    return {
        "tenor": tenor,
        "t_years": t_years,
        "zero_rate": zero_rate,
        "discount_factor": discount_factor,
    }


def load_nodes_from_file(path: Path) -> List[Dict]:
    if not path or not path.exists():
        return []
    nodes: List[Dict] = []
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and "nodes" in data:
                data = data["nodes"]
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        nodes.append(_normalize_row(row))
        except Exception:
            return []
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            return []
        for _, row in df.iterrows():
            nodes.append(_normalize_row(row.to_dict()))
    return nodes
