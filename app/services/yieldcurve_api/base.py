from __future__ import annotations

import requests
from typing import Any, Dict, List, Protocol


class YieldCurveApiProvider(Protocol):
    def fetch_nodes(self) -> List[dict]:
        ...


def safe_get_json(url: str, params: Dict[str, Any] | None = None, timeout: float = 5.0) -> Dict[str, Any] | None:
    """
    Safe HTTP GET returning JSON or None. Swallows all exceptions.
    """
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None
