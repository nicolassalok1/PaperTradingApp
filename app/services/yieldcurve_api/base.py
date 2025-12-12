from __future__ import annotations

from typing import List, Protocol


class YieldCurveApiProvider(Protocol):
    def fetch_nodes(self) -> List[dict]:
        ...
