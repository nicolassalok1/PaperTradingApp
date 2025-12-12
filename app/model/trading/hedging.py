"""
Canonical order representation for hedging flows.

Kept in the model layer so controllers and views can share a stable,
Alpaca-agnostic schema.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class HedgingOrder:
    """
    Canonical hedging order schema exchanged between controller and view.

    All fields are explicit primitives so that no Alpaca SDK objects leak
    into the view layer.
    """

    symbol: str
    asset_type: str  # "equity" / "option"
    side: str  # "buy" / "sell"
    quantity: float
    order_type: str  # "market" / "limit"
    estimated_price: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["HedgingOrder"]

