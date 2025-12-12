"""
Structures de données pour l'UI Options (dataclasses).
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class OptionConfig:
    underlying: str
    strike: float
    maturity: float
    option_type: Literal["call", "put"]
    side: Literal["long", "short"]
    quantity: float
    model: str  # e.g. "CRR", "BS", "MC"
