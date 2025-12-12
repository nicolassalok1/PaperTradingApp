from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class YieldCurve(ABC):
    """
    Interface for yield curves consumed by pricing and UI layers.
    """

    @property
    @abstractmethod
    def maturities(self) -> List[float]:
        ...

    @abstractmethod
    def zero_rate(self, T_years: float) -> float:
        ...

    @abstractmethod
    def discount_factor(self, T_years: float) -> float:
        ...

    @abstractmethod
    def forward_rate(self, start_years: float, end_years: float) -> float | None:
        ...
