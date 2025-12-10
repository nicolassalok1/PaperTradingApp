import math
from typing import Union

Number = Union[int, float]


def floor_n(v: Number, decimals: int = 2) -> float:
    """Floor to a given number of decimals (never rounds)."""
    v = float(v or 0.0)
    factor = 10 ** max(decimals, 0)
    return math.floor(v * factor) / factor


def floor_2(v: Number) -> float:
    """Floor (not round) to 2 decimals."""
    return floor_n(v, 2)


def floor_3(v: Number) -> float:
    """Floor (not round) to 3 decimals."""
    return floor_n(v, 3)


def floor_4(v: Number) -> float:
    """Floor (not round) to 4 decimals."""
    return floor_n(v, 4)
