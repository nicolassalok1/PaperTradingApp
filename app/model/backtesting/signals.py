from typing import Dict

from app.utils.math_utils import floor_2


def compute_level_prices(
    entry_price: float, drawdown_decimal: float, levels: int = 1
) -> Dict[str, float]:
    """Generate price levels around entry based on drawdown percentage using floor (no rounding)."""
    if levels <= 0:
        return {}
    return {
        str(i + 1): floor_2(entry_price * (1 + drawdown_decimal * (i + 1))) for i in range(levels)
    }
