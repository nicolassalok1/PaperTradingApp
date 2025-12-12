"""
Yield curve controller: thin wrappers to model.yieldcurve services.
"""

from __future__ import annotations

from app.model.yieldcurve.service import (
    available_currencies,
    build_curve,
    compute_forward_price,
    compute_forward_rate,
    compute_forward_rates,
    compute_instantaneous_rate,
    compute_instantaneous_rates,
    get_rate,
    get_active_curve,
    get_curve_snapshot,
    get_risk_free_rate,
    get_spot,
    interpolate_curve_rate,
    load_curve,
    load_forwards,
    prepare_forward_rows,
    refresh_curve,
    save_forwards,
    yield_curve_cache_file,
)
from app.model.market_data.history import clear_closing_history_cache
from app.utils.math_utils import floor_3, floor_4

__all__ = [
    "build_curve",
    "compute_forward_price",
    "compute_forward_rate",
    "compute_forward_rates",
    "compute_instantaneous_rate",
    "compute_instantaneous_rates",
    "get_rate",
    "available_currencies",
    "get_active_curve",
    "get_curve_snapshot",
    "get_risk_free_rate",
    "get_spot",
    "interpolate_curve_rate",
    "load_curve",
    "load_forwards",
    "prepare_forward_rows",
    "refresh_curve",
    "save_forwards",
    "yield_curve_cache_file",
    "clear_closing_history_cache",
    "floor_3",
    "floor_4",
]
