"""
Controller for the Exercices tab — thin wrappers over the exercise model engines.

Keeps the view layer free of any direct model import, mirroring the other
controllers. The numerical engine (`engine.backtest`) is the validated reference
and is called, never modified.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from app.model.exercises.portfolio_allocation import BUNDLED_CSV, engine, yahoo_data

# Mandate parameters the UI is allowed to override (engine.CFG keys).
PA_PARAM_KEYS = ("lookback", "vol_target", "gross_cap", "name_cap", "var_limit")


def pa_default_config() -> dict[str, Any]:
    """Mandate defaults — reproduce the take-home exactly."""
    return dict(engine.CFG)


def pa_build_config(**overrides: float) -> dict[str, Any]:
    """Start from the mandate defaults, apply UI overrides for the allowed keys."""
    cfg = dict(engine.CFG)
    for key in PA_PARAM_KEYS:
        val = overrides.get(key)
        if val is not None:
            cfg[key] = val
    cfg["lookback"] = int(cfg["lookback"])
    return cfg


def pa_bundled_csv_path() -> str:
    """Absolute path to the bundled `spx_vix_daily.csv` (1990 -> 2023)."""
    return str(BUNDLED_CSV)


def pa_load_csv(path_or_buffer) -> pd.DataFrame:
    """Parse a `Date,SPX,VIX` CSV via the validated engine loader (path or buffer)."""
    return engine.load_csv(path_or_buffer)


def pa_fetch_yahoo(start: str = "1990-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch ^GSPC/^VIX daily closes server-side (no backend, no CORS)."""
    return yahoo_data.fetch_yahoo_prices(start=start, end=end)


def pa_run_backtest(prices: pd.DataFrame, cfg: dict[str, Any], sample: int = 5):
    """Run the validated backtest; return (metrics, series). `df` is dropped."""
    metrics, series, _df = engine.backtest(prices, cfg, sample=sample)
    return metrics, series


__all__ = [
    "PA_PARAM_KEYS",
    "pa_default_config",
    "pa_build_config",
    "pa_bundled_csv_path",
    "pa_load_csv",
    "pa_fetch_yahoo",
    "pa_run_backtest",
]
