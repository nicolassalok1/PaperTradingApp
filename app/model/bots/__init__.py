"""
Bots domain package.

Contains reusable, UI-agnostic building blocks for:
- assistant / copilots
- execution bots (e.g., grid/DCA)
- volatility utilities
"""

from __future__ import annotations

from .assistant import ask_portfolio_copilot, get_portfolio_snapshot
from .grid_bot import run_grid_bot_once
from .storage import (
    GridBotConfig,
    delete_grid_config,
    load_grid_configs,
    upsert_grid_config,
)
from .volatility import (
    compute_realized_vol_regime,
    compute_straddle_snapshot,
    compute_straddle_iv_crush,
)

__all__ = [
    # Assistant
    "get_portfolio_snapshot",
    "ask_portfolio_copilot",
    # Grid bot
    "GridBotConfig",
    "load_grid_configs",
    "upsert_grid_config",
    "delete_grid_config",
    "run_grid_bot_once",
    # Volatility
    "compute_realized_vol_regime",
    "compute_straddle_snapshot",
    "compute_straddle_iv_crush",
]

