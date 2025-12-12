"""
Controller for Hedger RL Live v2.
Thin wrappers over the rl_live_v2 model service.
"""

from __future__ import annotations

from typing import Any, Dict

from app.model.rl_live_v2.service import (
    get_live_snapshot as _get_live_snapshot,
    get_rl_suggestion as _get_rl_suggestion,
    execute_rl_hedge as _execute_rl_hedge,
    run_backtest as _run_backtest,
)


def get_live_snapshot() -> Dict[str, Any]:
    return _get_live_snapshot()


def get_rl_suggestion(underlying_symbol: str) -> Dict[str, Any]:
    return _get_rl_suggestion(underlying_symbol)


def execute_rl_hedge(underlying_symbol: str) -> Dict[str, Any]:
    return _execute_rl_hedge(underlying_symbol)


def run_backtest(underlying_symbol: str, lookback_days: int = 60) -> Dict[str, Any]:
    return _run_backtest(underlying_symbol, lookback_days=lookback_days)


__all__ = [
    "get_live_snapshot",
    "get_rl_suggestion",
    "execute_rl_hedge",
    "run_backtest",
]
