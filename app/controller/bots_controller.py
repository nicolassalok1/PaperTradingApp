"""
Controller for the top-level Bots tab.

Thin wrappers over `app.model.bots.*` to preserve MVC boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from app.model.bots.assistant import ask_portfolio_copilot, get_portfolio_snapshot
from app.model.bots.grid_bot import GridBotRunReport, run_grid_bot_once
from app.model.bots.storage import GridBotConfig, delete_grid_config, load_grid_configs, upsert_grid_config
from app.model.bots.volatility import (
    compute_markov_vol_transition,
    compute_realized_vol_mean_reversion,
    compute_realized_vol_regime,
    compute_straddle_iv_crush,
    compute_straddle_snapshot,
)
from app.model.market_data.market_data import fetch_spot_price as _fetch_spot_price


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


# --- Market data convenience ------------------------------------------------

def get_spot_price(symbol: str) -> float | None:
    try:
        return _fetch_spot_price(symbol)
    except Exception:
        return None


# --- Assistant --------------------------------------------------------------

def get_snapshot() -> dict[str, Any]:
    return get_portfolio_snapshot()


def ask(question: str, snapshot: dict[str, Any] | None = None) -> str:
    return ask_portfolio_copilot(question, snapshot=snapshot)


# --- Grid / DCA bot ---------------------------------------------------------

def list_grid_configs() -> dict[str, GridBotConfig]:
    return load_grid_configs()


def save_grid_config(
    *,
    symbol: str,
    enabled: bool,
    qty: float,
    n_levels: int,
    step_pct: float,
    dry_run: bool,
    reference: str = "spot",
    time_in_force: str = "gtc",
) -> GridBotConfig:
    cfg = GridBotConfig(
        symbol=symbol,
        enabled=enabled,
        qty=qty,
        n_levels=n_levels,
        step_pct=step_pct,
        reference=reference,
        time_in_force=time_in_force,
        dry_run=dry_run,
    )
    return upsert_grid_config(cfg)


def remove_grid_config(symbol: str) -> None:
    delete_grid_config(symbol)


def run_grid_once(
    config: GridBotConfig,
    *,
    allow_submit: bool = False,
    allow_live: bool = False,
) -> dict[str, Any]:
    report: GridBotRunReport = run_grid_bot_once(
        config,
        allow_submit=allow_submit,
        allow_live=allow_live,
    )
    return _jsonable(report)


# --- Volatility tools -------------------------------------------------------

def realized_vol_regime(
    symbol: str,
    *,
    period: str = "2y",
    window: int = 20,
    annualization: int = 252,
) -> dict[str, Any]:
    return compute_realized_vol_regime(
        symbol,
        period=period,
        window=window,
        annualization=annualization,
    )


def realized_vol_mean_reversion(
    symbol: str,
    *,
    period: str = "2y",
    vol_window: int = 20,
    forward_window: int = 30,
    annualization: int = 252,
) -> dict[str, Any]:
    return compute_realized_vol_mean_reversion(
        symbol,
        period=period,
        vol_window=vol_window,
        forward_window=forward_window,
        annualization=annualization,
    )


def markov_vol_transition(
    symbol: str,
    *,
    period: str = "2y",
    window: int = 20,
    annualization: int = 252,
    n_states: int = 3,
) -> dict[str, Any]:
    return compute_markov_vol_transition(
        symbol,
        period=period,
        window=window,
        annualization=annualization,
        n_states=n_states,
    )


def straddle_snapshot(
    *,
    spot: float,
    strike: float,
    days_to_expiry: int,
    iv: float,
    r: float = 0.0,
    q: float = 0.0,
) -> dict[str, Any]:
    return _jsonable(
        compute_straddle_snapshot(
            spot=spot,
            strike=strike,
            days_to_expiry=days_to_expiry,
            iv=iv,
            r=r,
            q=q,
        )
    )


def straddle_iv_crush(
    *,
    pre_spot: float,
    post_spot: float,
    strike: float,
    days_to_expiry: int,
    pre_iv: float,
    post_iv: float,
    r: float = 0.0,
    q: float = 0.0,
    qty: float = 1.0,
) -> dict[str, Any]:
    return _jsonable(
        compute_straddle_iv_crush(
            pre_spot=pre_spot,
            post_spot=post_spot,
            strike=strike,
            days_to_expiry=days_to_expiry,
            pre_iv=pre_iv,
            post_iv=post_iv,
            r=r,
            q=q,
            qty=qty,
        )
    )


__all__ = [
    # Market
    "get_spot_price",
    # Assistant
    "get_snapshot",
    "ask",
    # Grid bot
    "list_grid_configs",
    "save_grid_config",
    "remove_grid_config",
    "run_grid_once",
    # Vol tools
    "realized_vol_regime",
    "realized_vol_mean_reversion",
    "markov_vol_transition",
    "straddle_snapshot",
    "straddle_iv_crush",
]
