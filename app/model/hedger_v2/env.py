"""
Simple hedging environment for a placeholder DQN agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from .alpaca_client import AlpacaHedgerClient


@dataclass
class HedgingEnv:
    client: AlpacaHedgerClient
    underlying_symbol: str
    position_scale: float = 1.0

    def reset(self) -> Dict[str, float]:
        return self._current_state()

    def _current_state(self) -> Dict[str, float]:
        positions = self.client.get_positions()
        equities = positions.get("equities", [])
        options = positions.get("options", [])
        underlying = (self.underlying_symbol or "").strip().upper()

        equity_qty = 0.0
        for p in equities:
            if (p.get("symbol") or "").upper() == underlying:
                equity_qty += float(p.get("qty", 0.0) or 0.0)

        net_delta_proxy = 0.0
        for opt in options:
            qty = float(opt.get("qty", 0.0) or 0.0)
            opt_type = str(opt.get("option_type", opt.get("type", ""))).lower()
            sign = 1.0 if opt_type == "call" else -1.0
            net_delta_proxy += qty * sign

        price = self.client.get_latest_price(underlying)
        account = self.client.get_account()
        cash = float(account.get("cash", 0.0) or 0.0)

        return {
            "underlying_price": price,
            "net_delta_proxy": net_delta_proxy,
            "equity_position": equity_qty,
            "cash": cash,
        }

    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        state = self._current_state()
        equity_pos = state["equity_position"]
        action_map = {
            0: 0.0,  # hold
            1: +1.0 * self.position_scale,
            2: -1.0 * self.position_scale,
            3: -equity_pos,  # flatten
        }
        delta_qty = action_map.get(action, 0.0)
        new_equity_pos = equity_pos + delta_qty
        net_delta_after = state["net_delta_proxy"] + new_equity_pos
        reward = -abs(net_delta_after)
        next_state = {
            **state,
            "equity_position": new_equity_pos,
            "net_delta_proxy": net_delta_after,
        }
        info = {"delta_qty": delta_qty}
        done = False
        return next_state, reward, done, info


@dataclass
class HistoricalHedgingEnv:
    """
    Hedging environment driven by historical price paths (no option greeks required).

    net_delta_proxy is derived from log returns, scaled/clipped to stay in a realistic range.
    """

    prices: Tuple[float, ...]
    position_scale: float = 1.0
    delta_scale: float = 50.0
    transaction_cost: float = 0.02
    max_episode_length: int = 64
    seed: int | None = None

    _rng: np.random.Generator = field(init=False, repr=False)
    _t: int = field(init=False, default=0, repr=False)
    _end: int = field(init=False, default=0, repr=False)
    _equity_pos: float = field(init=False, default=0.0, repr=False)
    _log_returns: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.prices) < 3:
            raise ValueError("HistoricalHedgingEnv requires at least 3 price points.")
        self._rng = np.random.default_rng(self.seed)
        # Use log returns to avoid negative prices; length = len(prices) - 1
        self._log_returns = np.diff(np.log(np.asarray(self.prices, dtype=np.float32)))

    def reset(self, seed: int | None = None) -> Dict[str, float]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Random start so episodes span different segments
        max_start = max(len(self.prices) - int(self.max_episode_length) - 1, 1)
        self._t = int(self._rng.integers(1, max_start + 1))
        self._end = min(len(self.prices) - 1, self._t + int(self.max_episode_length))
        self._equity_pos = float(self._rng.integers(-2, 3))
        return self._state()

    def _state(self) -> Dict[str, float]:
        # Clamp index to stay within bounds
        idx = min(max(self._t, 1), len(self.prices) - 1)
        ret = float(self._log_returns[idx - 1]) if idx - 1 < len(self._log_returns) else 0.0
        net_delta_proxy = float(np.clip(ret * self.delta_scale, -self.delta_scale, self.delta_scale))
        return {
            "underlying_price": float(self.prices[idx]),
            "net_delta_proxy": net_delta_proxy,
            "equity_position": float(self._equity_pos),
            "cash": 0.0,
        }

    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        if action == 1:
            delta_qty = +1.0 * self.position_scale
        elif action == 2:
            delta_qty = -1.0 * self.position_scale
        elif action == 3:
            delta_qty = -self._equity_pos
        else:
            delta_qty = 0.0

        self._equity_pos = float(self._equity_pos + delta_qty)
        self._t += 1
        done = self._t >= self._end or self._t >= len(self.prices) - 1
        next_state = self._state()
        total_delta = float(next_state["net_delta_proxy"] + self._equity_pos)
        reward = -abs(total_delta) - self.transaction_cost * abs(float(delta_qty))
        info = {"delta_qty": float(delta_qty), "total_delta": total_delta, "t": int(self._t)}
        return next_state, float(reward), bool(done), info


__all__ = ["HedgingEnv", "HistoricalHedgingEnv"]
