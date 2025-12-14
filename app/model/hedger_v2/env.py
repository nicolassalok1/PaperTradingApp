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
class SyntheticHedgingEnv:
    """
    Toy hedging environment used to train the DQN offline (no Alpaca dependency).

    State is kept intentionally minimal:
    - net_delta_proxy: exogenous option delta exposure (drifts stochastically)
    - equity_position: hedge position (shares, delta=1 per share)

    Reward penalizes absolute residual delta + transaction costs.
    """

    max_steps: int = 32
    max_abs_delta: float = 10.0
    position_scale: float = 1.0
    delta_drift_std: float = 0.60
    transaction_cost: float = 0.02
    seed: int | None = 42

    _rng: np.random.Generator = field(init=False, repr=False)
    _t: int = field(init=False, default=0, repr=False)
    _net_delta: float = field(init=False, default=0.0, repr=False)
    _equity_pos: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> Dict[str, float]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._net_delta = float(self._rng.uniform(-self.max_abs_delta, self.max_abs_delta))
        # start hedge position around 0 to let the agent learn adjustments
        self._equity_pos = float(self._rng.integers(-2, 3))
        return self._state()

    def _state(self) -> Dict[str, float]:
        return {
            "net_delta_proxy": float(self._net_delta),
            "equity_position": float(self._equity_pos),
        }

    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, Dict[str, Any]]:
        if action == 1:
            delta_qty = +1.0 * self.position_scale
        elif action == 2:
            delta_qty = -1.0 * self.position_scale
        elif action == 3:
            # flatten hedge leg (close stock position)
            delta_qty = -self._equity_pos
        else:
            delta_qty = 0.0

        self._equity_pos = float(self._equity_pos + delta_qty)
        # option delta drifts
        self._net_delta = float(
            np.clip(
                self._net_delta + self._rng.normal(0.0, self.delta_drift_std),
                -self.max_abs_delta,
                self.max_abs_delta,
            )
        )

        total_delta = float(self._net_delta + self._equity_pos)
        reward = -abs(total_delta) - self.transaction_cost * abs(float(delta_qty))

        self._t += 1
        done = self._t >= int(self.max_steps)
        info = {"delta_qty": float(delta_qty), "total_delta": total_delta, "t": self._t}
        return self._state(), float(reward), bool(done), info


__all__ = ["HedgingEnv", "SyntheticHedgingEnv"]
