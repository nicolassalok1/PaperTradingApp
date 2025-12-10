"""
Simple hedging environment for a placeholder DQN agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

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


__all__ = ["HedgingEnv"]
