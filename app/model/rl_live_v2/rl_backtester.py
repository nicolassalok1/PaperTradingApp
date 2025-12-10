"""
RL backtester for Hedger RL Live v2.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .rl_inference import LiveRLAgentV2


class RLBacktester:
    def __init__(self, historical_bars: List[Dict[str, Any]], agent: LiveRLAgentV2, hedging_params: Dict[str, Any]):
        self.bars = historical_bars
        self.agent = agent
        self.hedging_params = hedging_params or {}

    def _state_from_bar(self, bar: Dict[str, Any], equity_position: float, cash: float) -> List[float]:
        # Minimal state: [price, equity_pos, delta, gamma, vega, theta, cash, time_norm]
        price = float(bar.get("close", 0.0) or 0.0)
        time_norm = (pd.to_datetime(bar.get("time")).hour * 60 + pd.to_datetime(bar.get("time")).minute) / (24 * 60)
        # Greeks placeholders from bar meta if available
        delta = float(bar.get("delta", 0.0) or 0.0)
        gamma = float(bar.get("gamma", 0.0) or 0.0)
        vega = float(bar.get("vega", 0.0) or 0.0)
        theta = float(bar.get("theta", 0.0) or 0.0)
        return [price, equity_position, delta, gamma, vega, theta, cash, time_norm]

    def run(self) -> Dict[str, Any]:
        equity_pos = 0.0
        cash = 0.0
        pnl_curve: List[Dict[str, Any]] = []
        hedge_err_curve: List[Dict[str, Any]] = []
        pos_curve: List[Dict[str, Any]] = []

        prev_price = None
        for bar in self.bars:
            state = self._state_from_bar(bar, equity_pos, cash)
            action = self.agent.select_action(state)
            delta_qty = float(action.get("delta_qty", 0.0) or 0.0) * float(self.hedging_params.get("hedge_size", 1.0))
            price = float(bar.get("close", 0.0) or 0.0)

            # Simulate trade
            equity_pos += delta_qty
            cash -= delta_qty * price

            if prev_price is None:
                pnl = 0.0
                hedge_err = 0.0
            else:
                pnl = equity_pos * (price - prev_price)
                hedge_err = abs(equity_pos)  # crude proxy
            prev_price = price

            ts = pd.to_datetime(bar.get("time"))
            pnl_curve.append({"t": ts, "pnl": pnl})
            hedge_err_curve.append({"t": ts, "error": hedge_err})
            pos_curve.append({"t": ts, "qty": equity_pos})

        return {
            "pnl_curve": pnl_curve,
            "hedge_error_curve": hedge_err_curve,
            "positions_curve": pos_curve,
        }


__all__ = ["RLBacktester"]
