"""
Controller for DQN-based options hedging.

Bridges:
  - OptionSpec / portfolio / market data (model layer)
  - DQN hedger agent (model.hedger)
  - Canonical HedgingOrder representation (model.trading.hedging)
  - Alpaca execution wrapper (services.trading.alpaca_execution)

The view layer only interacts with simple controller functions and never
touches the DQN or Alpaca SDK directly.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from app.model.hedger.dqn_agent import DQNAgent
from app.model.hedger.hedger_models import OptionSpec, build_state
from app.model.hedger.service import load_options_portfolio, option_specs_from_portfolio
from app.model.portfolio.positions import load_portfolio_default
from app.model.trading.hedging import HedgingOrder
from app.model.trading.service import get_market_price, floor_4
from app.services.trading.alpaca_execution import execute_hedging_orders as _exec_orders


def load_option_specs() -> List[OptionSpec]:
    """Expose options universe to the view."""
    portfolio = load_options_portfolio()
    return option_specs_from_portfolio(portfolio)


def _current_underlying_position(symbol: str) -> float:
    """
    Return current underlying position (long positive, short negative)
    for the given symbol based on the dashboard portfolio.
    """
    portfolio = load_portfolio_default()
    if not portfolio:
        return 0.0
    data = portfolio.get(symbol) or portfolio.get(symbol.upper()) or portfolio.get(symbol.lower())
    if not data:
        return 0.0
    qty = float(data.get("quantity", 0.0) or 0.0)
    side = str(data.get("side", "long")).lower()
    return qty if side == "long" else -qty


def _build_live_state(option: OptionSpec, hedge_lot: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build the DQN state vector using live market data and current position.

    The state structure mirrors build_state used during training:
      [S_norm, t_norm, moneyness, pos_norm, side]
    where we approximate t_norm as 0 (start of trajectory) and derive
    S, position from live data.
    """
    spot = get_market_price(option.symbol, fallback=option.S0)
    spot = float(spot or option.S0 or 0.0)
    position = _current_underlying_position(option.symbol)

    # Minimal price path: keep the current spot only; build_state only needs
    # price_path[t], t, and len(price_path) to derive the features.
    price_path = np.array([spot], dtype=np.float32)
    state_vec = build_state(option, price_path, t=0, position=position)
    meta = {
        "spot": spot,
        "position": position,
        "hedge_lot": float(hedge_lot),
    }
    return state_vec, meta


def _build_agent(agent_state: Dict[str, Any] | None = None) -> DQNAgent:
    """
    Instantiate a DQNAgent and optionally load a trained state dict.

    If agent_state is None, the untrained weights are used; in practice,
    callers should provide a trained state from the hedger training flow.
    """
    agent = DQNAgent(state_dim=5, action_dim=3)
    if agent_state:
        agent.q_net.load_state_dict(agent_state)
        agent.target_net.load_state_dict(agent_state)
    return agent


def compute_hedging_orders(
    option: OptionSpec,
    hedge_lot: float,
    *,
    agent_state: Dict[str, Any] | None = None,
) -> List[HedgingOrder]:
    """
    Use the DQN hedger as a black box to compute one-step hedging orders.

    Parameters
    ----------
    option:
        OptionSpec describing the option to hedge.
    hedge_lot:
        Size of the underlying hedge trade (absolute quantity).
    agent_state:
        Optional DQN agent state dict as returned by the training service.

    Returns
    -------
    List of HedgingOrder objects. The list may be empty when the DQN
    recommends to hold (no hedge action).
    """
    hedge_lot_val = float(hedge_lot or 0.0)
    if hedge_lot_val <= 0:
        raise ValueError("hedge_lot must be strictly positive")

    state_vec, meta = _build_live_state(option, hedge_lot_val)
    agent = _build_agent(agent_state)

    # eps=0 -> greedy action under the learned value function.
    action_idx = int(agent.act(state_vec, eps=0.0))

    # Map DQN discrete action to trade intent.
    # 0 -> sell hedge_lot, 1 -> hold, 2 -> buy hedge_lot
    if action_idx == 1:
        # No hedge recommended.
        return []
    side = "sell" if action_idx == 0 else "buy"

    spot = float(meta.get("spot", 0.0) or 0.0)
    price = floor_4(spot) if spot > 0 else 0.0
    if price <= 0:
        raise ValueError("Invalid market price for hedging instrument")

    qty = abs(hedge_lot_val)
    symbol = option.symbol.strip().upper()

    order = HedgingOrder(
        symbol=symbol,
        asset_type="equity",
        side=side,
        quantity=qty,
        order_type="limit",
        estimated_price=price,
    )
    return [order]


def execute_orders(
    orders: Iterable[HedgingOrder | Dict[str, Any]],
    *,
    mode: str = "paper",
) -> List[Dict[str, Any]]:
    """
    Execute hedging orders via the Alpaca execution service.

    The view layer must call this only after explicit user confirmation.
    """
    # Delegate to service layer; this is the single orchestration point
    # for Alpaca execution from the hedging feature.
    return _exec_orders(list(orders), mode=mode)


__all__ = [
    "HedgingOrder",
    "load_option_specs",
    "compute_hedging_orders",
    "execute_orders",
]

