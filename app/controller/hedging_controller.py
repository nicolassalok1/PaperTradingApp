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

from typing import Any, Dict, Iterable, List

from app.model.hedger.hedger_models import OptionSpec
from app.model.hedger.service import (
    load_options_portfolio,
    option_specs_from_portfolio,
    compute_hedging_orders as _compute_hedging_orders,
)
from app.model.trading.hedging import HedgingOrder
from app.model.trading.alpaca_execution import execute_hedging_orders as _exec_orders


def load_option_specs() -> List[OptionSpec]:
    """Expose options universe to the view."""
    portfolio = load_options_portfolio()
    return option_specs_from_portfolio(portfolio)


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
    return _compute_hedging_orders(option, hedge_lot, agent_state=agent_state)


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
