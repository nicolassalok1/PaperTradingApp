"""
Alpaca execution wrapper for hedging flows (service layer).

This module is responsible for translating canonical HedgingOrder objects
into Alpaca API calls, and for handling paper vs live mode configuration.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

from app.model.alpaca_orders.service import AlpacaKeys, AlpacaOrdersService
from app.model.trading.hedging import HedgingOrder


def _build_keys(mode: str = "paper") -> AlpacaKeys:
    """
    Build AlpacaKeys with explicit base_url depending on requested mode.

    Paper mode is the default and uses the paper API URL. Live mode is only
    activated when explicitly requested.
    """
    base_env = os.getenv("APCA_API_BASE_URL") or ""
    if mode == "live":
        base_url = (
            base_env
            if "api.alpaca.markets" in base_env and "paper" not in base_env.lower()
            else "https://api.alpaca.markets"
        )
    else:
        # Safe default: paper trading
        base_url = (
            base_env
            if "paper" in base_env.lower()
            else "https://paper-api.alpaca.markets"
        )
    # Reuse API keys from env; let AlpacaKeys validate their presence.
    keys = AlpacaKeys.from_env()
    # Override base_url explicitly to avoid relying on environment only.
    keys.base_url = base_url
    return keys


def _build_service(mode: str = "paper") -> AlpacaOrdersService:
    if mode not in {"paper", "live"}:
        raise ValueError("mode must be 'paper' or 'live'")
    keys = _build_keys(mode)
    return AlpacaOrdersService(keys=keys)


def execute_hedging_orders(
    orders: Iterable[HedgingOrder],
    *,
    mode: str = "paper",
) -> List[Dict[str, Any]]:
    """
    Execute a batch of hedging orders via Alpaca.

    Parameters
    ----------
    orders:
        Iterable of HedgingOrder objects to execute. Only non-zero quantity
        equity orders are currently supported.
    mode:
        "paper" (default) executes against Alpaca paper environment.
        "live" executes against live API and must only be enabled explicitly.

    Returns
    -------
    List of Alpaca order responses as plain dictionaries.
    """
    svc = _build_service(mode)
    executed: List[Dict[str, Any]] = []

    for order in orders:
        if not isinstance(order, HedgingOrder):
            # Defensive: accept dict-like objects with the canonical schema.
            data: Dict[str, Any] = dict(order)  # type: ignore[arg-type]
            order = HedgingOrder(
                symbol=str(data.get("symbol", "")),
                asset_type=str(data.get("asset_type", "equity")),
                side=str(data.get("side", "buy")),
                quantity=float(data.get("quantity", 0.0) or 0.0),
                order_type=str(data.get("order_type", "limit")),
                estimated_price=float(data.get("estimated_price", 0.0) or 0.0),
            )

        if order.asset_type != "equity":
            # For now, only underlying equity hedges are supported.
            continue
        qty = float(order.quantity or 0.0)
        price = float(order.estimated_price or 0.0)
        symbol = (order.symbol or "").strip().upper()
        if not symbol or qty <= 0 or price <= 0:
            continue

        # Use limit orders at the estimated price; the canonical schema keeps
        # track of order_type, but AlpacaOrdersService currently exposes only
        # limit/advanced order helpers.
        resp = svc.submit_limit_order(symbol, qty, price, order.side)
        executed.append(resp)

    return executed


__all__ = ["execute_hedging_orders"]

