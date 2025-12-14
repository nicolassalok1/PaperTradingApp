"""
Controller for Hedger v2 (Alpaca-backed).
"""

from __future__ import annotations

from typing import Any, Dict, List

from app.model.hedger_v2.alpaca_client import AlpacaHedgerClient
from app.model.hedger_v2.dqn_hedger import DQNConfig, get_dqn_model_info, load_or_train_dqn_model, suggest_hedge_action

_CLIENT: AlpacaHedgerClient | None = None


def get_client() -> AlpacaHedgerClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AlpacaHedgerClient()
    return _CLIENT


def get_account_snapshot() -> Dict[str, Any]:
    client = get_client()
    return client.get_account()


def get_equity_positions() -> List[Dict[str, Any]]:
    return get_client().get_equity_positions()


def get_option_positions() -> List[Dict[str, Any]]:
    return get_client().get_option_positions()


def manual_order(symbol: str, qty: float, side: str) -> Dict[str, Any]:
    order = get_client().submit_market_order(symbol, qty, side)
    return {
        "id": order.get("id"),
        "symbol": order.get("symbol", symbol),
        "qty": order.get("qty", qty),
        "side": side,
        "status": order.get("status", "submitted"),
    }


def get_dqn_hedge_suggestion(underlying_symbol: str) -> Dict[str, Any]:
    return suggest_hedge_action(get_client(), underlying_symbol)


def execute_dqn_hedge(underlying_symbol: str) -> Dict[str, Any]:
    suggestion = get_dqn_hedge_suggestion(underlying_symbol)
    side = suggestion.get("side")
    delta_qty = float(suggestion.get("delta_qty", 0.0) or 0.0)
    order_resp: Dict[str, Any] | None = None
    if side in {"buy", "sell"} and abs(delta_qty) > 1e-6:
        order_resp = manual_order(suggestion.get("underlying", underlying_symbol), abs(delta_qty), side)
    elif side == "flatten" and abs(delta_qty) > 1e-6:
        # flatten means trade delta_qty (could be negative)
        side_exec = "buy" if delta_qty > 0 else "sell"
        order_resp = manual_order(suggestion.get("underlying", underlying_symbol), abs(delta_qty), side_exec)
    return {"suggestion": suggestion, "order": order_resp}


def _underlying_from_option_symbol(option_symbol: str) -> str:
    """
    Derive the underlying ticker from an OCC-style option symbol.

    Very simple heuristic: take characters up to the first digit.
    """
    sym = (option_symbol or "").strip().upper()
    if not sym:
        return ""
    root = []
    for ch in sym:
        if ch.isdigit():
            break
        root.append(ch)
    return "".join(root) or sym


def get_dqn_hedge_suggestion_for_option(option_symbol: str) -> Dict[str, Any]:
    underlying = _underlying_from_option_symbol(option_symbol)
    return get_dqn_hedge_suggestion(underlying)


def execute_dqn_hedge_for_option(option_symbol: str) -> Dict[str, Any]:
    underlying = _underlying_from_option_symbol(option_symbol)
    return execute_dqn_hedge(underlying)


def get_dqn_model_status() -> Dict[str, Any]:
    return get_dqn_model_info()


def train_dqn_model(
    *,
    train_steps: int | None = None,
    seed: int | None = None,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    cfg = DQNConfig()
    if train_steps is not None:
        try:
            ts = int(train_steps)
        except Exception:
            ts = cfg.train_steps
        ts = max(250, min(ts, 250_000))
        cfg = DQNConfig(**{**cfg.__dict__, "train_steps": ts})
    if seed is not None:
        try:
            s = int(seed)
        except Exception:
            s = cfg.seed
        cfg = DQNConfig(**{**cfg.__dict__, "seed": s})

    return load_or_train_dqn_model(config=cfg, force_retrain=bool(force_retrain))


__all__ = [
    "get_client",
    "get_account_snapshot",
    "get_equity_positions",
    "get_option_positions",
    "manual_order",
    "get_dqn_hedge_suggestion",
    "execute_dqn_hedge",
    "get_dqn_hedge_suggestion_for_option",
    "execute_dqn_hedge_for_option",
    "get_dqn_model_status",
    "train_dqn_model",
]
