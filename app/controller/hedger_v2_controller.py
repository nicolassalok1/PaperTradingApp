"""
Controller for Hedger v2 (Alpaca-backed).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from app.model.hedger_v2.alpaca_client import AlpacaHedgerClient
from app.model.hedger_v2.dqn_hedger import DQNConfig, get_dqn_model_info, load_or_train_dqn_model, suggest_hedge_action

_CLIENT: AlpacaHedgerClient | None = None


def get_client() -> AlpacaHedgerClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = AlpacaHedgerClient()
    return _CLIENT


def get_status() -> Dict[str, Any]:
    client = get_client()
    return {"ready": client.is_ready(), "offline": client.offline}


def get_account_snapshot() -> Dict[str, Any]:
    client = get_client()
    return client.get_account()


def get_equity_positions() -> List[Dict[str, Any]]:
    return get_client().get_equity_positions()


def get_option_positions() -> List[Dict[str, Any]]:
    return get_client().get_option_positions()


def get_option_underlyings() -> List[str]:
    """Unique underlyings inferred from current option positions."""
    underlyings: set[str] = set()
    for opt in get_option_positions():
        sym = (opt.get("symbol") or "").strip().upper()
        if not sym:
            continue
        underlyings.add(_underlying_from_option_symbol(sym))
    return sorted(u for u in underlyings if u)


def manual_order(symbol: str, qty: float, side: str) -> Dict[str, Any]:
    client = get_client()
    if not client.is_ready():
        raise RuntimeError("Alpaca credentials missing or client offline; cannot send orders.")
    order = client.submit_market_order(symbol, qty, side)
    return {
        "id": order.get("id"),
        "symbol": order.get("symbol", symbol),
        "qty": order.get("qty", qty),
        "side": side,
        "status": order.get("status", "submitted"),
    }


def get_dqn_hedge_suggestion(underlying_symbol: str) -> Dict[str, Any]:
    sym = (underlying_symbol or "").strip().upper()
    if not sym:
        raise ValueError("Underlying symbol is required for hedge suggestion.")
    return suggest_hedge_action(get_client(), underlying_symbol)


def _extract_available_qty_from_error(exc: Exception) -> float:
    """
    Attempt to parse an 'available' quantity from Alpaca error payloads.
    """
    msg = str(exc)
    matches = re.findall(r"available[\"']?:\\?\"?([0-9]+(?:\\.[0-9]+)?)", msg)
    if not matches:
        return 0.0
    try:
        return float(matches[0])
    except Exception:
        return 0.0


def _manual_order_with_fallback(symbol: str, qty: float, side: str) -> Dict[str, Any]:
    """
    Submit order; if insufficient qty for a sell/flatten, clamp to available and retry once.
    Always returns a dict (never raises) so the UI can surface a clear error.
    """
    try:
        return manual_order(symbol, qty, side)
    except Exception as exc:
        avail = _extract_available_qty_from_error(exc)
        side_lower = (side or "").lower()
        if side_lower == "sell" and avail > 1e-6 and avail < float(qty) - 1e-6:
            try:
                resp = manual_order(symbol, avail, side)
                resp["note"] = f"Clamped to available quantity {avail}"
                return resp
            except Exception:
                return {"status": "error", "message": str(exc), "available_qty": avail}
        return {"status": "error", "message": str(exc), "available_qty": avail}


def execute_dqn_hedge(underlying_symbol: str) -> Dict[str, Any]:
    if not get_client().is_ready():
        raise RuntimeError("Alpaca credentials missing or client offline; cannot execute hedge.")
    suggestion = get_dqn_hedge_suggestion(underlying_symbol)
    side = suggestion.get("side")
    delta_qty = float(suggestion.get("delta_qty", 0.0) or 0.0)
    order_resp: Dict[str, Any] | None = None
    if side in {"buy", "sell"} and abs(delta_qty) > 1e-6:
        order_resp = _manual_order_with_fallback(
            suggestion.get("underlying", underlying_symbol),
            abs(delta_qty),
            side,
        )
    elif side == "flatten" and abs(delta_qty) > 1e-6:
        # flatten means trade delta_qty (could be negative)
        side_exec = "buy" if delta_qty > 0 else "sell"
        order_resp = _manual_order_with_fallback(
            suggestion.get("underlying", underlying_symbol),
            abs(delta_qty),
            side_exec,
        )
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


def get_portfolio_option_hedges(execute: bool = False) -> List[Dict[str, Any]]:
    """
    Suggest (and optionally execute) hedges for all underlyings present in option positions.
    """
    underlyings = get_option_underlyings()
    results: List[Dict[str, Any]] = []
    for u in underlyings:
        suggestion = get_dqn_hedge_suggestion(u)
        order_resp = None
        if execute and suggestion.get("side") in {"buy", "sell", "flatten"}:
            delta_qty = float(suggestion.get("delta_qty", 0.0) or 0.0)
            if abs(delta_qty) > 1e-6 and suggestion.get("side") != "none":
                exec_side = suggestion["side"]
                if exec_side == "flatten":
                    exec_side = "buy" if delta_qty > 0 else "sell"
                try:
                    order_resp = _manual_order_with_fallback(
                        suggestion.get("underlying", u),
                        abs(delta_qty),
                        exec_side,
                    )
                except Exception as exc:
                    order_resp = {"status": "error", "message": str(exc)}
        results.append({"underlying": u, "suggestion": suggestion, "order": order_resp})
    return results


def get_dqn_model_status() -> Dict[str, Any]:
    return get_dqn_model_info()


def train_dqn_model(
    *,
    train_steps: int | None = None,
    seed: int | None = None,
    force_retrain: bool = False,
    historical_symbol: str | None = None,
    historical_timeframe: str | None = None,
    historical_lookback_days: int | None = None,
    historical_episode_length: int | None = None,
) -> Dict[str, Any]:
    base_cfg = DQNConfig()
    cfg_kwargs = {**base_cfg.__dict__}
    if train_steps is not None:
        try:
            ts = int(train_steps)
        except Exception:
            ts = base_cfg.train_steps
        ts = max(250, min(ts, 250_000))
        cfg_kwargs["train_steps"] = ts
    if seed is not None:
        try:
            s = int(seed)
        except Exception:
            s = base_cfg.seed
        cfg_kwargs["seed"] = s

    if historical_symbol:
        cfg_kwargs["historical_symbol"] = (historical_symbol or "").strip().upper()
    if historical_timeframe:
        cfg_kwargs["historical_timeframe"] = str(historical_timeframe).strip()
    if historical_lookback_days is not None:
        try:
            lb = int(historical_lookback_days)
        except Exception:
            lb = base_cfg.historical_lookback_days
        cfg_kwargs["historical_lookback_days"] = max(30, min(lb, 1_500))
    if historical_episode_length is not None:
        try:
            elen = int(historical_episode_length)
        except Exception:
            elen = base_cfg.historical_episode_length
        cfg_kwargs["historical_episode_length"] = max(8, min(elen, 512))

    cfg = DQNConfig(**cfg_kwargs)

    return load_or_train_dqn_model(config=cfg, force_retrain=bool(force_retrain))


__all__ = [
    "get_client",
    "get_status",
    "get_account_snapshot",
    "get_equity_positions",
    "get_option_positions",
    "get_option_underlyings",
    "manual_order",
    "get_dqn_hedge_suggestion",
    "execute_dqn_hedge",
    "get_dqn_hedge_suggestion_for_option",
    "execute_dqn_hedge_for_option",
    "get_portfolio_option_hedges",
    "get_dqn_model_status",
    "train_dqn_model",
]
