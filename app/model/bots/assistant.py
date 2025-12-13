from __future__ import annotations

import json
from typing import Any

from app.model.ai.chatgpt import chatgpt_response
from app.model.alpaca_orders.service import AlpacaOrdersService
from app.model.dashboard_v2.engine import DashboardV2Client


def get_portfolio_snapshot() -> dict[str, Any]:
    """
    Portfolio snapshot used by the Bots assistant.

    Uses DashboardV2Client for a safe offline fallback when keys are not configured.
    Adds open orders when Alpaca trading keys are available.
    """
    dash = DashboardV2Client()
    snapshot: dict[str, Any] = {
        "account": dash.get_account_summary(),
        "spot_positions": dash.get_spot_positions(),
        "option_positions": dash.get_option_positions(),
        "open_orders": [],
        "warnings": [],
    }

    try:
        snapshot["open_orders"] = AlpacaOrdersService().get_open_orders()
    except Exception as exc:  # noqa: BLE001
        snapshot["warnings"].append(f"Open orders unavailable: {exc}")

    return snapshot


def ask_portfolio_copilot(question: str, snapshot: dict[str, Any] | None = None) -> str:
    q = (question or "").strip()
    if not q:
        return "Question vide."

    snap = snapshot or get_portfolio_snapshot()
    # Keep prompt reasonably small; trim big lists defensively.
    compact = {
        "account": snap.get("account", {}),
        "spot_positions": (snap.get("spot_positions") or [])[:50],
        "option_positions": (snap.get("option_positions") or [])[:50],
        "open_orders": (snap.get("open_orders") or [])[:50],
    }
    prompt = (
        "You are a portfolio risk assistant for a paper trading app.\n"
        "Use the snapshot below (JSON) to answer the user question.\n"
        "Be concise, actionable, and highlight risks (concentration, leverage, orders, liquidity).\n\n"
        f"SNAPSHOT_JSON:\n{json.dumps(compact, ensure_ascii=False)}\n\n"
        f"USER_QUESTION:\n{q}\n"
    )
    return chatgpt_response(prompt)


__all__ = ["get_portfolio_snapshot", "ask_portfolio_copilot"]

