from __future__ import annotations

from typing import Mapping, MutableMapping


def get_option_context_from_state(state_dict: Mapping | MutableMapping | None = None):
    """
    Build a lightweight options context from an explicit state mapping.
    No UI dependency: the caller must pass the state values.
    """
    state = state_dict or {}
    S0 = float(state.get("common_spot_value", 100.0))
    ticker = (
        state.get("common_underlying")
        or state.get("tkr_common")
        or state.get("heston_cboe_ticker")
        or ""
    )

    ctx = {
        "S0": S0,
        "ticker": str(ticker).upper(),
        "close_series": S0,
    }

    key_builder = state.get("_k")
    if key_builder is not None:
        ctx["_k"] = key_builder

    return ctx


def get_option_context(state_dict: Mapping | MutableMapping | None = None):
    """
    Backward-compatible wrapper to keep the old name while expecting an explicit state.
    """
    return get_option_context_from_state(state_dict)


__all__ = ["get_option_context_from_state", "get_option_context"]
