"""
Utility to display a static payoff chart for any option record stored in the JSON
book (open or expired). It reuses the payoff logic from the Streamlit app and
supports multi-leg structures, Asian (with provided closing prices), digital,
barrier (simple path check), spreads, and vanilla fallbacks.

Usage:
    python scripts/payoff_viewer.py path/to/option.json [--output payoff.png]

The input JSON can be:
    - a single option dict (as stored in options_portfolio.json / expired_options.json)
    - or a file containing that dict.

The script builds a spot grid around the provided reference spot (underlying_close,
S0, or strike) and plots payoff + P&L (vs. avg_price if present).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import datetime
import requests
import math
import yfinance as yf


def _compute_leg_payoff(leg: dict[str, Any], spot: float) -> float:
    """Payoff for a single leg (vanilla call/put, long/short)."""
    option_type = str(leg.get("option_type") or leg.get("type") or "call").lower()
    strike = float(leg.get("strike", 0.0) or 0.0)
    qty = float(leg.get("qty", leg.get("quantity", 1.0)) or 1.0)
    payoff = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if str(leg.get("side", "long")).lower() == "short":
        payoff *= -1.0
    return payoff * qty


def compute_option_payoff(option: dict[str, Any], spot: float) -> float:
    """
    Evaluate payoff for an option/structure at a given spot.
    Supports multi-leg, Asian (via closing_prices), digital, barrier, spreads, and vanilla.
    """
    legs = option.get("legs") or []
    if legs:
        return sum(_compute_leg_payoff(leg, spot) for leg in legs)

    product = (
        option.get("product_type")
        or option.get("product")
        or option.get("structure")
        or option.get("type")
        or ""
    ).lower()
    option_type_raw = (
        option.get("option_type")
        or option.get("cpflag")
        or option.get("cp_flag")
        or option.get("cp")
        or option.get("cpflag")
        or ""
    )
    option_type = str(option_type_raw or "call").lower()
    strike = float(option.get("strike", 0.0) or 0.0)
    strike2 = float(option.get("strike2", option.get("strike_upper", 0.0) or 0.0) or 0.0)
    misc = option.get("misc") if isinstance(option.get("misc"), dict) else {}
    closing_prices = misc.get("closing_prices") if isinstance(misc, dict) else None

    # Asian via provided path
    if "asian" in product and closing_prices:
        vals = [v for v in closing_prices if v and v > 0]
        if not vals:
            avg_level = spot
        elif "geom" in product:
            avg_level = float(np.exp(np.mean(np.log(vals))))
        else:
            avg_level = float(np.mean(vals))
        if option_type == "put":
            return max(strike - avg_level, 0.0)
        return max(avg_level - strike, 0.0)

    # Digital
    if "digital" in product:
        payout = float(misc.get("payout", 1.0) or 1.0)
        if option_type == "put":
            return payout if spot < strike else 0.0
        return payout if spot > strike else 0.0

    # Barrier (simple in/out check using closing_prices if provided)
    if "barrier" in product:
        barrier = float(misc.get("barrier", misc.get("barrier_level", 0.0)) or 0.0)
        barrier_type = str(misc.get("barrier_type", "up")).lower()
        knock = str(misc.get("knock", misc.get("direction", "out"))).lower()
        path_vals: Iterable[float] = closing_prices or [spot]
        hit = any((p >= barrier if barrier_type == "up" else p <= barrier) for p in path_vals)
        if knock == "out" and hit:
            return 0.0
        if knock == "in" and not hit:
            return 0.0
        # otherwise vanilla payoff falls through

    # Spreads / combos
    if product in {"straddle"}:
        return max(spot - strike, 0.0) + max(strike - spot, 0.0)
    if product in {"strangle"}:
        k_put = min(strike, strike2 or strike)
        k_call = max(strike, strike2 or strike)
        return max(k_put - spot, 0.0) + max(spot - k_call, 0.0)
    if product in {"call_spread", "bull_call_spread", "callspread"}:
        k1 = strike
        k2 = strike2 or strike
        return max(spot - k1, 0.0) - max(spot - k2, 0.0)
    if product in {"put_spread", "bear_put_spread", "putspread"}:
        k_long = strike
        k_short = strike2 or strike
        return max(k_long - spot, 0.0) - max(k_short - spot, 0.0)

    # Vanilla fallback
    if option_type == "put" or product == "put":
        return max(strike - spot, 0.0)
    return max(spot - strike, 0.0)


def build_grid_and_payoff(option: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a spot grid and corresponding payoff/P&L arrays.
    Uses underlying_close, S0, or strike as reference center.
    """
    spot_ref = float(
        option.get("underlying_close")
        or option.get("S0")
        or option.get("S_T")
        or option.get("spot")
        or option.get("spot_at_pricing", 0.0)
        or option.get("strike", 0.0)
        or 1.0
    )
    strike = float(option.get("strike", 0.0) or 0.0)
    if spot_ref <= 0:
        spot_ref = strike if strike > 0 else 1.0
    s_grid = np.linspace(max(0.01, 0.5 * spot_ref), 1.5 * spot_ref, 200)
    pay_grid = np.array([compute_option_payoff(option, s) for s in s_grid], dtype=float)
    side = str(option.get("side", "long")).lower()
    if side == "short":
        pay_grid *= -1.0
    avg_price = float(option.get("avg_price", option.get("T_0_price", 0.0)) or 0.0)
    pnl_grid = pay_grid - avg_price if side == "long" else avg_price + pay_grid
    return s_grid, pay_grid, pnl_grid


def plot_payoff(option: dict[str, Any], output: Path | None = None) -> None:
    """Plot payoff & P&L curves; show or save depending on output."""
    s_grid, pay_grid, pnl_grid = build_grid_and_payoff(option)
    strike = float(option.get("strike", 0.0) or 0.0)
    spot_ref = s_grid[len(s_grid) // 2]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s_grid, pay_grid, label="Payoff (signé)")
    ax.plot(s_grid, pnl_grid, label="P&L vs. prime", color="darkorange")
    if strike > 0:
        ax.axvline(strike, color="gray", linestyle="--", label=f"K = {strike:.2f}")
    ax.axvline(spot_ref, color="crimson", linestyle="-.", label=f"Ref = {spot_ref:.2f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Spot")
    ax.set_ylabel("Payoff / P&L")
    title = option.get("product_type") or option.get("product") or option.get("type") or "Option"
    ax.set_title(f"{title} payoff")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, linestyle="--")

    if output:
        fig.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def _load_option(json_arg: str) -> dict[str, Any]:
    """Load option dict from file or JSON string."""
    path = Path(json_arg)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return json.loads(json_arg)


def main():
    parser = argparse.ArgumentParser(description="Display payoff graph for a stored option JSON.")
    parser.add_argument("option_json", help="Path to option JSON file or raw JSON string.")
    parser.add_argument("--output", "-o", type=Path, help="Optional path to save PNG instead of showing.")
    args = parser.parse_args()

    option = _load_option(args.option_json)
    plot_payoff(option, output=args.output)


if __name__ == "__main__":
    main()
