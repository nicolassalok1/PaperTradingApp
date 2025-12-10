"""
Hedger controller.
Thin wrappers over model hedger services for the view layer.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from app.model.hedger.service import (
    OptionSpec,
    calibrate_heston_params,
    check_heston_support,
    load_options_portfolio,
    option_specs_from_portfolio,
    simulate_hedge,
    train_dqn_agent,
)


def load_option_specs() -> List[OptionSpec]:
    portfolio = load_options_portfolio()
    return option_specs_from_portfolio(portfolio)


def train_agent(option: OptionSpec, steps: int, episodes: int, hedge_lot: float) -> Dict:
    return train_dqn_agent(option, steps, episodes, hedge_lot)


def run_simulation(
    option: OptionSpec, steps: int, hedge_lot: float, agent_state: Dict
) -> List[Dict]:
    return simulate_hedge(option, steps, hedge_lot, agent_state)


__all__ = [
    "OptionSpec",
    "calibrate_heston_params",
    "check_heston_support",
    "load_option_specs",
    "train_agent",
    "run_simulation",
]
