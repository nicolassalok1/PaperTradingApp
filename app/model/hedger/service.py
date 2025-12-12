"""
Hedger business services (no UI).
Provides option loading, DQN training, and hedge simulation helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from app.model.hedger.delta_hedger import HedgingEnvSim, generate_price_path
from app.model.hedger.dqn_agent import DQNAgent
from app.model.hedger.hedger_models import OptionSpec, build_state
from app.model.options.core.iv import OPTIONS_BOOK_FILE
from app.model.portfolio.positions import load_portfolio_default
from app.model.trading.hedging import HedgingOrder
from app.model.trading.service import get_market_price, floor_4
from app.utils.io import load_json_file


def load_options_portfolio(path: Path | None = None) -> Dict:
    json_path = path or OPTIONS_BOOK_FILE
    return load_json_file(json_path, {})


def option_specs_from_portfolio(portfolio: Dict) -> List[OptionSpec]:
    return [OptionSpec.from_json(k, v) for k, v in (portfolio or {}).items()]


def train_dqn_agent(option: OptionSpec, steps: int, episodes: int, hedge_lot: float) -> Dict:
    env_maker = lambda: HedgingEnvSim(
        option, generate_price_path(option, int(steps)), hedge_lot=hedge_lot
    )
    agent = DQNAgent(state_dim=5, action_dim=3)
    eps = 1.0
    eps_end = 0.05
    eps_decay = 0.98
    rewards: List[float] = []

    for _ in range(episodes):
        env = env_maker()
        state = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.act(state, eps)
            ns, r, done, _ = env.step(action)
            agent.store(state, action, r, ns, done)
            agent.train_step()
            state = ns
            ep_reward += float(r)
        eps = max(eps_end, eps * eps_decay)
        rewards.append(ep_reward)

    return {"agent_state": agent.q_net.state_dict(), "rewards": rewards}


def simulate_hedge(
    option: OptionSpec, steps: int, hedge_lot: float, agent_state: Dict
) -> List[Dict]:
    env = HedgingEnvSim(option, generate_price_path(option, int(steps)), hedge_lot=hedge_lot)
    agent = DQNAgent(state_dim=5, action_dim=3)
    agent.q_net.load_state_dict(agent_state)
    agent.target_net.load_state_dict(agent_state)

    action_labels = {0: "Vendre hedge lot", 1: "Attente", 2: "Acheter hedge lot"}
    logs: List[Dict] = []
    state = env.reset()
    for t in range(env.N):
        action = agent.act(state, eps=0.0)
        ns, _, done, _ = env.step(action)
        logs.append(
            {
                "step": t,
                "spot": float(env.price_path[env.t]),
                "action": action_labels.get(action, str(action)),
                "position": env.position,
                "cash": env.cash,
            }
        )
        state = ns
        if done:
            break
    return logs


def _current_underlying_position(symbol: str) -> float:
    """
    Return current underlying position (long positive, short negative)
    for the given symbol based on the default portfolio.
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


def _build_live_state(option: OptionSpec, hedge_lot: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build the DQN state vector using live market data and current position.
    """
    spot = get_market_price(option.symbol, fallback=option.S0)
    spot = float(spot or option.S0 or 0.0)
    position = _current_underlying_position(option.symbol)

    price_path = np.array([spot], dtype=np.float32)
    state_vec = build_state(option, price_path, t=0, position=position)
    meta = {
        "spot": spot,
        "position": position,
        "hedge_lot": float(hedge_lot),
    }
    return state_vec, meta


def _build_agent(agent_state: Dict[str, np.ndarray] | None = None) -> DQNAgent:
    """
    Instantiate a DQNAgent and optionally load a trained state dict.
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
    agent_state: Dict[str, np.ndarray] | None = None,
) -> List[HedgingOrder]:
    """
    Use the DQN hedger as a black box to compute one-step hedging orders.
    """
    hedge_lot_val = float(hedge_lot or 0.0)
    if hedge_lot_val <= 0:
        raise ValueError("hedge_lot must be strictly positive")

    state_vec, meta = _build_live_state(option, hedge_lot_val)
    agent = _build_agent(agent_state)

    action_idx = int(agent.act(state_vec, eps=0.0))

    if action_idx == 1:
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


__all__ = [
    "OptionSpec",
    "load_options_portfolio",
    "option_specs_from_portfolio",
    "train_dqn_agent",
    "simulate_hedge",
    "compute_hedging_orders",
]
