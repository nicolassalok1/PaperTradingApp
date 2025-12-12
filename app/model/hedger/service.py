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
from app.model.hedger.hedger_models import OptionSpec
from app.model.options.core.iv import OPTIONS_BOOK_FILE
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


__all__ = [
    "OptionSpec",
    "load_options_portfolio",
    "option_specs_from_portfolio",
    "train_dqn_agent",
    "simulate_hedge",
]
