"""
Simple DQN-style hedge suggester (placeholder, untrained).
"""

from __future__ import annotations

import numpy as np

from .alpaca_client import AlpacaHedgerClient
from .env import HedgingEnv


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, seed: int | None = None) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)
        # simple linear approximator weights
        self.weights = self.rng.normal(0, 0.1, size=(state_dim, n_actions))

    def select_action(self, state: dict, epsilon: float = 0.1) -> int:
        state_vec = np.array(list(state.values()), dtype=float)
        if state_vec.shape[0] != self.state_dim:
            state_vec = np.resize(state_vec, self.state_dim)
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q_values = state_vec @ self.weights
        return int(np.argmax(q_values))

    def load_dummy_weights_or_init(self) -> None:
        # Placeholder: weights already initialized randomly.
        return


def suggest_hedge_action(client: AlpacaHedgerClient, underlying_symbol: str) -> dict:
    env = HedgingEnv(client=client, underlying_symbol=underlying_symbol, position_scale=1.0)
    state = env.reset()
    agent = DQNAgent(state_dim=len(state), n_actions=4)
    agent.load_dummy_weights_or_init()
    action = agent.select_action(state, epsilon=0.05)

    action_map = {
        0: {"side": "none", "delta_qty": 0.0},
        1: {"side": "buy", "delta_qty": +1.0},
        2: {"side": "sell", "delta_qty": -1.0},
        3: {"side": "flatten", "delta_qty": -state.get("equity_position", 0.0)},
    }
    mapped = action_map.get(action, {"side": "none", "delta_qty": 0.0})
    return {
        "underlying": (underlying_symbol or "").strip().upper(),
        "action": action,
        "side": mapped["side"],
        "delta_qty": float(mapped["delta_qty"]),
        "comment": "DQN hedge suggestion (basic)",
        "state": state,
    }


__all__ = ["DQNAgent", "suggest_hedge_action"]
