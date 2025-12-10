"""
RL inference helpers for Hedger RL Live v2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class LiveRLAgentV2:
    def __init__(self, state_dim: int = 8, n_actions: int = 4) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.weights = np.zeros((state_dim, n_actions))

    def load_model(self, path: str | Path) -> None:
        try:
            arr = np.load(path)
            if arr.shape == self.weights.shape:
                self.weights = arr
        except Exception:
            # If loading fails, keep zeros
            pass

    def select_action(self, state_vector: List[float]) -> Dict[str, Any]:
        s = np.array(state_vector, dtype=float)
        if s.shape[0] != self.state_dim:
            s = np.resize(s, self.state_dim)
        q_values = s @ self.weights
        action_id = int(np.argmax(q_values))
        action_map = {
            0: {"side": "none", "delta_qty": 0.0},
            1: {"side": "buy", "delta_qty": +1.0},
            2: {"side": "sell", "delta_qty": -1.0},
            3: {"side": "flatten", "delta_qty": 0.0},
        }
        mapped = action_map.get(action_id, {"side": "none", "delta_qty": 0.0})
        return {
            "action_id": action_id,
            "side": mapped["side"],
            "delta_qty": float(mapped["delta_qty"]),
            "q_values": q_values.tolist(),
        }


def load_latest_agent_v2(state_dim: int = 8, n_actions: int = 4) -> LiveRLAgentV2:
    agent = LiveRLAgentV2(state_dim=state_dim, n_actions=n_actions)
    models_dir = Path("models/rl_hedger")
    candidates = sorted(models_dir.glob("*.npy")) if models_dir.exists() else []
    if candidates:
        agent.load_model(candidates[-1])
    return agent


__all__ = ["LiveRLAgentV2", "load_latest_agent_v2"]
