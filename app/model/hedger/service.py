"""
Hedger business services (no UI).
Provides option loading, DQN training, and hedge simulation helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
except Exception as exc:  # noqa: BLE001
    torch = None
    _TORCH_ERROR = exc
else:
    _TORCH_ERROR = None

try:
    from app.model.heston.pricing_scripts.heston_torch import HestonParams, carr_madan_call_torch
except Exception as exc:  # noqa: BLE001
    HestonParams = None  # type: ignore
    carr_madan_call_torch = None  # type: ignore
    _HESTON_ERROR = exc
else:
    _HESTON_ERROR = None

from app.model.hedger.delta_hedger import HedgingEnvSim, generate_price_path
from app.model.hedger.dqn_agent import DQNAgent
from app.model.hedger.hedger_models import OptionSpec
from app.model.options.core.iv import OPTIONS_BOOK_FILE
from app.utils.io import load_json_file


def check_heston_support() -> Tuple[bool, str | None]:
    if torch is None:
        return False, str(_TORCH_ERROR)
    if HestonParams is None or carr_madan_call_torch is None:
        return False, str(_HESTON_ERROR)
    return True, None


def load_options_portfolio(path: Path | None = None) -> Dict:
    json_path = path or OPTIONS_BOOK_FILE
    return load_json_file(json_path, {})


def option_specs_from_portfolio(portfolio: Dict) -> List[OptionSpec]:
    return [OptionSpec.from_json(k, v) for k, v in (portfolio or {}).items()]


def calibrate_heston_params(
    df,
    r: float,
    q: float,
    *,
    max_iters: int = 800,
    lr: float = 1e-2,
    hp_seed: Dict | None = None,
) -> HestonParams:
    """
    Torch-based calibration loop (no UI). Expects df with columns S0, K, T, C_mkt.
    """
    ok, err = check_heston_support()
    if not ok or torch is None or HestonParams is None or carr_madan_call_torch is None:
        raise RuntimeError(f"Heston non disponible: {err}")

    hp_seed = hp_seed or {}
    df_clean = df.dropna(subset=["S0", "K", "T", "C_mkt"])
    df_clean = df_clean[(df_clean["T"] > 0.4) & (df_clean["C_mkt"] > 0.05)]
    if df_clean.empty:
        raise ValueError("Pas de points pour la calibration")

    S0_ref = float(df_clean["S0"].median())
    moneyness = df_clean["K"].values / S0_ref

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S0_t = torch.tensor(df_clean["S0"].values, dtype=torch.float64, device=device)
    K_t = torch.tensor(df_clean["K"].values, dtype=torch.float64, device=device)
    T_t = torch.tensor(df_clean["T"].values, dtype=torch.float64, device=device)
    C_mkt_t = torch.tensor(df_clean["C_mkt"].values, dtype=torch.float64, device=device)

    weights_np = 1.0 / (np.abs(moneyness - 1.0) + 1e-3)
    weights_np = np.clip(weights_np / weights_np.mean(), 0.5, 5.0)
    weights_t = torch.tensor(weights_np, dtype=torch.float64, device=device)

    u_init = torch.tensor(
        [
            float(hp_seed.get("kappa", 1.5)),
            float(hp_seed.get("theta", 0.04)),
            float(hp_seed.get("sigma", hp_seed.get("eta", 0.5))),
            float(hp_seed.get("rho", -0.4)),
            float(hp_seed.get("v0", 0.04)),
        ],
        dtype=torch.float64,
        device=device,
    )

    u = u_init.clone().detach().requires_grad_(True)
    m = torch.zeros_like(u)
    v = torch.zeros_like(u)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8

    def prices_from_u(u_vec: torch.Tensor):
        params = HestonParams.from_unconstrained(u_vec[0], u_vec[1], u_vec[2], u_vec[3], u_vec[4])
        prices = []
        for S0_i, K_i, T_i in zip(S0_t, K_t, T_t):
            prices.append(
                carr_madan_call_torch(
                    S0_i.double(), float(r), float(q), T_i.double(), params, K_i.double()
                ).double()
            )
        return torch.stack(prices)

    for iteration in range(max_iters):
        if u.grad is not None:
            u.grad.zero_()
        model_prices = prices_from_u(u)
        diff = model_prices - C_mkt_t
        loss_val = 0.5 * (weights_t * diff**2).mean()
        loss_val.backward()
        with torch.no_grad():
            grad = u.grad
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            m_hat = m / (1 - beta1 ** (iteration + 1))
            v_hat = v / (1 - beta2 ** (iteration + 1))
            u -= lr * m_hat / (torch.sqrt(v_hat) + eps)

    params_f = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
    return params_f


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
    "check_heston_support",
    "load_options_portfolio",
    "option_specs_from_portfolio",
    "calibrate_heston_params",
    "train_dqn_agent",
    "simulate_hedge",
]
