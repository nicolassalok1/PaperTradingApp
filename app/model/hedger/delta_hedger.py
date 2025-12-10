import math
import numpy as np

from app.model.hedger.hedger_models import OptionSpec, build_state, compute_terminal_reward


class HedgingEnvSim:
    def __init__(self, option: OptionSpec, price_path: np.ndarray, hedge_lot: float = 1.0):
        self.option = option
        self.price_path = price_path.astype(np.float32)
        self.N = len(price_path)
        self.hedge_lot = hedge_lot
        self.t = 0
        self.cash = 0.0
        self.position = 0.0

    def reset(self):
        self.t = 0
        self.cash = 0.0
        self.position = 0.0
        return build_state(self.option, self.price_path, self.t, self.position)

    def step(self, action: int):
        S = float(self.price_path[self.t])
        if action == 0:
            q = -self.hedge_lot
        elif action == 1:
            q = 0.0
        elif action == 2:
            q = self.hedge_lot
        else:
            raise ValueError("bad action")

        if q != 0.0:
            self.position += q
            self.cash -= q * S

        self.t += 1
        done = self.t >= self.N - 1
        next_state = build_state(self.option, self.price_path, self.t, self.position)
        reward = (
            compute_terminal_reward(self.option, self.position, self.cash, self.price_path)
            if done
            else 0.0
        )
        if done:
            self.t = self.N - 1
        return next_state, reward, done, {}


def generate_price_path(option: OptionSpec, N: int = 120) -> np.ndarray:
    params_src = option.heston_params or {}
    if not params_src:
        S = np.zeros(N, dtype=np.float32)
        S[0] = option.S0
        mu, sig = 0.0, 0.2
        dt = option.maturity_years / max(1, N - 1)
        for t in range(1, N):
            Z = np.random.randn()
            S[t] = S[t - 1] * np.exp((mu - 0.5 * sig**2) * dt + sig * math.sqrt(dt) * Z)
        return S
    kappa = float(params_src.get("kappa", 1.5))
    theta = float(params_src.get("theta", 0.04))
    sigma_h = float(params_src.get("sigma", params_src.get("eta", 0.5)))
    rho = float(params_src.get("rho", -0.4))
    v = float(params_src.get("v0", 0.04))
    r = float(option.r)
    q = float(option.q)
    S = np.zeros(N, dtype=np.float32)
    S[0] = option.S0
    dt = option.maturity_years / max(1, N - 1)
    sqdt = math.sqrt(dt)
    for t in range(1, N):
        z1, z2 = np.random.randn(2)
        z2 = rho * z1 + math.sqrt(max(1e-12, 1.0 - rho**2)) * z2
        v = max(
            v + kappa * (theta - v) * dt + sigma_h * math.sqrt(max(v, 1e-12)) * sqdt * z2, 1e-12
        )
        S[t] = S[t - 1] * math.exp((r - q - 0.5 * v) * dt + math.sqrt(v) * sqdt * z1)
    return S
