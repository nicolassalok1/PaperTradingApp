import math

import numpy as np


def price_american_crr(
    S0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    steps: int,
    option_type: str = "call",
) -> float:
    """
    American option price via a CRR binomial tree (exercise allowed at each node).
    """
    call_flag = str(option_type).lower().startswith("c")
    dt = float(T) / float(steps)
    r_eff = float(r) - float(q)
    u = math.exp(float(sigma) * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp(r_eff * dt)
    p = (a - d) / (u - d)
    q_prob = 1.0 - p
    discount = math.exp(-r_eff * dt)

    asset = np.array([float(S0) * (u**j) * (d ** (steps - j)) for j in range(steps + 1)])
    payoff = np.maximum(asset - float(K), 0.0) if call_flag else np.maximum(float(K) - asset, 0.0)
    values = payoff

    for step in range(int(steps) - 1, -1, -1):
        cont = discount * (p * values[1:] + q_prob * values[:-1])
        asset = np.array([float(S0) * (u**j) * (d ** (step - j)) for j in range(step + 1)])
        exercise = (
            np.maximum(asset - float(K), 0.0) if call_flag else np.maximum(float(K) - asset, 0.0)
        )
        values = np.maximum(exercise, cont)

    return float(values[0])


def price_bermuda_crr(
    S0: float,
    K: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    steps: int,
    exercise_dates: list[int] | tuple[int, ...],
    option_type: str = "call",
) -> float:
    """
    Bermudan option via binomial tree with discrete exercise dates.
    exercise_dates are step indices (1..steps); maturity is always exercisable.
    """
    exercise_steps = {int(s) for s in exercise_dates if 0 <= int(s) <= int(steps)}
    exercise_steps.add(int(steps))
    call_flag = str(option_type).lower().startswith("c")
    dt = float(T) / float(steps)
    r_eff = float(r) - float(q)
    u = math.exp(float(sigma) * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp(r_eff * dt)
    p = (a - d) / (u - d)
    q_prob = 1.0 - p
    discount = math.exp(-r_eff * dt)

    asset = np.array([float(S0) * (u**j) * (d ** (steps - j)) for j in range(steps + 1)])
    payoff = np.maximum(asset - float(K), 0.0) if call_flag else np.maximum(float(K) - asset, 0.0)

    values = payoff
    for step in range(int(steps) - 1, -1, -1):
        cont = discount * (p * values[1:] + q_prob * values[:-1])
        asset = np.array([float(S0) * (u**j) * (d ** (step - j)) for j in range(step + 1)])
        if step in exercise_steps:
            exercise = (
                np.maximum(asset - float(K), 0.0)
                if call_flag
                else np.maximum(float(K) - asset, 0.0)
            )
            values = np.maximum(exercise, cont)
        else:
            values = cont

    return float(values[0])
