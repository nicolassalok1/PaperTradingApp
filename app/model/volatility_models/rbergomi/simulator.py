from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RBergomiSimConfig:
    n_paths: int = 5000
    n_steps: int = 60
    seed: int | None = None


def _select_time_indices(t_grid: np.ndarray, T_max: float, n_steps: int) -> dict[float, int]:
    t_grid = np.asarray(t_grid, dtype=float)
    dt = float(T_max) / float(n_steps)
    out: dict[float, int] = {}
    for T in t_grid:
        if float(T) <= 0:
            continue
        idx = int(round(float(T) / dt))
        idx = min(max(idx, 1), n_steps)
        out[float(T)] = idx
    return out


def simulate_rbergomi_paths(
    *,
    S0: float,
    r: float,
    q: float,
    t_grid: np.ndarray,
    H: float,
    eta: float,
    rho: float,
    xi0: float,
    cfg: RBergomiSimConfig | None = None,
) -> dict[float, np.ndarray]:
    """
    Rough Bergomi (Bayer-Friz-Gatheral) - MC only.

    Uses a simple Riemann sum for the Riemann-Liouville fBM:
      W^H_t = sqrt(2H) * ∫_0^t (t-s)^{H-1/2} dB_s
    and variance:
      V_t = xi0 * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})

    Returns {T: S_T(paths)} for requested maturities in t_grid.
    """
    cfg = cfg or RBergomiSimConfig()
    n_paths = int(cfg.n_paths)
    n_steps = int(cfg.n_steps)
    if n_paths <= 0 or n_steps <= 1:
        return {}

    S0 = float(S0)
    r = float(r)
    q = float(q)
    H = float(H)
    if not (0.0 < H < 0.5):
        raise ValueError("H must be in (0, 0.5).")
    eta = float(max(eta, 1e-12))
    rho = float(np.clip(rho, -0.999, 0.999))
    xi0 = float(max(xi0, 1e-12))

    t_grid = np.asarray(t_grid, dtype=float)
    T_max = float(np.max(t_grid)) if t_grid.size else 0.0
    if T_max <= 0:
        return {}

    dt = T_max / n_steps
    t = dt * np.arange(n_steps + 1, dtype=float)
    idx_map = _select_time_indices(t_grid, T_max, n_steps)
    want_idx = sorted(set(idx_map.values()))

    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    z1 = rng.standard_normal(size=(n_steps, n_paths))
    z2 = rng.standard_normal(size=(n_steps, n_paths))

    dB = np.sqrt(dt) * z1
    dW = rho * dB + np.sqrt(max(1e-12, 1.0 - rho * rho)) * (np.sqrt(dt) * z2)

    # Build RL-fBM W^H on the grid (O(n_steps^2) kernel convolution)
    wh = np.zeros((n_steps + 1, n_paths), dtype=float)
    sqrt_2H = float(np.sqrt(2.0 * H))
    for i in range(1, n_steps + 1):
        ti = t[i]
        tj = t[:i]  # t_0,...,t_{i-1}
        kern = (ti - tj) ** (H - 0.5)  # shape (i,)
        # Riemann sum for the stochastic integral: Σ f(t_j) ΔB_j, j=0..i-1
        wh[i, :] = sqrt_2H * np.sum(kern[:, None] * dB[:i, :], axis=0)

    # Variance path
    v = xi0 * np.exp(eta * wh - 0.5 * (eta * eta) * (t[:, None] ** (2.0 * H)))
    v = np.maximum(v, 0.0)

    # Simulate log-price
    logS = np.full((n_steps + 1, n_paths), np.log(S0), dtype=float)
    drift = (r - q) * dt
    for i in range(n_steps):
        vi = v[i, :]
        logS[i + 1, :] = logS[i, :] + drift - 0.5 * vi * dt + np.sqrt(np.maximum(vi, 0.0)) * dW[i, :]

    out: dict[float, np.ndarray] = {}
    for T, idx in idx_map.items():
        if idx in want_idx:
            out[float(T)] = np.exp(logS[idx, :]).astype(float)
    return out


__all__ = ["RBergomiSimConfig", "simulate_rbergomi_paths"]
