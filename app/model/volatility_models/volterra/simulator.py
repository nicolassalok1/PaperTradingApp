from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from app.model.volatility_models.volterra.kernels import KernelFn


@dataclass(frozen=True)
class VolterraSimConfig:
    n_paths: int = 4000
    n_steps: int = 60
    seed: int | None = None


def simulate_volterra_sv_paths(
    *,
    S0: float,
    r: float,
    q: float,
    t_grid: np.ndarray,
    kernel: KernelFn,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    cfg: VolterraSimConfig | None = None,
) -> Dict[float, np.ndarray]:
    """
    Generic Volterra SV simulator (research-grade; O(n_steps^2)).

    Variance:
      V_t = v0 + ∫_0^t K(t-s) [kappa(θ - V_s) ds + xi sqrt(V_s) dB_s]

    Price:
      dS/S = (r-q) dt + sqrt(V_t) dW, corr(W,B)=rho

    Returns {T: S_T(paths)} for maturities in t_grid.
    """
    cfg = cfg or VolterraSimConfig()
    n_paths = int(cfg.n_paths)
    n_steps = int(cfg.n_steps)
    if n_paths <= 0 or n_steps <= 1:
        return {}

    S0 = float(S0)
    r = float(r)
    q = float(q)
    v0 = float(max(v0, 1e-12))
    kappa = float(max(kappa, 0.0))
    theta = float(max(theta, 1e-12))
    xi = float(max(xi, 1e-12))
    rho = float(np.clip(rho, -0.999, 0.999))

    t_grid = np.asarray(t_grid, dtype=float)
    T_max = float(np.max(t_grid)) if t_grid.size else 0.0
    if T_max <= 0:
        return {}

    dt = T_max / n_steps
    t = dt * np.arange(n_steps + 1, dtype=float)

    # map requested maturities to nearest indices
    idx_map: Dict[float, int] = {}
    for T in t_grid:
        if float(T) <= 0:
            continue
        idx = int(round(float(T) / dt))
        idx = min(max(idx, 1), n_steps)
        idx_map[float(T)] = idx

    rng = np.random.default_rng(cfg.seed) if cfg.seed is not None else np.random.default_rng()
    z1 = rng.standard_normal(size=(n_steps, n_paths))
    z2 = rng.standard_normal(size=(n_steps, n_paths))
    dB = np.sqrt(dt) * z1
    dW = rho * dB + np.sqrt(max(1e-12, 1.0 - rho * rho)) * (np.sqrt(dt) * z2)

    # Precompute kernel values K(t_i - t_j) as a lower-triangular matrix of weights in time differences
    # For i>j: delta = (i-j)*dt
    deltas = dt * np.arange(n_steps + 1, dtype=float)
    K_vals = kernel(deltas)
    K_vals = np.asarray(K_vals, dtype=float).reshape(-1)
    if K_vals.size != deltas.size:
        raise ValueError("Kernel must return same shape as input.")

    V = np.full((n_steps + 1, n_paths), v0, dtype=float)
    Z = np.zeros((n_steps, n_paths), dtype=float)  # dZ increments (drift+diff) at each step

    for i in range(1, n_steps + 1):
        # compute previous dZ_{0..i-1} with current V path
        j = i - 1
        Vj = np.maximum(V[j, :], 0.0)
        Z[j, :] = kappa * (theta - Vj) * dt + xi * np.sqrt(Vj) * dB[j, :]

        # Volterra convolution: V_i = v0 + Σ_{m=0}^{i-1} K((i-m)*dt) * Z_m
        weights = K_vals[1 : i + 1][::-1]  # K(dt),...,K(i*dt)
        # dot across time dimension (weights length i, Z[0:i,:] shape (i,n_paths))
        V[i, :] = v0 + np.dot(weights, Z[:i, :])
        V[i, :] = np.maximum(V[i, :], 0.0)

    logS = np.full((n_steps + 1, n_paths), np.log(S0), dtype=float)
    drift = (r - q) * dt
    for i in range(n_steps):
        vi = np.maximum(V[i, :], 0.0)
        logS[i + 1, :] = logS[i, :] + drift - 0.5 * vi * dt + np.sqrt(vi) * dW[i, :]

    out: Dict[float, np.ndarray] = {}
    for T, idx in idx_map.items():
        out[float(T)] = np.exp(logS[idx, :]).astype(float)
    return out


__all__ = ["VolterraSimConfig", "simulate_volterra_sv_paths"]

