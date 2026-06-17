from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.model.volatility_models.rheston.markovian_kernel import fractional_kernel_markovian_approx

_RE_CAP = 700.0  # exp(700) is near float64 max; beyond this np.exp overflows


@dataclass(frozen=True)
class RHestonMarkovianConfig:
    n_factors: int = 12
    steps_per_year: int = 120
    x_min_mult: float = 0.1  # x_min = x_min_mult / T_max
    x_max_mult: float = 1000.0  # x_max = x_max_mult * steps_per_year


def rheston_log_return_cf_markovian(
    u: np.ndarray,
    maturities: list[float],
    *,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    H: float,
    cfg: RHestonMarkovianConfig | None = None,
) -> dict[float, np.ndarray]:
    """
    Markovian approximation CF for rough Heston on multiple maturities.

    Returns {T: phi(u, T)} where u can be complex-valued array (vectorized).
    """
    u = np.asarray(u, dtype=complex)
    if u.size == 0:
        return {float(T): np.array([], dtype=complex) for T in maturities}

    maturities_sorted = sorted({float(t) for t in maturities if float(t) > 0})
    if not maturities_sorted:
        return {}

    cfg = cfg or RHestonMarkovianConfig()
    T_max = max(maturities_sorted)
    T_eff = max(1e-6, float(T_max))

    x_min = float(cfg.x_min_mult) / T_eff
    x_max = float(cfg.x_max_mult) * float(cfg.steps_per_year)
    kern = fractional_kernel_markovian_approx(H=H, n_factors=int(cfg.n_factors), x_min=x_min, x_max=x_max)
    rates = np.asarray(kern.rates, dtype=float)
    weights = np.asarray(kern.weights, dtype=float)

    kappa = float(max(kappa, 1e-12))
    theta = float(max(theta, 1e-12))
    xi = float(max(xi, 1e-12))
    rho = float(np.clip(rho, -0.999, 0.999))
    v0 = float(max(v0, 1e-12))
    r = float(r)
    q = float(q)

    iu = 1j * u
    half_u2_iu = 0.5 * (u * u + iu)

    # State arrays for each u (vectorized)
    A = np.zeros_like(u, dtype=complex)
    B = np.zeros((len(rates), u.size), dtype=complex)

    out: dict[float, np.ndarray] = {}
    t_prev = 0.0
    # Persistent mask: once a u-column diverges its true large-|u| CF tail is 0.
    # Keep it frozen for all subsequent steps and maturities.
    frozen = np.zeros(u.size, dtype=bool)

    for T in maturities_sorted:
        dt_total = float(T) - float(t_prev)
        if dt_total <= 0:
            out[float(T)] = np.exp(A).astype(complex)
            continue

        n_steps = max(1, int(round(float(cfg.steps_per_year) * dt_total)))
        dt = dt_total / n_steps

        for _ in range(n_steps):
            # Skip frozen columns entirely to avoid re-introducing NaN/inf.
            active = ~frozen
            if not active.any():
                break

            S = np.sum(B, axis=0)  # shape (n_u,)
            # Only compute G for active columns; frozen columns keep S=0.
            S_active = np.where(active, S, 0.0)
            G = (-half_u2_iu) + (-kappa * S_active) + (iu * rho * xi * S_active) + (0.5 * xi * xi) * (S_active * S_active)

            B = B + dt * ((-rates[:, None]) * B + (weights[:, None]) * G)
            A = A + dt * (iu * (r - q) + kappa * theta * S_active + v0 * G)

            # Freeze any u-column whose Riccati state diverged. The true CF tail
            # for large |u| decays to 0, so a blown-up column is set to exp(A)->0
            # instead of letting inf/NaN poison the whole FFT price row.
            bad = ~(np.isfinite(B).all(axis=0) & np.isfinite(A))
            new_bad = bad & ~frozen
            if new_bad.any():
                frozen |= new_bad
                B[:, frozen] = 0.0
                A[frozen] = complex(-_RE_CAP, 0.0)

        # Cap Re(A) to avoid overflow on borderline-huge finite values, then
        # zero out any residual non-finite entry.
        a_re = np.clip(A.real, -np.inf, _RE_CAP)
        phi = np.exp(a_re + 1j * A.imag)
        phi = np.where(np.isfinite(phi), phi, 0.0 + 0.0j)
        out[float(T)] = phi.reshape(u.shape).astype(complex)
        t_prev = float(T)

    return out


__all__ = ["RHestonMarkovianConfig", "rheston_log_return_cf_markovian"]

