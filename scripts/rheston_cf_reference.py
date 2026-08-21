"""Implémentation de référence VALIDÉE de la CF rough Heston markovienne.

Accompagne `docs/review-2026-08-rough-heston-qrhplus.md`. Corrige les défauts
C1/M1/M2/M3 de `app/model/volatility_models/rheston/cf_markovian.py` :

  C1  Euler explicite instable (x·dt jusqu'à ~558)  ->  schéma totalement
      implicite : à chaque pas, S_{n+1} résout la quadratique de Riccati
      (racine stable via Citardauq). Inconditionnellement stable.
  M1  masse [0, x_min) du noyau perdue              ->  facteur constant w_inf.
  M2  n_steps ∝ T (2 pas à T=1 semaine)             ->  minimum 200 pas/maturité.
  M3  colonnes explosées (moment explosion)         ->  φ=0 + compteur `n_zeroed`
      exposé (au-delà de ~10 %, l'évaluation doit être rejetée par l'appelant).

Validation (voir review, annexe) :
  * vs Heston fermé (Albrecher) à H=0.499 : 0.7–1.7 bp sur les smiles ;
  * vs Adams fractionnaire (El Euch–Rosenbaum) à H=0.07 : ~5 bp à T=1/52,
    |Δφ| ≤ 6.4e-4 à T=0.05 ;
  * coût ≈ 0.12 s / maturité (2048 u, 21 facteurs, 200 pas, numpy pur).

Usage : `python scripts/rheston_cf_reference.py` exécute l'auto-validation
contre l'oracle Heston fermé (rapide, ~10 s). Schéma d'intégration destiné à
remplacer le corps de `rheston_log_return_cf_markovian` (mêmes conventions :
CF du log-rendement, drift (r-q) inclus, u complexe autorisé pour le damping
Carr-Madan).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma as _gamma

__all__ = ["RHestonCFConfig", "kernel_geom_with_const", "rheston_cf_implicit", "heston_cf_closed_form"]

_RE_CAP = 50.0  # une CF dampée légitime a Re(A)=O(10); au-delà: artefact -> 0


@dataclass(frozen=True)
class RHestonCFConfig:
    n_factors: int = 20          # facteurs exponentiels (hors facteur constant)
    steps_per_year: int = 120
    n_steps_min: int = 200       # M2: minimum par maturité (papier QRH+: N=200)
    x_min_mult: float = 0.1      # x_min = x_min_mult / T
    x_max: float = 120000.0      # borne haute fixe de la quadrature


def kernel_geom_with_const(H: float, n_factors: int, x_min: float, x_max: float):
    """K(t)=t^{H-1/2}/Γ(H+1/2) ≈ w_inf + Σ w_i e^{-x_i t} (quadrature géométrique
    de la représentation de Laplace + facteur constant pour la masse [0, x_min))."""
    if not (0.0 < H < 0.5):
        raise ValueError("H must be in (0, 0.5).")
    power = 0.5 - H
    c = 1.0 / (float(_gamma(H + 0.5)) * float(_gamma(0.5 - H)))
    edges = np.geomspace(float(x_min), float(x_max), int(n_factors) + 1)
    rates = np.sqrt(edges[:-1] * edges[1:])
    weights = c * (edges[1:] ** power - edges[:-1] ** power) / power
    w_inf = c * (float(x_min) ** power) / power          # M1
    rates = np.concatenate([[0.0], rates]).astype(float)
    weights = np.concatenate([[w_inf], weights]).astype(float)
    return rates, weights


def rheston_cf_implicit(
    u: np.ndarray,
    T: float,
    *,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    H: float,
    cfg: RHestonCFConfig | None = None,
):
    """CF du log-rendement du rough Heston, schéma totalement implicite.

    Retourne (phi, n_zeroed) : phi[u] = E[exp(iu·log(S_T/S_0))] évaluée en u
    (complexe autorisé), n_zeroed = nombre de colonnes u mises à 0 (non finies
    ou Re(A) > _RE_CAP). L'appelant DOIT rejeter l'évaluation si n_zeroed
    dépasse ~10 % de u.size (explosion de moment: réduire xi ou le damping).
    """
    cfg = cfg or RHestonCFConfig()
    u = np.asarray(u, dtype=complex)
    T = float(T)
    if T <= 0 or u.size == 0:
        return np.ones_like(u), 0

    rates, weights = kernel_geom_with_const(
        H, cfg.n_factors, cfg.x_min_mult / max(T, 1e-6), cfg.x_max
    )
    n_steps = int(max(cfg.n_steps_min, round(cfg.steps_per_year * T)))   # M2
    dt = T / n_steps

    iu = 1j * u
    half = 0.5 * (u * u + iu)            # -F(u,0)
    beta = iu * rho * xi - kappa
    gam = 0.5 * xi * xi

    rates_c = rates[:, None]
    weights_c = weights[:, None]
    denom = 1.0 + rates_c * dt
    W = float((weights / (1.0 + rates * dt)).sum())

    A = np.zeros_like(u)
    B = np.zeros((rates.size, u.size), dtype=complex)
    S_prev = np.zeros_like(u)
    G_prev = -half + beta * S_prev + gam * S_prev * S_prev

    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_steps):
            # S_{n+1} implicite:  dtWγ S² + (dtWβ−1) S + (P − dtW·half) = 0
            P = (B / denom).sum(axis=0)
            a2 = dt * W * gam
            b1 = dt * W * beta - 1.0
            c0 = P - dt * W * half
            disc = np.sqrt(b1 * b1 - 4.0 * a2 * c0)
            qq = -0.5 * (b1 + np.where(np.real(np.conj(b1) * disc) >= 0.0, disc, -disc))
            S = c0 / qq                                   # racine stable (S→P, dt→0)
            G = -half + beta * S + gam * S * S
            A = A + dt * (iu * (r - q) + kappa * theta * 0.5 * (S_prev + S) + v0 * 0.5 * (G_prev + G))
            B = (B + dt * weights_c * G) / denom
            S_prev, G_prev = S, G

    bad = ~np.isfinite(A) | (A.real >= _RE_CAP)           # M3
    phi = np.where(bad, 0.0 + 0.0j, np.exp(np.where(bad, 0.0, A)))
    return phi.astype(complex), int(bad.sum())


def heston_cf_closed_form(u, T, *, kappa, theta, xi, rho, v0, r, q):
    """Oracle: CF Heston classique (Albrecher et al., pas de 'little trap').
    Le rough Heston converge dessus quand H→0.5 (K→1)."""
    u = np.asarray(u, dtype=complex)
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b * b + xi * xi * (iu + u * u))
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = (kappa * theta / (xi * xi)) * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = ((b - d) / (xi * xi)) * (1.0 - e) / (1.0 - g * e)
    return np.exp(iu * (r - q) * T + C + D * v0)


if __name__ == "__main__":
    # Auto-validation rapide contre l'oracle Heston fermé (H→0.5).
    P = dict(kappa=2.0, theta=0.04, xi=0.6, rho=-0.5, v0=0.04)
    R, Q = 0.02, 0.0
    print("1) |phi| <= 1 sur u réels (propriété de CF), H=0.07:")
    for T in (1.0 / 52.0, 0.05, 0.25, 1.0):
        phi, nz = rheston_cf_implicit(np.linspace(0.0, 60.0, 121), T, r=R, q=Q, H=0.07, **P)
        print(f"   T={T:7.4f}  max|phi|={np.abs(phi).max():.6f}  zeroed={nz}/121")

    print("2) écart CF vs Heston fermé, u réels, H=0.499:")
    for T in (0.1, 0.25, 1.0):
        uu = np.linspace(0.0, 60.0, 121)
        phi, nz = rheston_cf_implicit(uu, T, r=R, q=Q, H=0.499, **P)
        ph = heston_cf_closed_form(uu, T, r=R, q=Q, **P)
        print(f"   T={T:5.2f}  max|phi_rh - phi_heston| = {np.abs(phi - ph).max():.2e}  zeroed={nz}")

    print("3) sensibilité à H (doit être nette — le code d'origine n'en avait aucune):")
    uu = np.linspace(0.0, 30.0, 61)
    p1, _ = rheston_cf_implicit(uu, 0.25, r=R, q=Q, H=0.07, **P)
    p2, _ = rheston_cf_implicit(uu, 0.25, r=R, q=Q, H=0.40, **P)
    print(f"   max|phi(H=0.07) - phi(H=0.40)| = {np.abs(p1 - p2).max():.4f}")
