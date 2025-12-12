"""
CRR tree helpers extracted for reuse without UI dependencies.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def build_crr_tree(option=None, r: float = 0.0, sigma: float = 0.0, n_steps: int = 1):
    """
    Build a simple CRR tree of underlying prices and option values.

    Args:
        option: object with attributes s0, T, and payoff(s).
        r: risk-free rate.
        sigma: volatility.
        n_steps: depth of the tree.
    Returns:
        (spot_tree, value_tree) numpy arrays.
    """
    if option is None or n_steps <= 0:
        return np.zeros((1, 1)), np.zeros((1, 1))

    try:
        T = float(option.T)
        S0 = float(option.s0)
    except Exception:
        return np.zeros((1, 1)), np.zeros((1, 1))

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    q = 1 - p
    discount = np.exp(-r * dt)

    prices = np.zeros((n_steps + 1, n_steps + 1), dtype=float)
    values = np.zeros_like(prices)
    for i in range(n_steps + 1):
        for j in range(i + 1):
            prices[i, j] = S0 * (u**j) * (d ** (i - j))
    values[-1, : n_steps + 1] = option.payoff(prices[-1, : n_steps + 1])

    for i in range(n_steps - 1, -1, -1):
        continuation = discount * (p * values[i + 1, 1 : i + 2] + q * values[i + 1, : i + 1])
        exercise = option.payoff(prices[i, : i + 1])
        values[i, : i + 1] = np.maximum(exercise, continuation)

    return prices, values


def plot_crr_tree(spot_tree: np.ndarray, value_tree: np.ndarray):
    """
    Plot the CRR tree using matplotlib. Returns the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    max_depth = spot_tree.shape[0] - 1
    # Color nodes by option value for visual readability; no text overlays.
    vals = value_tree[: max_depth + 1, : max_depth + 1].flatten()
    vmin, vmax = float(vals.min()), float(vals.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-6)
    for i in range(max_depth + 1):
        for j in range(i + 1):
            ax.scatter(
                i,
                spot_tree[i, j],
                c=[[plt.cm.viridis(norm(value_tree[i, j]))]],
                s=32,
                edgecolors="k",
                linewidths=0.3,
            )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Valeur de l'option")
    ax.set_xlabel("Étapes")
    ax.set_ylabel("Spot")
    ax.set_title("Arbre CRR (spot & valeur)")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig
