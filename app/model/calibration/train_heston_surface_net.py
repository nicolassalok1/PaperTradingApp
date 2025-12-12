"""
Offline training script for HestonSurfaceNet.
Generates synthetic IV surfaces, trains the CNN, and saves weights to:
    app/model/calibration/weights/heston_surface_net.pt
Run manually; not used in the UI.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app.model.calibration.heston_nn import HestonSurfaceNet, WEIGHTS_PATH, IV_MEAN, IV_STD, EPS
from app.model.calibration.heston_pricer import price_grid_from_params
from app.model.calibration.implied_vol import implied_vol_grid
from app.model.calibration.market_surface import default_grid


def _sample_params(n: int):
    params = []
    for _ in range(n):
        kappa = random.uniform(0.5, 5.0)
        theta = random.uniform(0.02, 0.3)
        sigma = random.uniform(0.1, 1.0)
        rho = random.uniform(-0.8, 0.0)
        v0 = random.uniform(0.01, 0.3)
        params.append((kappa, theta, sigma, rho, v0))
    return params


def generate_dataset(n_samples: int = 500, device: str = "cpu"):
    m_grid, t_grid = default_grid()
    S0 = 100.0
    r = 0.02
    q = 0.0

    xs = []
    ys = []
    for (kappa, theta, sigma, rho, v0) in _sample_params(n_samples):
        prices = price_grid_from_params(S0, m_grid, t_grid, r, q, (kappa, theta, sigma, rho, v0))
        iv = implied_vol_grid(prices, S0, m_grid, t_grid, r, q)
        if not np.isfinite(iv).all():
            continue
        iv_norm = (iv - IV_MEAN) / (IV_STD + EPS)
        xs.append(iv_norm.astype(np.float32))
        ys.append(np.array([kappa, theta, sigma, rho, v0], dtype=np.float32))

    if not xs:
        raise RuntimeError("Dataset vide, augmentez n_samples.")

    x_t = torch.tensor(np.stack(xs), device=device).unsqueeze(1)
    y_t = torch.tensor(np.stack(ys), device=device)
    return x_t, y_t, m_grid, t_grid


def train(n_samples: int = 500, epochs: int = 20, batch_size: int = 32, lr: float = 1e-3, device: str = "cpu"):
    x, y, m_grid, t_grid = generate_dataset(n_samples=n_samples, device=device)
    model = HestonSurfaceNet(m_size=len(m_grid), t_size=len(t_grid)).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        avg = total / max(1, len(loader))
        print(f"Epoch {epoch+1}/{epochs} - loss {avg:.6f}")

    Path(WEIGHTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Poids sauvegardés dans {WEIGHTS_PATH}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=device)
