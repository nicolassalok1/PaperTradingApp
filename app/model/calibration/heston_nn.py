from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

# Expected weights path:
#     app/model/calibration/weights/heston_surface_net.pt
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "heston_surface_net.pt"

IV_MEAN = 0.2
IV_STD = 0.1
EPS = 1e-6


class HestonSurfaceNet(nn.Module):
    """
    Simple CNN regressor mapping IV surface -> Heston params.
    """

    def __init__(self, m_size: int, t_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * m_size * t_size, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.net(x)


def _postprocess(raw: torch.Tensor) -> Dict[str, float]:
    kappa = torch.nn.functional.softplus(raw[0]) + EPS
    theta = torch.nn.functional.softplus(raw[1]) + EPS
    sigma = torch.nn.functional.softplus(raw[2]) + EPS
    rho = torch.tanh(raw[3]) * 0.999
    v0 = torch.nn.functional.softplus(raw[4]) + EPS
    return {
        "kappa": float(kappa.item()),
        "theta": float(theta.item()),
        "sigma": float(sigma.item()),
        "rho": float(rho.item()),
        "v0": float(v0.item()),
    }


def load_model(weights_path: str | Path, m_size: int, t_size: int, device: str = "cpu"):
    path = Path(weights_path)
    if not path.exists():
        return None
    model = HestonSurfaceNet(m_size=m_size, t_size=t_size).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_params(
    iv_grid: np.ndarray, m_grid: np.ndarray, t_grid: np.ndarray, weights_path: str | Path, device: str = "cpu"
) -> Dict:
    if iv_grid is None or iv_grid.size == 0:
        return {"success": False, "message": "Surface IV vide.", "params": {}}
    model = load_model(weights_path, m_size=len(m_grid), t_size=len(t_grid), device=device)
    if model is None:
        return {
            "success": False,
            "message": f"Poids manquants. Placez un fichier à {WEIGHTS_PATH}",
            "params": {},
        }
    with torch.no_grad():
        iv_norm = (iv_grid - IV_MEAN) / (IV_STD + EPS)
        tensor = torch.tensor(iv_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        out = model(tensor)[0]
        params = _postprocess(out)
    return {"success": True, "message": "OK", "params": params}


__all__ = ["HestonSurfaceNet", "predict_params", "load_model", "WEIGHTS_PATH"]
