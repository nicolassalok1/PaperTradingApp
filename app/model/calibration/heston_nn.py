from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

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


def postprocess_tensor(raw: torch.Tensor) -> torch.Tensor:
    kappa = torch.nn.functional.softplus(raw[..., 0]) + EPS
    theta = torch.nn.functional.softplus(raw[..., 1]) + EPS
    sigma = torch.nn.functional.softplus(raw[..., 2]) + EPS
    rho = torch.tanh(raw[..., 3]) * 0.999
    v0 = torch.nn.functional.softplus(raw[..., 4]) + EPS
    return torch.stack([kappa, theta, sigma, rho, v0], dim=-1)


def _postprocess(raw: torch.Tensor) -> Dict[str, float]:
    out = postprocess_tensor(raw)
    kappa, theta, sigma, rho, v0 = [float(x) for x in out.detach().cpu().numpy().tolist()]
    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "rho": float(rho),
        "v0": float(v0),
    }


def _fill_nan_surface(iv_grid: np.ndarray) -> np.ndarray:
    arr = np.asarray(iv_grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError("iv_grid doit être une matrice 2D.")

    out = arr.copy()
    out[~np.isfinite(out)] = np.nan
    out[out <= 0] = np.nan

    if not np.isfinite(out).any():
        return np.full_like(out, float(IV_MEAN), dtype=float)

    global_med = float(np.nanmedian(out))

    try:
        row_med = np.nanmedian(out, axis=1)
        for i in range(out.shape[0]):
            if not np.isnan(out[i]).any():
                continue
            fill = float(row_med[i]) if np.isfinite(row_med[i]) else global_med
            out[i] = np.where(np.isfinite(out[i]), out[i], fill)
    except Exception:
        out = np.where(np.isfinite(out), out, global_med)

    out = np.where(np.isfinite(out), out, global_med)
    return np.clip(out, 1e-4, 5.0).astype(float)


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
    if iv_grid is None or np.asarray(iv_grid).size == 0:
        return {"success": False, "message": "Surface IV vide.", "params": {}}

    iv_arr = np.asarray(iv_grid, dtype=float)
    if iv_arr.ndim != 2 or iv_arr.shape != (len(t_grid), len(m_grid)):
        return {
            "success": False,
            "message": f"Shape IV invalide: {tuple(iv_arr.shape)} attendu {(len(t_grid), len(m_grid))}.",
            "params": {},
        }

    model = load_model(weights_path, m_size=len(m_grid), t_size=len(t_grid), device=device)
    if model is None:
        return {
            "success": False,
            "message": f"Poids manquants. Placez un fichier à {WEIGHTS_PATH}",
            "params": {},
        }
    with torch.no_grad():
        iv_filled = _fill_nan_surface(iv_arr)
        iv_norm = (iv_filled - IV_MEAN) / (IV_STD + EPS)
        tensor = torch.tensor(iv_norm, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        out = model(tensor)[0]
        params = _postprocess(out)
    return {"success": True, "message": "OK", "params": params}


def _generate_synthetic_dataset(
    *,
    n_samples: int,
    m_grid: np.ndarray,
    t_grid: np.ndarray,
    S0: float,
    r: float,
    q: float,
    u_max: float,
    n_integration: int,
    seed: int | None,
    min_finite_ratio: float = 0.35,
    progress: Callable[[float], None] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    from app.model.calibration.heston_pricer import call_price_cf
    from app.model.calibration.implied_vol import implied_vol_grid

    n_samples = int(max(1, n_samples))
    rng = np.random.default_rng(seed)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    max_attempts = max(n_samples * 3, 200)
    for attempt in range(max_attempts):
        kappa = float(rng.uniform(0.5, 5.0))
        theta = float(rng.uniform(0.02, 0.3))
        sigma = float(rng.uniform(0.1, 1.0))
        rho = float(rng.uniform(-0.9, 0.3))
        v0 = float(rng.uniform(0.01, 0.3))
        params = (kappa, theta, sigma, rho, v0)

        prices = np.zeros((len(t_grid), len(m_grid)), dtype=float)
        for i_t, t in enumerate(t_grid):
            t_val = float(t)
            for j_m, m in enumerate(m_grid):
                K = float(float(m) * float(S0))
                prices[i_t, j_m] = call_price_cf(
                    float(S0),
                    K,
                    t_val,
                    float(r),
                    float(q),
                    params,
                    u_max=float(u_max),
                    N=int(n_integration),
                )

        iv = implied_vol_grid(prices, float(S0), m_grid, t_grid, float(r), float(q))
        finite_ratio = float(np.isfinite(iv).mean()) if iv.size else 0.0
        if finite_ratio < float(min_finite_ratio):
            continue

        iv_filled = _fill_nan_surface(iv)
        iv_norm = (iv_filled - IV_MEAN) / (IV_STD + EPS)
        xs.append(iv_norm.astype(np.float32))
        ys.append(np.array([kappa, theta, sigma, rho, v0], dtype=np.float32))

        if progress is not None:
            try:
                progress(min(0.95, float(len(xs)) / float(n_samples)))
            except Exception:
                pass

        if len(xs) >= n_samples:
            break

        if attempt % 250 == 0 and attempt > 0 and progress is not None:
            try:
                progress(min(0.95, float(len(xs)) / float(n_samples)))
            except Exception:
                pass

    if not xs:
        raise RuntimeError("Dataset vide: augmentez n_samples ou ajustez les bornes de sampling.")

    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    return x, y


def train_heston_surface_net(
    *,
    n_samples: int = 2000,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int | None = 42,
    u_max: float = 50.0,
    n_integration: int = 800,
    S0: float = 100.0,
    r: float = 0.02,
    q: float = 0.0,
    weights_path: str | Path = WEIGHTS_PATH,
    progress: Callable[[float], None] | None = None,
) -> Dict[str, Any]:
    from torch.utils.data import DataLoader, TensorDataset

    from app.model.calibration.market_surface import default_grid

    started = time.time()
    device_str = str(device or "cpu").lower().strip()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"

    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    m_grid, t_grid = default_grid()
    x_np, y_np = _generate_synthetic_dataset(
        n_samples=int(n_samples),
        m_grid=m_grid,
        t_grid=t_grid,
        S0=float(S0),
        r=float(r),
        q=float(q),
        u_max=float(u_max),
        n_integration=int(n_integration),
        seed=int(seed) if seed is not None else None,
        progress=progress,
    )

    x = torch.tensor(x_np, dtype=torch.float32, device=device_str).unsqueeze(1)
    y = torch.tensor(y_np, dtype=torch.float32, device=device_str)

    model = HestonSurfaceNet(m_size=len(m_grid), t_size=len(t_grid)).to(device_str)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=int(max(1, batch_size)),
        shuffle=True,
        drop_last=False,
    )

    history: list[float] = []
    model.train()
    for epoch in range(int(max(1, epochs))):
        running = 0.0
        seen = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            preds = postprocess_tensor(model(xb))
            loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * int(xb.shape[0])
            seen += int(xb.shape[0])

        avg = float(running / max(1, seen))
        history.append(avg)
        if progress is not None:
            try:
                progress(0.95 + 0.05 * float(epoch + 1) / float(max(1, int(epochs))))
            except Exception:
                pass

    out_path = Path(weights_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    elapsed = float(time.time() - started)
    return {
        "success": True,
        "message": "OK",
        "weights_path": str(out_path),
        "device": device_str,
        "n_samples": int(x.shape[0]),
        "epochs": int(epochs),
        "final_loss": float(history[-1]) if history else None,
        "loss_history": history,
        "elapsed_s": elapsed,
        "grid": {"m_size": int(len(m_grid)), "t_size": int(len(t_grid))},
    }


__all__ = [
    "HestonSurfaceNet",
    "WEIGHTS_PATH",
    "load_model",
    "postprocess_tensor",
    "predict_params",
    "train_heston_surface_net",
]
