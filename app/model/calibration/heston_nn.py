from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np

try:  # optional dependency (Streamlit Cloud lean deploy)
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(exc)

# Expected weights path:
#     app/model/calibration/weights/heston_surface_net.pt
WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "heston_surface_net.pt"
TRIPLET_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "heston_param_triplet.pt"

IV_MEAN = 0.2
IV_STD = 0.1
EPS = 1e-6


if TORCH_AVAILABLE:

    class HestonSurfaceNet(nn.Module):
        """
        CNN encoder + MLP head mapping IV surface -> Heston params.
        """

        def __init__(
            self,
            m_size: int,
            t_size: int,
            hidden_units: int = 128,
            hidden_layers: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            channels = 16
            self.features = nn.Sequential(
                nn.Conv2d(1, channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
                nn.ReLU(),
            )

            feat_dim = (channels * 2) * m_size * t_size
            mlp: list[nn.Module] = []
            n_hidden = max(1, int(hidden_layers))
            for layer_idx in range(n_hidden):
                in_dim = feat_dim if layer_idx == 0 else hidden_units
                mlp.append(nn.Linear(in_dim, hidden_units))
                mlp.append(nn.ReLU())
                if dropout > 0:
                    rate = float(dropout if layer_idx == 0 else max(0.0, dropout * 0.5))
                    mlp.append(nn.Dropout(rate))

            self.mlp = nn.Sequential(*mlp)
            self.out = nn.Linear(hidden_units, 5)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.mlp(x)
            return self.out(x)

    # Differentiable Heston pricer (for price RMSE loss)
    def _heston_cf_torch(u, S0, r, q, t, kappa, theta, sigma, rho, v0):
        u = u.to(dtype=torch.complex64)
        S0 = torch.as_tensor(S0, dtype=torch.float32, device=u.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=u.device)
        q = torch.as_tensor(q, dtype=torch.float32, device=u.device)
        t = torch.as_tensor(t, dtype=torch.float32, device=u.device)
        kappa = torch.as_tensor(kappa, dtype=torch.float32, device=u.device)
        theta = torch.as_tensor(theta, dtype=torch.float32, device=u.device)
        sigma = torch.clamp(torch.as_tensor(sigma, dtype=torch.float32, device=u.device), min=1e-8)
        rho = torch.clamp(torch.as_tensor(rho, dtype=torch.float32, device=u.device), -0.999, 0.999)
        v0 = torch.as_tensor(v0, dtype=torch.float32, device=u.device)

        iu = 1j * u
        d = torch.sqrt((rho * sigma * iu - kappa) ** 2 + sigma * sigma * (iu + u * u))
        g = (kappa - rho * sigma * iu - d) / (kappa - rho * sigma * iu + d + 1e-16)
        exp_dt = torch.exp(-d * t)
        one_minus_g_exp = 1 - g * exp_dt
        one_minus_g = 1 - g
        one_minus_g_exp = torch.where(one_minus_g_exp == 0, torch.tensor(1e-16, device=u.device), one_minus_g_exp)
        one_minus_g = torch.where(one_minus_g == 0, torch.tensor(1e-16, device=u.device), one_minus_g)

        C = (r - q) * iu * t + (kappa * theta / (sigma * sigma)) * (
            (kappa - rho * sigma * iu - d) * t - 2.0 * torch.log(one_minus_g_exp / one_minus_g)
        )
        D = ((kappa - rho * sigma * iu - d) / (sigma * sigma)) * ((1 - exp_dt) / one_minus_g_exp)
        return torch.exp(C + D * v0 + iu * torch.log(S0))

    def _P_torch(j, S0, K, t, r, q, params, u_max=50.0, N=512):
        kappa, theta, sigma, rho, v0 = params
        du = float(u_max) / float(N)
        u = torch.arange(1, N + 1, device=S0.device, dtype=torch.float32) * du
        if j == 1:
            phi = _heston_cf_torch(u - 1j, S0, r, q, t, kappa, theta, sigma, rho, v0)
            numerator = torch.exp(-1j * u * torch.log(torch.as_tensor(K, device=u.device))) * phi
            denom = 1j * u * S0 * torch.exp(-q * t)
        else:
            phi = _heston_cf_torch(u, S0, r, q, t, kappa, theta, sigma, rho, v0)
            numerator = torch.exp(-1j * u * torch.log(torch.as_tensor(K, device=u.device))) * phi
            denom = 1j * u
        integrand = (numerator / denom).real
        return 0.5 + (du / np.pi) * integrand.sum()

    def call_price_cf_torch(S0, K, t, r, q, params, u_max=50.0, N=512):
        if t <= 0:
            return torch.tensor(0.0, device=S0.device, dtype=torch.float32)
        p1 = _P_torch(1, S0, K, t, r, q, params, u_max=u_max, N=N)
        p2 = _P_torch(2, S0, K, t, r, q, params, u_max=u_max, N=N)
        return S0 * torch.exp(-q * t) * p1 - torch.as_tensor(K, device=S0.device, dtype=torch.float32) * torch.exp(-r * t) * p2

    def price_grid_from_params_torch(S0, m_grid, t_grid, r, q, params, u_max=50.0, N=512):
        rows = []
        for t in t_grid:
            row = []
            for m in m_grid:
                K = float(m) * float(S0)
                row.append(call_price_cf_torch(S0, K, float(t), float(r), float(q), params, u_max=u_max, N=N))
            rows.append(torch.stack(row))
        return torch.stack(rows)

else:  # pragma: no cover

    class HestonSurfaceNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is required for the Heston Neural Net features. "
                "Install torch (or use Least Squares)."
                + (f" Import error: {TORCH_IMPORT_ERROR}" if TORCH_IMPORT_ERROR else "")
            )


def postprocess_tensor(raw: torch.Tensor) -> torch.Tensor:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")
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
    if not TORCH_AVAILABLE:
        return None, "PyTorch non installé"
    path = Path(weights_path)
    if not path.exists():
        return None, "Poids absents"
    model = HestonSurfaceNet(m_size=m_size, t_size=t_size).to(device)
    try:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Chargement impossible ({exc})"
    model.eval()
    return model, None


def predict_params(
    iv_grid: np.ndarray, m_grid: np.ndarray, t_grid: np.ndarray, weights_path: str | Path, device: str = "cpu"
) -> Dict:
    if not TORCH_AVAILABLE:
        return {
            "success": False,
            "message": "PyTorch non installé → méthode Neural Net indisponible.",
            "params": {},
        }
    if iv_grid is None or np.asarray(iv_grid).size == 0:
        return {"success": False, "message": "Surface IV vide.", "params": {}}

    iv_arr = np.asarray(iv_grid, dtype=float)
    if iv_arr.ndim != 2 or iv_arr.shape != (len(t_grid), len(m_grid)):
        return {
            "success": False,
            "message": f"Shape IV invalide: {tuple(iv_arr.shape)} attendu {(len(t_grid), len(m_grid))}.",
            "params": {},
        }

    model, load_err = load_model(weights_path, m_size=len(m_grid), t_size=len(t_grid), device=device)
    if model is None:
        return {
            "success": False,
            "message": f"Poids manquants ou incompatibles. Recréez {WEIGHTS_PATH}. ({load_err})",
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
    input_mode: str = "iv",
    return_price_targets: bool = False,
    progress: Callable[[float], None] | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    from app.model.calibration.heston_pricer import call_price_cf
    from app.model.calibration.implied_vol import implied_vol_grid

    n_samples = int(max(1, n_samples))
    rng = np.random.default_rng(seed)

    xs: list[np.ndarray] = []
    ys_params: list[np.ndarray] = []
    ys_price: list[np.ndarray] = []
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
        price_norm = prices / float(S0)

        if str(input_mode).lower() == "price":
            xs.append(price_norm.astype(np.float32))
        else:
            xs.append(iv_norm.astype(np.float32))

        ys_params.append(np.array([kappa, theta, sigma, rho, v0], dtype=np.float32))
        if return_price_targets:
            ys_price.append(price_norm.astype(np.float32))

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
    y_params = np.stack(ys_params, axis=0)
    y_price = np.stack(ys_price, axis=0) if return_price_targets else None
    return x, y_params, y_price


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
    S0_min: float | None = None,
    S0_max: float | None = None,
    r: float = 0.02,
    q: float = 0.0,
    weights_path: str | Path = WEIGHTS_PATH,
    progress: Callable[[float], None] | None = None,
    progress_epoch: Callable[[int, float, int], None] | None = None,
    val_split: float = 0.1,
    patience: int = 5,
    min_delta: float = 1e-4,
    loss_mode: str = "params",  # "params" (MSE on params) or "price_rmse" (RMSE on prices)
    input_mode: str | None = None,  # auto: iv for params, price for price_rmse
    price_u_max: float | None = None,
    price_n_integration: int | None = None,
) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        return {
            "success": False,
            "message": "PyTorch non installé → entraînement NN indisponible.",
        }
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

    loss_mode = str(loss_mode or "params").lower().strip()
    if loss_mode not in {"params", "price_rmse"}:
        loss_mode = "params"
    input_mode = str(input_mode or ("price" if loss_mode == "price_rmse" else "iv")).lower().strip()
    price_u_max = float(price_u_max if price_u_max is not None else u_max)
    price_n_integration = int(price_n_integration if price_n_integration is not None else 512)

    if loss_mode == "price_rmse" and not TORCH_AVAILABLE:
        return {"success": False, "message": "PyTorch requis pour loss 'price_rmse'.", "details": {}}

    m_grid, t_grid = default_grid()
    x_np, y_params_np, y_price_np = _generate_synthetic_dataset(
        n_samples=int(n_samples),
        m_grid=m_grid,
        t_grid=t_grid,
        S0=float(S0),
        r=float(r),
        q=float(q),
        u_max=float(u_max),
        n_integration=int(n_integration),
        seed=int(seed) if seed is not None else None,
        input_mode=input_mode,
        return_price_targets=bool(loss_mode == "price_rmse"),
        progress=progress,
    )
    if loss_mode == "price_rmse" and y_price_np is None:
        return {"success": False, "message": "Targets de prix manquants pour loss_mode=price_rmse."}

    x = torch.tensor(x_np, dtype=torch.float32, device=device_str).unsqueeze(1)
    y_params = torch.tensor(y_params_np, dtype=torch.float32, device=device_str)
    y_price = torch.tensor(y_price_np, dtype=torch.float32, device=device_str) if y_price_np is not None else None

    perm = torch.randperm(x.shape[0], device=device_str)
    x = x[perm]
    y_params = y_params[perm]
    if y_price is not None:
        y_price = y_price[perm]

    val_split = float(max(0.0, min(0.5, val_split)))
    patience = int(max(0, patience))
    num_epochs = int(max(1, epochs))

    n_val = int(x.shape[0] * val_split)
    if n_val >= x.shape[0] and x.shape[0] > 1:
        n_val = x.shape[0] - 1

    has_val = n_val > 0
    if has_val:
        x_val, y_val_params = x[:n_val], y_params[:n_val]
        y_val_price = y_price[:n_val] if y_price is not None else None
        x_train, y_train_params = x[n_val:], y_params[n_val:]
        y_train_price = y_price[n_val:] if y_price is not None else None
    else:
        x_val = y_val_params = y_val_price = None
        x_train, y_train_params = x, y_params
        y_train_price = y_price

    model = HestonSurfaceNet(m_size=len(m_grid), t_size=len(t_grid)).to(device_str)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    if loss_mode == "price_rmse":
        train_dataset = TensorDataset(x_train, y_train_price, y_train_params)  # params kept for diagnostics
        val_dataset = (
            TensorDataset(x_val, y_val_price, y_val_params) if has_val and y_val_price is not None else None
        )
    else:
        train_dataset = TensorDataset(x_train, y_train_params)
        val_dataset = TensorDataset(x_val, y_val_params) if has_val else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(max(1, batch_size)),
        shuffle=True,
        drop_last=False,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=int(max(1, batch_size)), shuffle=False, drop_last=False)
        if val_dataset is not None
        else None
    )

    if loss_mode == "price_rmse":
        m_grid_t = torch.tensor(m_grid, dtype=torch.float32, device=device_str)
        t_grid_t = torch.tensor(t_grid, dtype=torch.float32, device=device_str)
        S0_t = torch.tensor(float(S0), dtype=torch.float32, device=device_str)
        r_t = torch.tensor(float(r), dtype=torch.float32, device=device_str)
        q_t = torch.tensor(float(q), dtype=torch.float32, device=device_str)

        def _price_grid_batch(params_batch: torch.Tensor) -> torch.Tensor:
            outs: list[torch.Tensor] = []
            for i in range(params_batch.shape[0]):
                prm = params_batch[i]
                grid = price_grid_from_params_torch(
                    S0_t, m_grid_t, t_grid_t, r_t, q_t, tuple(prm), u_max=price_u_max, N=price_n_integration
                )
                outs.append(grid / S0_t)  # normalise par S0 pour la stabilité
            return torch.stack(outs)

    history: list[Dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    model.train()
    for epoch in range(num_epochs):
        running = 0.0
        seen = 0
        for batch in train_loader:
            if loss_mode == "price_rmse":
                xb, yb_price, _yb_params = batch
            else:
                xb, yb = batch
            opt.zero_grad(set_to_none=True)
            preds_raw = model(xb)
            preds = postprocess_tensor(preds_raw)
            if loss_mode == "price_rmse":
                price_pred = _price_grid_batch(preds)
                loss = torch.sqrt(torch.mean((price_pred - yb_price) ** 2))
            else:
                loss = loss_fn(preds, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * int(xb.shape[0])
            seen += int(xb.shape[0])

        train_loss = float(running / max(1, seen))

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                v_running = 0.0
                v_seen = 0
                for batch in val_loader:
                    if loss_mode == "price_rmse":
                        xb, yb_price, _yb_params = batch
                    else:
                        xb, yb = batch
                    preds = postprocess_tensor(model(xb))
                    if loss_mode == "price_rmse":
                        price_pred = _price_grid_batch(preds)
                        v_loss = torch.sqrt(torch.mean((price_pred - yb_price) ** 2))
                    else:
                        v_loss = loss_fn(preds, yb)
                    v_running += float(v_loss.item()) * int(xb.shape[0])
                    v_seen += int(xb.shape[0])
            val_loss = float(v_running / max(1, v_seen))
            model.train()

        history.append({"epoch": int(epoch + 1), "train_loss": train_loss, "val_loss": val_loss})

        if progress is not None:
            try:
                progress(min(0.95 + 0.05 * float(epoch + 1) / float(max(1, num_epochs)), 1.0))
            except Exception:
                pass
        if progress_epoch is not None:
            try:
                progress_epoch(int(epoch + 1), float(train_loss), int(num_epochs))
            except Exception:
                pass

        improved = (val_loss + float(min_delta)) < best_loss
        if improved or best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_loss = val_loss
            best_epoch = int(epoch + 1)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            break

    out_path = Path(weights_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), out_path)

    elapsed = float(time.time() - started)
    final_loss = float(best_loss if best_state is not None else history[-1]["val_loss"]) if history else None
    if progress is not None:
        try:
            progress(1.0)
        except Exception:
            pass
    return {
        "success": True,
        "message": "OK",
        "weights_path": str(out_path),
        "device": device_str,
        "n_samples": int(x.shape[0]),
        "epochs": int(len(history)),
        "epochs_requested": int(num_epochs),
        "final_loss": final_loss,
        "best_val_loss": final_loss,
        "loss_history": [float(h["val_loss"]) for h in history],
        "train_loss_history": [float(h["train_loss"]) for h in history],
        "best_epoch": int(best_epoch or len(history)),
        "patience": int(patience),
        "val_split": float(val_split if has_val else 0.0),
        "lr": float(lr),
        "elapsed_s": elapsed,
        "grid": {"m_size": int(len(m_grid)), "t_size": int(len(t_grid))},
        "loss_mode": loss_mode,
        "input_mode": input_mode,
    }


__all__ = [
    "HestonSurfaceNet",
    "WEIGHTS_PATH",
    "TORCH_AVAILABLE",
    "TORCH_IMPORT_ERROR",
    "load_model",
    "postprocess_tensor",
    "predict_params",
    "train_heston_surface_net",
]


# ---------------------------------------------------------------------------
# New lightweight MLP: (S0, K, T) -> Heston params, trained via price RMSE
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class HestonParamMLP(nn.Module):
        """
        Petit MLP qui prend (S0, K, T) normalisés et prédit les paramètres Heston.
        La loss se calcule via le pricer torch et une RMSE sur les prix.
        """

        def __init__(self, hidden_units: int = 128, hidden_layers: int = 3, dropout: float = 0.0):
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = 3  # S0_norm, moneyness, T
            for i in range(max(1, hidden_layers)):
                layers.append(nn.Linear(in_dim if i == 0 else hidden_units, hidden_units))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(float(dropout)))
            layers.append(nn.Linear(hidden_units, 5))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


def _sample_heston_params(rng: np.random.Generator) -> tuple[float, float, float, float, float]:
    kappa = float(rng.uniform(0.5, 5.0))
    theta = float(rng.uniform(0.02, 0.3))
    sigma = float(rng.uniform(0.1, 1.0))
    rho = float(rng.uniform(-0.9, 0.3))
    v0 = float(rng.uniform(0.01, 0.3))
    return kappa, theta, sigma, rho, v0


def _normalize_triplet(S0: float, K: float, T: float) -> np.ndarray:
    # On encode S0, moneyness (K/S0) et maturité T
    m = float(K) / float(S0) if S0 > 0 else 1.0
    return np.array([float(S0), m, float(T)], dtype=np.float32)


def train_heston_param_net_from_prices(
    *,
    n_samples: int = 5000,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int | None = 42,
    S0: float = 100.0,
    S0_min: float | None = None,  # ignoré (API legacy)
    S0_max: float | None = None,  # ignoré (API legacy)
    r: float = 0.02,
    q: float = 0.0,
    K_min: float = 0.5,
    K_max: float = 1.5,
    T_min: float = 0.05,
    T_max: float = 2.0,
    u_max: float = 50.0,
    n_integration: int = 512,
    dropout: float = 0.0,
    hidden_units: int = 128,
    hidden_layers: int = 3,
    progress_epoch: Callable[[int, float, int], None] | None = None,
    weights_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Entraîne un MLP simple qui mappe (S0, K, T) -> params Heston en minimisant la RMSE prix.
    """
    return {
        "success": False,
        "message": "Désactivé: paramètres Heston globaux uniquement (pas de modèle par point).",
        "details": {},
    }
    if not TORCH_AVAILABLE:
        return {"success": False, "message": "PyTorch requis pour l'entraînement.", "details": {}}

    from app.model.calibration.heston_pricer import call_price_cf

    rng = np.random.default_rng(seed)
    device_str = str(device or "cpu").lower().strip()
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    pxs: list[float] = []

    for _ in range(int(max(1, n_samples))):
        params = _sample_heston_params(rng)
        # S0 fixe = spot fourni
        S0_draw = float(S0)
        K = float(rng.uniform(K_min, K_max) * S0_draw)
        T = float(rng.uniform(T_min, T_max))
        price = call_price_cf(float(S0_draw), float(K), float(T), float(r), float(q), params, u_max=float(u_max), N=int(n_integration))
        xs.append(_normalize_triplet(S0_draw, K, T))
        ys.append(np.array(params, dtype=np.float32))
        pxs.append(float(price))

    X = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32, device=device_str)
    Y_params = torch.tensor(np.stack(ys, axis=0), dtype=torch.float32, device=device_str)
    Y_price = torch.tensor(pxs, dtype=torch.float32, device=device_str)

    model = HestonParamMLP(hidden_units=hidden_units, hidden_layers=hidden_layers, dropout=dropout).to(device_str)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    dataset = torch.utils.data.TensorDataset(X, Y_params, Y_price)
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(max(1, batch_size)), shuffle=True, drop_last=False)

    history: list[float] = []
    for epoch in range(int(max(1, epochs))):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb_params, yb_price in loader:
            opt.zero_grad(set_to_none=True)
            raw = model(xb)
            preds = postprocess_tensor(raw)
            # Prix prédit via pricer torch (normalisé S0 constant ici)
            price_pred = []
            for i in range(preds.shape[0]):
                prm = preds[i]
                S0_t = xb[i, 0]
                K_t = xb[i, 1] * xb[i, 0]  # moneyness * S0
                T_t = xb[i, 2]
                price_pred.append(
                    call_price_cf_torch(S0_t, K_t, T_t, float(r), float(q), tuple(prm), u_max=float(u_max), N=int(n_integration))
                )
            price_pred_t = torch.stack(price_pred)
            loss = torch.sqrt(torch.mean((price_pred_t - yb_price) ** 2))
            loss.backward()
            opt.step()
            running += float(loss.item()) * int(xb.shape[0])
            seen += int(xb.shape[0])
        epoch_loss = float(running / max(1, seen))
        history.append(epoch_loss)
        if progress_epoch is not None:
            try:
                progress_epoch(int(epoch + 1), float(epoch_loss), int(epochs))
            except Exception:
                pass

    out_path = Path(weights_path or TRIPLET_WEIGHTS_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)

    return {
        "success": True,
        "message": "OK",
        "epochs": int(epochs),
        "loss_history": history,
        "final_loss": history[-1] if history else None,
        "device": device_str,
        "model_state": model.state_dict(),
        "weights_path": str(out_path),
    }


def predict_params_from_triplet(model_state: Dict[str, Any], S0: float, K: float, T: float, device: str = "cpu") -> Dict[str, float]:
    """
    Inférence pour le MLP (S0, K, T) -> params Heston.
    """
    return {"success": False, "message": "Prédicteur par point désactivé (paramètres globaux uniquement).", "params": {}}
    if not TORCH_AVAILABLE:
        return {"success": False, "message": "PyTorch non installé.", "params": {}}
    device_str = str(device or "cpu").lower().strip()
    mdl = HestonParamMLP()
    mdl.load_state_dict(model_state)
    mdl.to(device_str)
    mdl.eval()
    x = torch.tensor(_normalize_triplet(S0, K, T), dtype=torch.float32, device=device_str).unsqueeze(0)
    with torch.no_grad():
        raw = mdl(x)[0]
        params = _postprocess(raw)
    return {"success": True, "params": params}


def predict_params_from_triplet_weights(weights_path: str | Path, S0: float, K: float, T: float, device: str = "cpu") -> Dict[str, float]:
    return {"success": False, "message": "Prédicteur par point désactivé (paramètres globaux uniquement).", "params": {}}
    if not TORCH_AVAILABLE:
        return {"success": False, "message": "PyTorch non installé.", "params": {}}
    path = Path(weights_path)
    if not path.exists():
        return {"success": False, "message": f"Poids introuvables: {path}", "params": {}}
    mdl = HestonParamMLP()
    state = torch.load(path, map_location=device)
    mdl.load_state_dict(state)
    mdl.to(device)
    mdl.eval()
    x = torch.tensor(_normalize_triplet(S0, K, T), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        raw = mdl(x)[0]
        params = _postprocess(raw)
    return {"success": True, "params": params, "weights_path": str(path)}
