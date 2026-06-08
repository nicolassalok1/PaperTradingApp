"""
Offline training script for HestonSurfaceNet.
Generates synthetic IV surfaces, trains the CNN, and saves weights to:
    app/model/calibration/weights/heston_surface_net.pt
Run manually; not used in the UI.
"""

from __future__ import annotations

import torch

from app.model.calibration.heston_nn import WEIGHTS_PATH, train_heston_surface_net


def train(
    n_samples: int = 500,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    val_split: float = 0.1,
    patience: int = 5,
):
    res = train_heston_surface_net(
        n_samples=n_samples,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        val_split=val_split,
        patience=patience,
        weights_path=WEIGHTS_PATH,
    )
    if not res.get("success"):
        raise RuntimeError(f"Entraînement NN échoué: {res.get('message')}")

    best_loss = res.get("best_val_loss")
    print(
        f"Fin entraînement Heston NN → loss={best_loss} | epochs={res.get('epochs')} "
        f"(demandé={res.get('epochs_requested')}) | device={res.get('device')}"
    )
    print(f"Weights saved to: {res.get('weights_path')}")
    return res


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=device)
