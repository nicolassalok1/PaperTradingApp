"""
Heston "NN" Calibration (safe fallback)
Originally planned for a small MLP; replaced with a deterministic heuristic to
avoid TensorFlow shape issues in Streamlit. Computes rough params from the IV surface.
"""

import numpy as np
from app.model.heston.calibration.heston_stub import calibrate_heston_stub


def prepare_nn_dataset(strikes, maturities, iv_matrix):
    """
    Prepare (X,y) dataset:
    X = [K, T]
    y = IV
    """
    Ks, Ts = np.meshgrid(strikes, maturities, indexing="xy")
    X = np.column_stack([Ks.flatten(), Ts.flatten()])
    y = iv_matrix.flatten()
    mask = np.isfinite(y)
    return X[mask], y[mask]


def calibrate_heston_nn_IV(strikes, maturities, iv_matrix, epochs: int = 40, **kwargs):
    """
    Delegates to the TensorFlow-free stub to avoid graph errors.
    """
    return calibrate_heston_stub(strikes, maturities, iv_matrix, epochs=epochs, **kwargs)


__all__ = [
    "prepare_nn_dataset",
    "calibrate_heston_nn_IV",
]
