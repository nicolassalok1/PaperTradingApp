from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class SurfaceGrid:
    """
    Canonical representation of a market IV surface on a fixed (T, moneyness) grid.
    - m_grid: K/S0
    - t_grid: year fractions
    - iv_market: shape (len(t_grid), len(m_grid))
    - mask: True where market observations exist (optional)
    """

    S0: float
    r: float
    q: float
    m_grid: np.ndarray
    t_grid: np.ndarray
    iv_market: np.ndarray
    mask: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CalibratorSettings:
    fit_to_observed_only: bool = True
    max_nfev: int = 80
    n_starts: int = 1
    seed: int | None = None


@dataclass
class SurfaceCalibrationResult:
    success: bool
    message: str
    model: str
    method: str
    params: Dict[str, Any]
    metrics: Dict[str, float] | None = None
    metrics_vw: Dict[str, float] | None = None
    iv_model: np.ndarray | None = None
    iv_error: np.ndarray | None = None
    vega_weights: np.ndarray | None = None
    details: Dict[str, Any] | None = None


def apply_degeneracy_guard(result: "SurfaceCalibrationResult") -> "SurfaceCalibrationResult":
    """Flip a falsely-successful calibration to an explicit failure.

    A model that returns success=True while producing an entirely non-finite
    (all-NaN/inf) IV surface, or non-finite error metrics, is degenerate: the UI
    would show blank heatmaps while claiming success. Detect that and set
    success=False with a clear message. Mutates and returns `result`.

    Does NOT flag a partially-NaN surface (e.g. SABR with unobserved maturities)
    as long as at least one model cell is finite and the metrics are finite.
    """
    if not result.success:
        return result

    iv = result.iv_model
    iv_arr = np.asarray(iv, dtype=float) if iv is not None else None
    all_nan = iv_arr is None or iv_arr.size == 0 or not np.isfinite(iv_arr).any()

    metrics = result.metrics or {}
    metric_vals = [
        float(v)
        for k, v in metrics.items()
        if k != "n" and isinstance(v, (int, float))
    ]
    metrics_bad = bool(metric_vals) and any(not np.isfinite(v) for v in metric_vals)

    if all_nan or metrics_bad:
        result.success = False
        result.message = f"Calibration dégénérée: surface modèle non finie (NaN). [{result.message}]"
    return result


class BaseSurfaceCalibrator(ABC):
    """
    Unified calibration interface (model layer).
    Controllers should only route requests + pass inputs; all numerics stay here.
    """

    model: str
    method: str

    @abstractmethod
    def calibrate(
        self,
        surface: SurfaceGrid,
        *,
        constraints: Dict[str, Any] | None = None,
        settings: CalibratorSettings | None = None,
    ) -> SurfaceCalibrationResult:
        raise NotImplementedError

