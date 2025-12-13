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
    iv_model: np.ndarray | None = None
    iv_error: np.ndarray | None = None
    details: Dict[str, Any] | None = None


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

