from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from app.model.calibration.base_calibrator import SurfaceGrid


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float


class VolSurfaceModel(ABC):
    """
    Minimal unified interface for models able to produce an implied volatility surface.
    """

    model_key: str
    label: str

    @property
    @abstractmethod
    def param_bounds(self) -> Dict[str, Tuple[float, float]]:
        raise NotImplementedError

    @abstractmethod
    def implied_vol_surface(self, surface: SurfaceGrid, params: Dict[str, Any], **kwargs: Any) -> np.ndarray:
        raise NotImplementedError


__all__ = ["ParameterSpec", "VolSurfaceModel"]

