from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class CalibrationModelName(Enum):
    HESTON = "heston"
    RHESTON = "rheston"
    RBERGOMI = "rbergomi"
    VOLTERRA = "volterra"
    SABR = "sabr"


class MarketSurfaceSource(Enum):
    FILE_UPLOAD = "file_upload"
    API_PLACEHOLDER = "api_placeholder"


@dataclass
class CalibrationRequest:
    model: CalibrationModelName
    source: MarketSurfaceSource
    ticker: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    surface_path: Optional[str] = None


@dataclass
class CalibrationResult:
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None


__all__ = [
    "CalibrationModelName",
    "MarketSurfaceSource",
    "CalibrationRequest",
    "CalibrationResult",
]
