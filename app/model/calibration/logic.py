from __future__ import annotations

from typing import List, Tuple

from app.model.calibration.types import (
    CalibrationModelName,
    CalibrationRequest,
    CalibrationResult,
    MarketSurfaceSource,
)


def get_supported_models() -> List[CalibrationModelName]:
    """Return available calibration models (placeholders)."""
    return [
        CalibrationModelName.HESTON,
        CalibrationModelName.RHESTON,
        CalibrationModelName.RBERGOMI,
        CalibrationModelName.VOLTERRA,
        CalibrationModelName.SABR,
    ]


def validate_request(req: CalibrationRequest) -> Tuple[bool, str]:
    """Basic validation stub."""
    if req.model not in get_supported_models():
        return False, "Modèle non supporté."
    if req.source not in (MarketSurfaceSource.FILE_UPLOAD, MarketSurfaceSource.API_PLACEHOLDER):
        return False, "Source de surface invalide."
    return True, "OK"


def run_calibration(req: CalibrationRequest) -> CalibrationResult:
    """
    Placeholder calibration runner.
    """
    return CalibrationResult(
        success=False,
        message="Calibration non implémentée (scaffold uniquement).",
        details={"model": req.model.value, "source": req.source.value},
    )


__all__ = ["get_supported_models", "validate_request", "run_calibration"]
